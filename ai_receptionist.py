import json
from groq import AsyncGroq
import os
from state_manager import StateManager, State
from jinja2 import Environment, FileSystemLoader
from vector_db import VectorDB
import torch

class AIReceptionist:
    def __init__(self):
        self.state_manager = StateManager()
        self.client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        self.conversation_history = []
        self.jinja_env = Environment(loader=FileSystemLoader('templates/prompts'))
        self.vector_db = VectorDB()
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AIReceptionist using device: {self.device}")

    async def process_input(self, user_input: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_input})
        
        response = await self.generate_response(user_input)
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response

    async def generate_response(self, user_input: str) -> str:
        try:
            system_prompt = self.jinja_env.get_template('system_prompt.j2').render(
                current_state=self.state_manager.state,
                context=self.state_manager.get_context()
            )

            messages = [
                {"role": "system", "content": system_prompt},
                *self.conversation_history,
                {"role": "user", "content": user_input},
            ]

            response = await self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                device=self.device  # Use the determined device
            )

            function_args = json.loads(response.choices[0].message.function_call.arguments)
            
            new_state = State[function_args["new_state"]]
            self.state_manager.transition_to(new_state)
            
            if "context_updates" in function_args:
                self.state_manager.update_context(**function_args["context_updates"])

            if new_state == State.EMERGENCY:
                emergency_type = function_args.get("context_updates", {}).get("emergency_type")
                if emergency_type:
                    instructions = await self.get_instructions_from_db(emergency_type)
                    function_args["response"] += f"\n\nHere are some first aid instructions:\n{instructions}"

            return function_args["response"]

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return "I'm sorry, an error occurred. Could you please try again?"

    def get_state_context(self) -> dict:
        return self.state_manager.get_context()

    async def get_instructions_from_db(self, query: str) -> str:
        search_result = self.vector_db.search(query)
        if search_result:
            return search_result[0].payload['response']
        return "I'm sorry, I couldn't find any specific instructions for that situation."

