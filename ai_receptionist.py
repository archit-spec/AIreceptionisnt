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
        self.state_history = []
        
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
                context=self.state_manager.get_context(),
                state_history=self.state_history
            )

            messages = [
                {"role": "system", "content": system_prompt},
                *self.conversation_history,
                {"role": "user", "content": user_input},
            ]

            print(f"Current state: {self.state_manager.state}")
            print(f"Current context: {self.state_manager.get_context()}")
            print(f"State history: {self.state_history}")
            print(f"User input: {user_input}")
            print("Sending request to Groq API...")
            
            response = await self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                response_format={"type": "json_object"}
            )
            print("Received response from Groq API")

            if not response.choices or not response.choices[0].message:
                raise Exception("No response received from the API")

            ai_response = response.choices[0].message.content
            print(f"Raw AI response: {ai_response}")

            # Parse the AI response
            try:
                parsed_response = json.loads(ai_response)
                print(f"Parsed AI response: {json.dumps(parsed_response, indent=2)}")
                
                new_state = State[parsed_response.get("new_state", "INITIAL")]
                state_transition_message = self.state_manager.transition_to(new_state)
                print(state_transition_message)
                self.state_history.append(new_state.name)
                
                if "context_updates" in parsed_response:
                    context_update_message = self.state_manager.update_context(**parsed_response["context_updates"])
                    print(context_update_message)

                if new_state == State.EMERGENCY:
                    emergency_type = parsed_response.get("context_updates", {}).get("emergency_type")
                    if emergency_type:
                        print(f"Emergency type detected: {emergency_type}")
                        db_result = await self.get_instructions_from_db(emergency_type)
                        print(f"Database result: {db_result}")
                        if db_result["source"] == "vector_db":
                            parsed_response["response"] += f"\n\nHere are some first aid instructions for {db_result['tag']} (retrieved from my knowledge base):\n{db_result['response']}"
                            parsed_response["response"] += f"\n(Confidence score: {db_result['score']:.2f})"
                        else:
                            parsed_response["response"] += f"\n\n{db_result['response']}"

                final_response = parsed_response.get("response", "I'm sorry, I couldn't generate a proper response.")
                print(f"Final response: {final_response}")
                return final_response

            except json.JSONDecodeError:
                print(f"Failed to parse AI response as JSON. Returning raw response.")
                return ai_response  # Return the raw response if it's not in the expected JSON format

        except Exception as e:
            print(f"An error occurred in generate_response: {str(e)}")
            return "I'm sorry, an error occurred while processing your request. Please try again later."

    def get_state_context(self) -> dict:
        return self.state_manager.get_context()

    async def get_instructions_from_db(self, query: str) -> dict:
        search_result = self.vector_db.search(query)
        if search_result:
            return search_result
        return {
            "source": "fallback",
            "response": "I'm sorry, I couldn't find any specific instructions for that situation in my database."
        }

