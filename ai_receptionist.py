import random
from groq import AsyncGroq
import os
import json
from state_manager import StateManager, State
from jinja2 import Environment, FileSystemLoader

class AIReceptionist:
    def __init__(self):
        self.state_manager = StateManager()
        self.client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        self.conversation_history = []
        self.jinja_env = Environment(loader=FileSystemLoader('templates/prompts'))

    async def process_input(self, user_input: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_input})
        
        response = await self.generate_response(user_input)
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response

    async def generate_response(self, user_input: str) -> str:
        try:
            system_prompt = self.jinja_env.get_template('system_prompt.j2').render(
                current_state=self.state_manager.state,
                emergency_type=self.state_manager.emergency_type,
                location=self.state_manager.location,
                message=self.state_manager.message
            )

            state_template_name = f'{self.state_manager.state.name.lower()}_state.j2'
            try:
                state_prompt = self.jinja_env.get_template(state_template_name).render(
                    user_input=user_input,
                    emergency_type=self.state_manager.emergency_type
                )
            except jinja2.exceptions.TemplateNotFound:
                print(f"Template not found: {state_template_name}")
                state_prompt = f"User input: '{user_input}'"

            messages = [
                {"role": "system", "content": system_prompt},
                *self.conversation_history,
                {"role": "user", "content": user_input},
                {"role": "user", "content": state_prompt}
            ]

            response = await self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                max_tokens=150
            )

            ai_response = response.choices[0].message.content
            try:
                function_args = json.loads(ai_response)
                
                if self.state_manager.state == State.INITIAL:
                    if function_args.get("is_emergency"):
                        self.state_manager.transition_to_emergency(function_args.get("emergency_type", ""))
                    elif function_args.get("is_message"):
                        self.state_manager.transition_to_message(function_args.get("message", ""))
                elif self.state_manager.state == State.EMERGENCY:
                    self.state_manager.transition_to_location("")
                elif self.state_manager.state == State.LOCATION:
                    location = function_args.get("location", user_input)
                    eta = function_args.get("eta", 15)  # Default to 15 minutes if not provided
                    instructions = function_args.get("instructions", "Please stay calm and apply pressure to the wound.")
                    
                    response = (f"I understand your location is {location}. Dr. Adrin will be there in approximately "
                                f"{eta} minutes. In the meantime, here are some instructions: {instructions}")
                    
                    self.state_manager.transition_to_location(location)
                    self.state_manager.transition_to_final()
                    return response

                return function_args.get("response", "I'm sorry, I don't understand. Could you please repeat that?")
            except json.JSONDecodeError:
                if self.state_manager.state == State.LOCATION:
                    # If we can't parse the JSON, assume the entire input is the location
                    location = user_input
                    eta = 15  # Default ETA
                    instructions = "Please stay calm and apply pressure to the wound."
                    
                    response = (f"I understand your location is {location}. Dr. Adrin will be there in approximately "
                                f"{eta} minutes. In the meantime, here are some instructions: {instructions}")
                    
                    self.state_manager.transition_to_location(location)
                    self.state_manager.transition_to_final()
                    return response
                else:
                    return "I'm sorry, I don't understand. Could you please repeat that?"
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return "I'm sorry, an error occurred. Please try again later."

    def get_state_context(self) -> dict:
        return self.state_manager.get_context()
    async def get_instructions(self, emergency_type: str) -> str:
        instructions_prompt = self.jinja_env.get_template('instructions.j2').render(
            emergency_type=emergency_type
        )

        response = await self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a medical expert providing emergency first aid instructions."},
                {"role": "user", "content": instructions_prompt}
            ],
            max_tokens=150
        )

        return response.choices[0].message.content

