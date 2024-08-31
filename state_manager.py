from enum import Enum
from typing import Optional

class State(Enum):
    INITIAL = 0
    EMERGENCY = 1
    MESSAGE = 2
    LOCATION = 3
    FINAL = 4

class StateManager:
    def __init__(self):
        self.current_state: State = State.INITIAL
        self.emergency_type: Optional[str] = None
        self.location: Optional[str] = None
        self.message: Optional[str] = None

    def transition_to_emergency(self, emergency_type: str):
        self.current_state = State.EMERGENCY
        self.emergency_type = emergency_type

    def transition_to_message(self, message: str):
        self.current_state = State.MESSAGE
        self.message = message

    def transition_to_location(self, location: str):
        self.current_state = State.LOCATION
        self.location = location

    def transition_to_final(self):
        self.current_state = State.FINAL

    def reset(self):
        self.__init__()

    @property
    def state(self) -> State:
        return self.current_state

    def get_context(self) -> dict:
        return {
            "state": self.current_state,
            "emergency_type": self.emergency_type,
            "location": self.location,
            "message": self.message
        }