from enum import Enum
from typing import Optional, Dict, Any

class State(Enum):
    INITIAL = 0
    EMERGENCY = 1
    MESSAGE = 2
    LOCATION = 3
    FINAL = 4

class StateManager:
    def __init__(self):
        self.current_state: State = State.INITIAL
        self.context: Dict[str, Any] = {}

    def transition_to(self, new_state: State):
        self.current_state = new_state

    def update_context(self, **kwargs):
        self.context.update(kwargs)

    def reset(self):
        self.__init__()

    @property
    def state(self) -> State:
        return self.current_state

    def get_context(self) -> dict:
        return {
            "state": self.current_state,
            **self.context
        }