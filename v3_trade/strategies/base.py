from dataclasses import dataclass
from typing import Literal

@dataclass
class Vote:
    strategy_id: str
    side: Literal["LONG", "SHORT", "FLAT"]
    confidence: float
    regime_alignment: bool

class BaseStrategy:
    def __init__(self):
        self.strategy_id = self.__class__.__name__
        
    def evaluate(self, data_dict, current_regime) -> Vote:
        raise NotImplementedError("Strategies must implement the evaluate method.")
