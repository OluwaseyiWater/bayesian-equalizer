
from dataclasses import dataclass
from typing import List
@dataclass
class Tap:
    delay: int
    mean: complex
    var: float

@dataclass
class State:
    taps: List[Tap]
    lengthscale: float = 10.0
    sigma2: float = 0.01
