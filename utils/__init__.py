
from typing import TypedDict, Callable
from .utils_math import ktoi, itok
from .utils_math import combine_csm, expand_csm
from .utils_math import setup_seed

__all__ = ['setup_seed', 'A']

class Ops(TypedDict):
    ktoi: Callable
    itok: Callable
    combine_csm: Callable
    expand_csm: Callable


OPS_REGISTRY: dict[str, Ops] = {
    "2D": {
        "ktoi": ktoi,
        "itok": itok,
        "combine_csm": combine_csm,
        "expand_csm": expand_csm,
    },
}

class A:
    def __init__(self, modality: str):
        try:
            ops = OPS_REGISTRY[modality]
        except KeyError:
            raise ValueError(f"Unsupported modality: {modality}")

        self.modality = modality
        self.ktoi = ops["ktoi"]
        self.itok = ops["itok"]
        self.combine_csm = ops["combine_csm"]
        self.expand_csm = ops["expand_csm"]