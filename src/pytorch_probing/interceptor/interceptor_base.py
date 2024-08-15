import abc
import warnings
from typing import List

import torch

class InterceptorBase(torch.nn.Module, abc.ABC):
    def __init__(self, module: torch.nn.Module, member_names:List[str]) -> None:
        super().__init__()

        self._module = module
        self._member_names = member_names

        self._reduced = False

    def reduce(self) -> torch.nn.Module:
        self._reduced = True

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name)

    def __setattr__(self, name: str, value):
        if name in ["_member_names", "_module"] or name in self._member_names:
            super().__setattr__(name, value)
        else:
            return setattr(self._module, name, value)
        
    def check_reduced(self):
        if self._reduced:
            warnings.warn("Model was reduced. Not intercepting results")