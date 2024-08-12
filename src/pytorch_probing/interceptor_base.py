import abc
from typing import List

import torch

class InterceptorBase(torch.nn.Module, abc.ABC):
    def __init__(self, module: torch.nn.Module, member_names:List[str]) -> None:
        super().__init__()

        self._module = module
        self._member_names = member_names

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