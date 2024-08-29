import abc
import warnings
from typing import List

import torch

class ModuleWrapper(torch.nn.Module, abc.ABC):
    '''
    Wraps a PyTorch Module, enabling to add additional features and passing through original module members.
    '''

    def __init__(self, module: torch.nn.Module, member_names:List[str]) -> None:
        '''
        ModuleWrapper init

        Subclass must pass all the new members names, because of the passthrought feature.

        Args:
            module (torch.nn.Module): module being wrapped
            member_names (List[str]): all the names of members of the Wrapper subclass
        '''
        super().__init__()

        self._module : torch.nn.Module = module
        self._member_names = member_names

        self._reduced = False

    def forward(self, *args, **kwargs):
        self._check_reduced()
        return self._module(*args, **kwargs)

    @abc.abstractmethod
    def reduce(self) -> torch.nn.Module:
        '''
        Reduces the wrapped module to the original module.

        May apply persistent alterations.

        Returns:
            torch.nn.Module: reduced module.
        '''
        self._reduced = True

        return self._module

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
        
    def _check_reduced(self):
        '''
        Checks if reduced, raising a warning if true.
        '''
        if self._reduced:
            warnings.warn("Model was reduced. Not intercepting results")