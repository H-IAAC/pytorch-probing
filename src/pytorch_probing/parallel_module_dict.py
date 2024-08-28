from typing import Any

import torch

class ParallelModuleDict(torch.nn.ModuleDict):
    '''
    Same as torch.nn.ModuleDict, but executes all the modules in 
    parallel and creates a dictionary with results.

    Examples
    ---
    >>> from pytorch_probing import ParallelModuleDict
    >>> import torch
    >>> from torch.nn import ReLU, Sigmoid
    >>> parallel_dict = ParallelModuleDict({"relu":ReLU(), "sigmoid":Sigmoid()})
    >>> inputs = torch.arange(-1, 2, 1)
    >>> outputs = parallel_dict(inputs)
    >>> print(outputs)
    {'relu': tensor([0, 0, 1]), 'sigmoid': tensor([0.2689, 0.5000, 0.7311])}
    '''

    def forward(self, *args, **kwargs) -> Any:
        '''
        Execute all the modules, returning its results.

        Returns:
            result: dictonary with modules outputs, indexed by the same keys as the modules.
        '''
        
        result = {}
        for key in self:
            result[key] = self[key](*args, **kwargs)

        return result