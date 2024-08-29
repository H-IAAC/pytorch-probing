from __future__ import annotations

from typing import List, Tuple, Dict, Any

import torch

from pytorch_probing.module_wrapper import ModuleWrapper

class InterceptorLayer(ModuleWrapper):
    '''
    ModuleWrapper that intercepts a Module output.

    Stores the intercepted output, avaiable in the "output" property.
    '''

    def __init__(self, module:torch.nn.Module, detach=True) -> None:
        '''
        InterceptorLayer init.

        Args:
            module (torch.nn.Module): Module to intercept the output.
            detach (bool, optional): If should detach the intercepted output. Defaults to True.
        '''
        super().__init__(module, ["_intercepted_output", "_detach"])

        self._intercepted_output : None | torch.Tensor | List[torch.Tensor] = None
        self._detach = detach

    @property
    def output(self) -> None | torch.Tensor | List[torch.Tensor]:
        '''
        Gets the output.

        Returns:
            None | torch.Tensor | List[torch.Tensor]: Intercepted output. Is None if the output was 
            cleared or no forwards were executed.
        '''
        return self._intercepted_output

    def forward(self, *args, **kwargs):
        '''
        Executes the module and intercepts its output.
        '''
        self._check_reduced()

        outputs = self._module(*args, **kwargs)
        
        if isinstance(outputs, tuple):
            self._intercepted_output = []

            for output in outputs:
                if self._detach:
                    output = output.detach()

                self._intercepted_output.append(output.clone())
        else:  
            if self._detach:
                outputs = outputs.detach()
            self._intercepted_output = outputs.clone()

        return outputs
    
    def interceptor_clear(self):
        '''
        Clears the intercepted output.
        '''
        self._intercepted_output = None

    def reduce(self):
        super().reduce()
        return self._module
    
    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state["_intercepted_output"] = None

        return state