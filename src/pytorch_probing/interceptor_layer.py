from typing import List, Tuple

import torch

from .interceptor_base import InterceptorBase


class InterceptorLayer(InterceptorBase):
    def __init__(self, module) -> None:
        super().__init__(module, ["_intercepted_output"])

        self._intercepted_output : None | torch.Tensor | List[torch.Tensor] = None

    @property
    def output(self) -> None | torch.Tensor | List[torch.Tensor]:
        return self._intercepted_output

    def forward(self, *args, **kwargs):
        outputs : Tuple[torch.Tensor] | torch.Tensor = self._module(*args, **kwargs)
        
        if isinstance(outputs, tuple):
            self._intercepted_output = []

            for output in outputs:
                self._intercepted_output.append(output.copy())
        else:  
            self._intercepted_output = outputs.detach().clone()

        return outputs
    
    def interceptor_clear(self):
        self._intercepted_output = None

    def reduce(self):
        return self._module