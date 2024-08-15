from __future__ import annotations

from typing import List, Tuple, Dict, Any

import torch

from pytorch_probing.module_wrapper import ModuleWrapper

class InterceptorLayer(ModuleWrapper):
    def __init__(self, module, detach=True) -> None:
        super().__init__(module, ["_intercepted_output", "_detach"])

        self._intercepted_output : None | torch.Tensor | List[torch.Tensor] = None
        self._detach = detach

    @property
    def output(self) -> None | torch.Tensor | List[torch.Tensor]:
        return self._intercepted_output

    def forward(self, *args, **kwargs):
        self.check_reduced()

        outputs : Tuple[torch.Tensor] | torch.Tensor = self._module(*args, **kwargs)
        
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
        self._intercepted_output = None

    def reduce(self):
        super().reduce()
        return self._module
    
    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state["_intercepted_output"] = None

        return state