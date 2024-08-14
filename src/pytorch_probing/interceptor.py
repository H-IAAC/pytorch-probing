from typing import Dict, List

import torch

from .interceptor_base import InterceptorBase
from .interceptor_layer import InterceptorLayer

class Interceptor(InterceptorBase):
    def __init__(self, module, intercept_paths, detach=True) -> None:
        super().__init__(module, ["_intercept_paths", "_interceptor_layers"])

        self._intercept_paths = intercept_paths

        self._interceptor_layers : Dict[str, InterceptorLayer] = {}
        for path in intercept_paths:
            try:
                submodule = self.get_submodule(path)
                parent = self.get_submodule_parent(path)
            except KeyError:
                self.reduce()
                raise ValueError(f"There is no module with path '{path}'.") from None
            
            name = path.split(".")[-1]

            interceptor_layer = InterceptorLayer(submodule, detach)
            self._interceptor_layers[path] = interceptor_layer

            parent._modules[name] = interceptor_layer

    def forward(self, *args, **kwargs):
        self.check_reduced()
        return self._module(*args, **kwargs)
        
    def reduce(self) -> torch.nn.Module:
        super().reduce()

        for path in self._intercept_paths:
            if path not in self._interceptor_layers:
                continue

            parent = self.get_submodule_parent(path)
            name = path.split(".")[-1]
            interceptor_layer = self._interceptor_layers[path]

            parent._modules[name] = interceptor_layer.reduce()

        return self._module
    
    @property
    def outputs(self) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        outputs = {}
        for path in self._intercept_paths:
            outputs[path] = self._interceptor_layers[path].output

        return outputs
    
    def interceptor_clear(self):
        for path in self._intercept_paths:
            self._interceptor_layers[path].interceptor_clear()

    def get_submodule(self, path:str):
        module = self._module
        if path == "":
            return module

        path_parts = path.split(".")

        for part in path_parts:
            module = module._modules[part]

        return module

    def get_submodule_parent(self, path:str):
        path_parts = path.split(".")
        path_parts = path_parts[:-1]
        path = ".".join(path_parts)

        return self.get_submodule(path)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.reduce()
