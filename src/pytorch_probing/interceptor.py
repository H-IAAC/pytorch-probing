from typing import List, Tuple, Dict

import torch

from .interceptor_base import InterceptorBase
from .interceptor_layer import InterceptorLayer

class Interceptor(InterceptorBase):
    def __init__(self, module, intercept_paths) -> None:
        super().__init__(module, ["_intercept_paths", "_interceptor_layers"])

        self._intercept_paths = intercept_paths

        self._interceptor_layers : Dict[str, InterceptorLayer] = {}
        for path in intercept_paths:
            submodule = self.get_submodule(path)
            parent = self.get_submodule_parent(path)
            name = path.split("/")[-1]

            interceptor_layer = InterceptorLayer(submodule)
            self._interceptor_layers[path] = interceptor_layer

            parent._modules[name] = interceptor_layer

    def forward(self, *args, **kwargs):
        return self._module(*args, **kwargs)
        
    def reduce(self) -> torch.nn.Module:
        for path in self._intercept_paths:
            parent = self.get_submodule_parent(path)
            name = path.split("/")[-1]
            interceptor_layer = self._interceptor_layers[path]

            parent._modules[name] = interceptor_layer.reduce()

        return self._module
    
    @property
    def outputs(self):
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

        path_parts = path.split("/")

        for part in path_parts:
            module = module._modules[part]

        return module

    def get_submodule_parent(self, path:str):
        path_parts = path.split("/")
        path_parts = path_parts[:-1]
        path = "/".join(path_parts)

        return self.get_submodule(path)