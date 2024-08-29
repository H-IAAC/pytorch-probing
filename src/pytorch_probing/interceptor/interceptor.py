from __future__ import annotations

from typing import Dict, List

import torch

from pytorch_probing.module_wrapper import ModuleWrapper
from .interceptor_layer import InterceptorLayer

class Interceptor(ModuleWrapper):
    '''
    ModuleWrapper that intercepts intermediary outputs.

    Stores the intercepted outputs, avaiable in the "outputs" property.

    Examples
    --------
    >>> import torch
    >>> from torch.nn import Sequential, Sigmoid, ReLU, Identity
    >>> from pytorch_probing import Interceptor
    >>> module = Sequential(Sequential(Sigmoid(), ReLU()), Identity())
    >>> paths = ["0.0", "0"] #Sequential.Sigmoid, Sequential = Sequential.ReLU
    >>> interceptor = Interceptor(module, paths)
    >>> inputs = torch.arange(-1, 2, 1)
    >>> _ = interceptor(inputs)
    >>> print(interceptor.outputs)
    {'0.0': tensor([0.2689, 0.5000, 0.7311]), '0': tensor([0.2689, 0.5000, 0.7311])}

    '''
    def __init__(self, module:torch.nn.Module, intercept_paths:List[str], detach:bool=True) -> None:
        '''
        _summary_

        Args:
            module (torch.nn.Module): Module to wrap.
            intercept_paths (List[str]): Paths of the modules to intercept the outputs. Can be submodules as "my_module.submodule.subsubmodule".
            detach (bool, optional):  If should detach the intercepted outputs. Defaults to True.

        Raises:
            ValueError: If there is no module with specified path.
        '''
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
        '''
        Executes the module and intercepts its intermediary outputs.
        '''
        self._check_reduced()
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
    def outputs(self) -> Dict[str, torch.Tensor | List[torch.Tensor]|None] | None:
        '''
        Gets the intercepted outputs.

        Returns:
            Dict[str, torch.Tensor | List[torch.Tensor]]: Intercepted outputs, indexed by the module path. Is None if the output was 
            cleared or no forwards were executed.
        '''
        if self._reduced:
            return None

        outputs = {}
        for path in self._intercept_paths:
            outputs[path] = self._interceptor_layers[path].output

        return outputs
    
    def interceptor_clear(self):
        '''
        Clears the stored outputs.
        '''
        for path in self._intercept_paths:
            self._interceptor_layers[path].interceptor_clear()

    def get_submodule(self, path:str):
        module = self._module
        if path == "":
            return module

        path_parts = path.split(".")

        for part in path_parts:
            submodule = module._modules[part]
            assert isinstance(submodule, torch.nn.Module)
            module = submodule

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

