from __future__ import annotations

from typing import Dict, Any, cast, MutableMapping 

import torch

from pytorch_probing.interceptor import Interceptor

class Prober(Interceptor):
    '''
    ModuleWrapper that injects a PyTorch Module (probe) in a 
    intermediary output and returns it outputs.

    Stores the probes outputs in the 'outputs' attribute.

    Examples
    --------
    
    >>> import torch
    >>> from torch.nn import Sequential, Sigmoid, Linear, ReLU
    >>> from pytorch_probing import Prober
    >>> module = Sequential(Sigmoid(), Linear(3, 1))
    >>> probes = {"0":ReLU()}
    >>> prober = Prober(module, probes)
    >>> inputs = torch.arange(-1, 2, 1)
    >>> _ = prober(inputs)
    >>> print(prober.outputs)
    {'0': tensor([0.2689, 0.5000, 0.7311])}

    '''
    def __init__(self, module: torch.nn.Module, 
                 probes:MutableMapping [str, torch.nn.Module|None],
                 return_in_forward:bool=True) -> None:
        '''
        Prober init.

        Args:
            module (torch.nn.Module): Module to wrap.
            probes (Dict[str, torch.nn.Module | None]): Probes to inject. Must be indexed by the path to inject and 
                can be submodules as "my_module.submodule.subsubmodule". If probe is 'None', creates a 'Identity' module.
            return_in_forward (bool, optional): If should return the probes outputs in the forward. Defaults to True.
        '''
        super().__init__(module, list(probes.keys()))

        self._member_names += ["_probes", "_return_in_forward", "_probe_outputs"]
    
        for path in probes:
            if probes[path] is None:
                probes[path] = torch.nn.Identity()

        probes_ = cast(MutableMapping [str, torch.nn.Module], probes)

        self._probes = torch.nn.ModuleDict(probes_)

        self._probe_outputs : Dict[str, Any] | None = None

        self._return_in_forward = return_in_forward


    @property
    def outputs(self) -> Dict[str, Any] | None:
        '''
        Gets the probes outputs.

        Returns:
            Dict[str, Any] | None: Probes outputs, indexed by the probed path. Is None if the output was 
            cleared or no forwards were executed.
        '''
        return self._probe_outputs

    def forward(self, *args, **kwargs):
        '''
        Executes the wrapped module and the probes.
        '''
        main_predictions = super().forward(*args, **kwargs)

        probe_predictions = {}

        for path in self._probes:
            probe_input = self._interceptor_layers[path].output
            probe_output = self._probes[path](probe_input)

            probe_predictions[path] = probe_output
        
        self.interceptor_clear()
        self._probe_outputs = probe_predictions

        if self._return_in_forward:
            return main_predictions, probe_predictions
        else:
            return main_predictions
        
    def probes_clear(self):
        '''
        Clears the stored outputs.
        '''
        self._probe_outputs = None

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state["_probe_outputs"] = None

        return state
        
            