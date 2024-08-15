from typing import Tuple, List, Dict
from typing import List, Callable, Any

import torch

from pytorch_probing.interceptor import Interceptor

class Prober(Interceptor):
    def __init__(self, module: torch.nn.Module, 
                 probes:Dict[str, torch.nn.Module|None],
                 return_in_forward:bool=True) -> None:
        super().__init__(module, list(probes.keys()))

        self._member_names += ["_probes", "_return_in_forward", "_probe_outputs"]

        for path in probes:
            if probes[path] is None:
                probes[path] = torch.nn.Identity()

        self._probes = torch.nn.ModuleDict(probes)

        self._return_in_forward = return_in_forward

        self._probe_outputs = None

    def outputs(self) -> Dict[str, Any]:
        return self._probe_outputs

    def forward(self, *args, **kwargs):
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
        self._probe_outputs = None

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state["_probe_outputs"] = None

        return state
        
            