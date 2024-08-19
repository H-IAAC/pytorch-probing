import torch

class ParallelModuleDict(torch.nn.ModuleDict):
    def forward(self, *args, **kwargs):
        result = {}
        for key in self:
            result[key] = self[key](*args, **kwargs)

        return result
