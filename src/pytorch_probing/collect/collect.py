from typing import List, Tuple, Union, Optional, Dict
import os
import datetime

import torch
from torch.utils.data import DataLoader
import numpy as np

from pytorch_probing import Interceptor

ModuleData = Union[torch.Tensor, List["ModuleData"], 
                   Tuple["ModuleData"], Dict[str, "ModuleData"]]

def to_cpu(x : ModuleData, detach:bool=False) -> ModuleData:
    if isinstance(x, torch.Tensor):
        if detach:
            x = x.detach()

        result = x.cpu()
    elif isinstance(x, list) or isinstance(x, tuple):
        result = []
        for element in x:
            result.append(to_cpu(element, detach))
    else:
        result = {}
        for key in x:
            result[key] = to_cpu(x[key], detach)

    return result

def collect(module:torch.nn.Module, paths:List[str], dataloader:DataLoader, 
            save_path:Optional[str] = None, dataset_name:Optional[str] = None,
            device:Optional[str]=None, 
            save_target:bool=False, save_prediction:bool=False) -> str:
    #dataloader Must be in CPU.
    
    original_mode = module.training
    module.eval()

    if save_path is None:
        save_path = "."
    
    if dataset_name is None:
        dataset_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

    dataset_path = os.path.join(save_path, dataset_name)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    

    if device is None:
        device = next(module.parameters()).device
    else:
        device = torch.device(device)
        module = module.to(device)

    with Interceptor(module, paths) as interceptor:
        with torch.no_grad():
            for chunk_index, (x, y) in enumerate(dataloader):
                x_device = x.to(device)

                pred : torch.Tensor | Tuple[torch.Tensor] = interceptor(x_device)

                intercepted_outputs = interceptor.outputs
                intercepted_outputs = to_cpu(intercepted_outputs, detach=True)
               
                chunk = {"inputs":x, "intercepted_outputs":intercepted_outputs, "index":chunk_index}

                if save_target:
                    chunk["target"] = y
                if save_prediction:
                    pred_cpu = to_cpu(pred, detach=True)
                    chunk["prediction"] = pred_cpu                        
                
                chunk_path = os.path.join(dataset_path, str(chunk_index))

                np.savez(chunk_path, **chunk)

    module.train(original_mode)

    return dataset_path
