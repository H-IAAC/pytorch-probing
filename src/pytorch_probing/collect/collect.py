from __future__ import annotations

import os
import json
import datetime
from typing import List, Tuple, Union, Optional, Dict

import torch
from torch.utils.data import DataLoader
import numpy as np

from pytorch_probing import Interceptor

ModuleData = Union[torch.Tensor, List["ModuleData"], 
                   Tuple["ModuleData"], Dict[str, "ModuleData"]]

def _to_cpu(x : ModuleData, detach:bool=False) -> ModuleData:
    '''
    Sends the data to the CPU.

    Args:
        x (ModuleData): Data to send to CPU.
        detach (bool, optional): If should detach the data. Defaults to False.

    Returns:
        ModuleData: CPU data.
    '''
    if isinstance(x, torch.Tensor):
        if detach:
            x = x.detach()

        result = x.cpu()
    elif isinstance(x, list) or isinstance(x, tuple):
        result = []
        for element in x:
            result.append(_to_cpu(element, detach))
    else:
        result = {}
        for key in x:
            result[key] = _to_cpu(x[key], detach)

    return result

def collect(module:torch.nn.Module, paths:List[str], dataloader:DataLoader, 
            save_path:Optional[str] = None, dataset_name:Optional[str] = None,
            device:Optional[str]=None, 
            save_input:bool=False, save_target:bool=False, save_prediction:bool=False) -> str:
    '''
    Executes a PyTorch module over a dataset, saving intermediary outputs.

    Args:
        module (torch.nn.Module): ,odule to execute.
        paths (List[str]): Paths of the modules to collect outputs. Can be submodules as "my_module.submodule.subsubmodule".
        dataloader (DataLoader): Dataloader with the data. Must return data in the CPU with format (input, output).
        save_path (Optional[str], optional): Directory to save the dataset. If 'None', uses the current path. Defaults to None.
        dataset_name (Optional[str], optional): Name of created dataset. If 'None', uses the current date-time. Defaults to None.
        device (Optional[str], optional): Device to execute the module. If 'None', uses the device of the first module parameter. Defaults to None.
        save_input (bool, optional): If should save the dataset input. Defaults to False.
        save_target (bool, optional): If should save the dataset targets. Defaults to False.
        save_prediction (bool, optional): If should save the dataset prediction. Defaults to False.

    Returns:
        str: the created dataset path.
    '''
    
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
                intercepted_outputs = _to_cpu(intercepted_outputs, detach=True)
               
                chunk = {"intercepted_outputs":intercepted_outputs, "index":chunk_index}

                if save_input:
                    chunk["input"] = x
                if save_target:
                    chunk["target"] = y
                if save_prediction:
                    pred_cpu = _to_cpu(pred, detach=True)
                    chunk["prediction"] = pred_cpu                        
                
                chunk_path = os.path.join(dataset_path, str(chunk_index)+".pt")

                torch.save(chunk, chunk_path)

    module.train(original_mode)

    info = {"dataset_name":dataset_name, 
            "n_chunk": len(dataloader),
            "n_sample": len(dataloader.dataset),
            "has_input":save_input,
            "has_target":save_target,
            "has_prediction":save_prediction,
            "module_name":module.__class__.__name__}
    info_path = os.path.join(dataset_path, "info.json") 
    with open(info_path, "w") as file:
        json.dump(info, file)

    return dataset_path
