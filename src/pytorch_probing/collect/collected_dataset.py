from __future__ import annotations

import os
import json
import math
from typing import Tuple

import torch
from torch.utils.data import Dataset
import numpy as np

from pytorch_probing.collect.collect import ModuleData

def get_element(x:ModuleData, index:int) -> ModuleData:
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        result = x[index]

    elif isinstance(x, list) or isinstance(x, tuple):
        result = []
        for element in x:
            result.append(get_element(element, index))
    else:
        result = {}
        for key in x:
            result[key] = get_element(x[key], index)

    return result

class CollectedDataset(Dataset):
    def __init__(self, dataset_path, 
                 get_target=False, get_prediction=False,
                 get_input=False) -> None:
        super().__init__()
        
        self._dataset_path = dataset_path

        info_path = os.path.join(dataset_path, "info.json")
        with open(info_path) as file:
            self._info = json.load(file)

        self._size = self._info["n_sample"]
        self._n_chunk = self._info["n_chunk"]
        self._name = self._info["dataset_name"]
        self._sample_per_chunk = math.ceil(self._size / self._n_chunk)

        if get_target and not self._info["has_target"]:
            raise ValueError("'get_target' is true, but given dataset doesn't have saved target.")
        if get_prediction and not self._info["has_prediction"]:
            raise ValueError("'get_prediction' is true, but given dataset doesn't have saved prediction.")
        if get_input and not self._info["has_input"]:
            raise ValueError("'get_input' is true, but given dataset doesn't have saved inputs.")

        self._get_target = get_target
        self._get_prediction = get_prediction
        self._get_input = get_input

        self._need_to_get = {
            "target":get_target,
            "prediction":get_prediction,
            "input":get_input
        }

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return self._size
    
    def __getitem__(self, index:int) -> Tuple[torch.Tensor] | torch.Tensor:
        chunk_index = index // self._sample_per_chunk
        sample_index_in_chunk = index % self._sample_per_chunk

        chunk_path = os.path.join(self._dataset_path, str(chunk_index)+".npz")
        chunk = np.load(chunk_path, allow_pickle=True)

        
        return_value = []

        intercepted_outputs = chunk["intercepted_outputs"].item()
        return_value.append(get_element(intercepted_outputs, sample_index_in_chunk))
        
        names = ["target", "prediction", "input"]
        for name in names:
            if self._need_to_get[name]:
                data = chunk[name]

                if isinstance(data, np.ndarray) and data.shape == ():
                    data = data.item()
                    
                sample = get_element(data, sample_index_in_chunk)

                return_value.append(sample)

        if len(return_value) == 1:
            return return_value[0]
        
        return tuple(return_value)
