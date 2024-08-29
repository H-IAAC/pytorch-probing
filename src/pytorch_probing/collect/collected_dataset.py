from __future__ import annotations

import os
import json
import math
import typing
from typing import Tuple

import torch
from torch.utils.data import Dataset
import numpy as np

from pytorch_probing.collect.collect import ModuleData

@typing.no_type_check
def _get_element(x:ModuleData, index:int) -> ModuleData:
    '''
    Get element from a complex data.

    Args:
        x (ModuleData): Data to get element from.
        index (int): Index of the element.

    Returns:
        ModuleData: Data element.
    '''
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        result = x[index]

    elif isinstance(x, list) or isinstance(x, tuple):
        result = []
        for element in x:
            result.append(_get_element(element, index))
    else:
        result = {}
        for key in x:
            result[key] = _get_element(x[key], index)

    return result

class CollectedDataset(Dataset):
    '''
    Dataset to access collected data from pytorch_probing.collect
    '''
    def __init__(self, dataset_path:str, 
                 get_target=False, get_prediction=False,
                 get_input=False) -> None:
        '''
        CollectedDataset init.

        Args:
            dataset_path (str): Path of the collected dataset.
            get_target (bool, optional): If should return the saved target, if avaiable. Defaults to False.
            get_prediction (bool, optional): If should return the saved prediction, if avaiable. Defaults to False.
            get_input (bool, optional): If should return the saved input, if avaiable. Defaults to False.

        Raises:
            ValueError: If get_* is true, but * is not avaiable in the collected dataset.
        '''
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
        '''
        Name of the saved dataset.
        '''
        return self._name

    def __len__(self) -> int:
        '''
        Gets the dataset lenght.

        Returns:
            int: Dataset Lenght
        '''
        return self._size
    
    def __getitem__(self, index:int) -> Tuple[torch.Tensor] | torch.Tensor:
        '''
        Gets a item from the dataset.

        If any get_* is true, return a tuple with the elements in order:
            (intercepted_outputs, target, prediction, input),
        if all is false, return only the intercepted_outputs value.

        Args:
            index (int): Index of the item.

        Returns:
            Tuple[torch.Tensor] | torch.Tensor: Item.
        '''
        chunk_index = index // self._sample_per_chunk
        sample_index_in_chunk = index % self._sample_per_chunk

        chunk_path = os.path.join(self._dataset_path, str(chunk_index)+".pt")
        chunk = torch.load(chunk_path)

        
        return_value = []

        intercepted_outputs = chunk["intercepted_outputs"]
        return_value.append(_get_element(intercepted_outputs, sample_index_in_chunk))
        
        names = ["target", "prediction", "input"]
        for name in names:
            if self._need_to_get[name]:
                data = chunk[name]
                    
                sample = _get_element(data, sample_index_in_chunk)

                return_value.append(sample)

        if len(return_value) == 1:
            return return_value[0]
        
        return tuple(return_value)
