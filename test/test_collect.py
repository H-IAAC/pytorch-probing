import unittest
import os
import shutil
import json
import math

import torch
from torch.utils.data import DataLoader
import numpy as np
from numpy.testing import assert_array_almost_equal

from pytorch_probing import collect, Interceptor, CollectedDataset

from .utils import TestModel, assert_tensor_almost_equal, TestDataset


class TestCollect(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.input_size = 2
        self.hidden_size = 3
        self.output_size = 1

        self.test_model = TestModel(self.input_size, self.hidden_size,
                                     self.output_size, n_hidden=0)
        self.test_model = self.test_model.eval()

        self.n_sample = 31
        self.batch_size = 4
        self.n_batch = math.ceil(self.n_sample / self.batch_size)

        self.test_dataset = TestDataset(self.input_size, self.output_size, self.n_sample)
        self.test_dataloader = DataLoader(self.test_dataset, self.batch_size, shuffle=False)
        
        self.save_path = "dataset"
        self.dataset_name = "test_dataset"

    def tearDown(self) -> None:
        super().tearDown()
    
        self.test_model = None
        self.test_dataset = None
        self.test_dataloader = None


    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("dataset")

    def test_collect(self) -> None:
        paths = ["linear1"]
        dataset_path = collect(self.test_model, paths, self.test_dataloader, 
                               self.save_path, self.dataset_name)

        assert dataset_path == os.path.join(self.save_path, self.dataset_name)

        info_path = os.path.join(dataset_path, "info.json")
        with open(info_path) as file:
            info = json.load(file)

        assert info["n_sample"] == self.n_sample
        assert info["n_chunk"] == self.n_batch
        assert info["dataset_name"] == self.dataset_name
        assert info["has_input"] == False
        assert info["has_target"] == False
        assert info["has_prediction"] == False
        assert info["module_name"] == "TestModel"

        chunk_path = os.path.join(dataset_path, "0.pt")
        chunk0 = torch.load(chunk_path)

        keys = set(chunk0.keys())
        expected_keys = set(["intercepted_outputs", "index"])
        assert keys == expected_keys
        
        assert chunk0["index"] == 0

        x, y = next(iter(self.test_dataloader))
        with Interceptor(self.test_model, paths) as interceptor:
            interceptor(x)
            intercepted_outputs = interceptor.outputs

        chunk_intercepted_outputs = chunk0["intercepted_outputs"]

        assert chunk_intercepted_outputs.keys() == intercepted_outputs.keys()
        assert_array_almost_equal(chunk_intercepted_outputs["linear1"], intercepted_outputs["linear1"], 5)

        dataset = CollectedDataset(dataset_path)
        assert_array_almost_equal(dataset[0]["linear1"], intercepted_outputs["linear1"][0], 5)
        assert type(dataset[0]["linear1"]) == torch.Tensor

    def test_collect_all(self) -> None:
        paths = ["linear1"]
        dataset_path = collect(self.test_model, paths, self.test_dataloader, 
                               self.save_path, self.dataset_name, 
                               save_input=True, save_target=True, save_prediction=True)

        assert dataset_path == os.path.join(self.save_path, self.dataset_name)

        info_path = os.path.join(dataset_path, "info.json")
        with open(info_path) as file:
            info = json.load(file)

        assert info["n_sample"] == self.n_sample
        assert info["n_chunk"] == self.n_batch
        assert info["dataset_name"] == self.dataset_name
        assert info["has_input"] == True
        assert info["has_target"] == True
        assert info["has_prediction"] == True
        assert info["module_name"] == "TestModel"

        chunk_path = os.path.join(dataset_path, "0.pt")
        chunk0 = torch.load(chunk_path)

        keys = set(chunk0.keys())
        expected_keys = set(["intercepted_outputs", "index", "target", "input", "prediction"])
        assert keys == expected_keys
        
        assert chunk0["index"] == 0

        x, y = next(iter(self.test_dataloader))
        with Interceptor(self.test_model, paths) as interceptor:
            interceptor(x)
            intercepted_outputs = interceptor.outputs

        chunk_intercepted_outputs = chunk0["intercepted_outputs"]

        assert chunk_intercepted_outputs.keys() == intercepted_outputs.keys()
        assert_array_almost_equal(chunk_intercepted_outputs["linear1"], intercepted_outputs["linear1"], 5)

        dataset = CollectedDataset(dataset_path, True, True, True)
        assert_array_almost_equal(dataset[0][0]["linear1"], intercepted_outputs["linear1"][0], 5)
        
        assert isinstance(dataset[0][0]["linear1"], torch.Tensor)
        for i in range(1, 4):
            assert isinstance(dataset[0][i], torch.Tensor)

        dataset = CollectedDataset(dataset_path)
        assert isinstance(dataset[0], dict)