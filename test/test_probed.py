import unittest
import os
import shutil
import json
import math

import torch
from torch.utils.data import DataLoader
import numpy as np
from numpy.testing import assert_array_almost_equal

from pytorch_probing import collect, Prober

from .utils import TestModel, assert_tensor_almost_equal


class TestProbed(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.input_size = 2
        self.hidden_size = 3
        self.output_size = 1

        self.test_model = TestModel(self.input_size, self.hidden_size,
                                     self.output_size, n_hidden=0)
        self.test_model = self.test_model.eval()


    def tearDown(self) -> None:
        super().tearDown()
    
        self.test_model = None


    def test_prober(self) -> None:
        probe_size = 2
        
        probe = torch.nn.Linear(self.hidden_size, probe_size)

        probes = {"linear1":probe, "relu":None}

        probed_model = Prober(self.test_model, probes)

        inputs = torch.randn([10, 2])
        outputs = probed_model(inputs)

        for name in probed_model._interceptor_layers:
            assert probed_model._interceptor_layers[name]._intercepted_output is None

        keys = set(outputs[1].keys())
        expected_keys = set(["linear1", "relu"])
        assert keys == expected_keys

        linear_output = probed_model.linear1(inputs)
        probe_output = probe(linear_output).detach()
        probed_output = outputs[1]["linear1"]
        assert_array_almost_equal(probe_output, probed_output)

        relu_output = probed_model.relu(linear_output).detach()
        probed_output = outputs[1]["relu"].detach()
        assert_array_almost_equal(relu_output, probed_output)
        
       

        probed_model.probes_clear()
        assert probed_model.outputs is None


    def test_empty(self) -> None:
        probes = {}

        probed_model = Prober(self.test_model, probes)

        inputs = torch.randn([10, 2])
        outputs = probed_model(inputs)

        assert len(outputs[1]) == 0