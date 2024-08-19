import unittest

import torch

from pytorch_probing import ParallelModuleDict

from .utils import assert_tensor_almost_equal

class TestParallelModuleDict(unittest.TestCase):

    def test(self) -> None:
        modules = {
            "linear":torch.nn.Linear(10, 10),
            "relu":torch.nn.ReLU(),
            "linear2":torch.nn.Linear(10, 10)
        }

        module = ParallelModuleDict(modules)

        x = torch.rand(10)
        result = module(x)

        assert len(result) == len(modules)

        for key in modules:
            y = modules[key](x)
            assert_tensor_almost_equal(result[key], y)


    def test_empty(self) -> None:
        module = ParallelModuleDict()

        x = torch.rand(5)
        y = module(x)

        assert len(y) == 0