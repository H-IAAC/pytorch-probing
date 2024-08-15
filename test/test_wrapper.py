import unittest
import pprint
import os

from pytorch_probing import Interceptor

from .utils import TestModel, assert_tensor_almost_equal


class TestWrapper(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.input_size = 2
        self.hidden_size = 3
        self.output_size = 1

        self.test_model = TestModel(self.input_size, self.hidden_size,
                                     self.output_size, n_hidden=2)
        self.test_model = self.test_model.eval()
        
    def tearDown(self) -> None:
        super().tearDown()
    
        self.test_model = None

    def test_bypass(self) -> None:
        paths = []
        intercepted_model = Interceptor(self.test_model, paths)

        assert intercepted_model.dummy_method() == 0
        assert intercepted_model.dummy_attribute == 0

    def test_create_attribute(self) -> None:
        paths = []
        intercepted_model = Interceptor(self.test_model, paths)

        intercepted_model.a = 1
        assert intercepted_model.a == 1

        self.test_model.b = 2

        intercepted_model.reduce()

        assert self.test_model.a == 1
        assert self.test_model.b == 2