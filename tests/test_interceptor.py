import unittest
import pprint
import os


import torch

from pytorch_probing import Interceptor

from .utils import TestModel, assert_tensor_almost_equal


class TestInterceptor(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        torch.set_grad_enabled(False)

        self.input_size = 2
        self.hidden_size = 3
        self.output_size = 1

        self.test_model = TestModel(self.input_size, self.hidden_size,
                                     self.output_size, n_hidden=2)
        self.test_model = self.test_model.eval()
        
    def tearDown(self) -> None:
        super().tearDown()
    
        self.test_model = None

        if os.path.exists("intercepted_model.pth"):
            os.remove("intercepted_model.pth")

    def test_no_intercept(self) -> None:
        paths = []

        inputs = torch.randn([10, self.input_size])
        outputs = self.test_model(inputs)

        intercepted_model = Interceptor(self.test_model, paths)
        outputs2 = intercepted_model(inputs)

        assert_tensor_almost_equal(outputs, outputs2)

    def test_intercept(self) -> None:
        paths = ["linear1", "hidden_layers.1"]
        
        inputs = torch.randn([10, self.input_size])
        linear1_output = self.test_model.linear1(inputs)
        relu_output = self.test_model.relu(linear1_output)
        hidden_layers0_output = self.test_model.hidden_layers[0](relu_output)
        hidden_layers1_output = self.test_model.hidden_layers[1](hidden_layers0_output)

        intercepted_model = Interceptor(self.test_model, paths)
        intercepted_model(inputs)
        intercepted_outputs = intercepted_model.outputs

        assert_tensor_almost_equal(linear1_output, intercepted_outputs["linear1"])


        assert_tensor_almost_equal(hidden_layers1_output, intercepted_outputs["hidden_layers.1"])
        
        
    def test_clear(self) -> None:
        paths = ["linear1", "hidden_layers.1"]
        
        intercepted_model = Interceptor(self.test_model, paths)

        inputs = torch.randn([10, self.input_size])
        intercepted_model(inputs)

        intercepted_model.interceptor_clear()
        for path in intercepted_model._interceptor_layers:
            layer = intercepted_model._interceptor_layers[path]
            assert layer._intercepted_output is None


    def test_reduce(self) -> None:
        paths = ["linear1", "hidden_layers.1"]
        
        intercepted_model = Interceptor(self.test_model, paths)

        intercepted_model.reduce()
        
        model_string = pprint.pformat(self.test_model)

        assert "Interceptor" not in model_string

    def test_wrong_path(self) -> None:
        paths = ["linear1", "WRONG_PATH", "hidden_layers.1"]

        with self.assertRaises(ValueError):
            Interceptor(self.test_model, paths)

    def test_detach(self) -> None:
        paths = ["linear1"]

        torch.set_grad_enabled(True)

        for detach in [True, False]:
        
            intercepted_model = Interceptor(self.test_model, paths, detach=detach)
            
            inputs = torch.randn([10, self.input_size])
            intercepted_model(inputs)
            
            assert intercepted_model.outputs["linear1"].requires_grad != detach

            intercepted_model.reduce()

    def test_save(self) -> None:
        paths = ["linear1", "hidden_layers.1"]
        
        intercepted_model = Interceptor(self.test_model, paths)

        inputs = torch.randn([10, self.input_size])
        intercepted_model(inputs)

        torch.save(intercepted_model, "intercepted_model.pth")
        
        intercepted_model2 = torch.load("intercepted_model.pth")
        intercepted_model2.eval()

        assert intercepted_model2.outputs["linear1"] is None
        assert intercepted_model2.outputs["hidden_layers.1"] is None