import torch
from numpy.testing import assert_array_almost_equal

class TestModel(torch.nn.Module):
    __test__ = False
    
    def __init__(self, input_size, hidden_size, output_size, n_hidden=0):
        super().__init__()

        self.dummy_attribute = 0

        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()

        if n_hidden > 0:
            layers = []
            for _ in range(n_hidden):
                layers.append(torch.nn.Linear(hidden_size, hidden_size))
                layers.append(torch.nn.ReLU())
            self.hidden_layers = torch.nn.Sequential(*layers)
        self._n_hidden = n_hidden

        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        if self._n_hidden > 0:
            x = self.hidden_layers(x)
        x = self.linear2(x)

        return x
    
    def dummy_method(self) -> int:
        return 0

def assert_tensor_almost_equal(tensor1:torch.Tensor, tensor2:torch.Tensor, decimal:int=5):
    tensor1 = tensor1.detach().numpy()
    tensor2 = tensor2.detach().numpy()

    assert_array_almost_equal(tensor1, tensor2, decimal)
