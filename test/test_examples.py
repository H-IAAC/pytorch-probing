import os
import pathlib

from torch import tensor #Because of eval
import numpy as np
from testbook import testbook
from testbook.client import TestbookNotebookClient
from numpy.testing import assert_array_almost_equal

tests_path = pathlib.Path(__file__).parent.resolve()
examples_path = os.path.join(tests_path, "../examples")

def ref_to_dict(ref):
    result = str(ref)
    result = result.replace("tensor(", "")
    result = result.replace(")", "")
    result = eval(result)

    return result

@testbook(os.path.join(examples_path, "Collect.ipynb"), execute=True)
def test_collect(tb: TestbookNotebookClient):
    assert tb.cell_output_text("Check dataset files") == '''['.\\\\CollectExample\\\\0.pt',
 '.\\\\CollectExample\\\\1.pt',
 '.\\\\CollectExample\\\\2.pt',
 '.\\\\CollectExample\\\\3.pt',
 '.\\\\CollectExample\\\\4.pt',
 '.\\\\CollectExample\\\\5.pt',
 '.\\\\CollectExample\\\\6.pt',
 '.\\\\CollectExample\\\\7.pt']'''
    
    assert tb.cell_output_text("Load dataset") == "'CollectExample'"

    assert tb.cell_output_text("Check dataset size") == "32"

    linear1_output = tb.ref("linear1_output.tolist()")
    intercepted_output_linear1 = tb.ref("intercepted_output['linear1'].tolist()")

    assert_array_almost_equal(linear1_output, intercepted_output_linear1)

    target = tb.ref("target.item()")
    y = tb.ref("y.item()")
    assert target == y

    prediction = tb.ref("prediction.item()")
    pred = tb.ref("pred.item()")
    assert prediction == pred

    saved_input = tb.ref("saved_input.tolist()")
    x = tb.ref("x.tolist()")
    assert_array_almost_equal(x, saved_input)

@testbook(os.path.join(examples_path, "Interceptor.ipynb"), execute=True)
def test_interceptor(tb: TestbookNotebookClient):
    assert tb.cell_output_text("Create Interceptor") == '''Interceptor(
  (_module): ExampleModel(
    (linear1): InterceptorLayer(
      (_module): Linear(in_features=2, out_features=3, bias=True)
    )
    (relu): ReLU()
    (hidden_layers): Sequential(
      (0): Linear(in_features=3, out_features=3, bias=True)
      (1): ReLU()
      (2): InterceptorLayer(
        (_module): Linear(in_features=3, out_features=3, bias=True)
      )
      (3): ReLU()
    )
    (linear2): InterceptorLayer(
      (_module): Linear(in_features=3, out_features=1, bias=True)
    )
  )
)'''

    assert tb.cell_output_text("View model") == '''ExampleModel(
  (linear1): InterceptorLayer(
    (_module): Linear(in_features=2, out_features=3, bias=True)
  )
  (relu): ReLU()
  (hidden_layers): Sequential(
    (0): Linear(in_features=3, out_features=3, bias=True)
    (1): ReLU()
    (2): InterceptorLayer(
      (_module): Linear(in_features=3, out_features=3, bias=True)
    )
    (3): ReLU()
  )
  (linear2): InterceptorLayer(
    (_module): Linear(in_features=3, out_features=1, bias=True)
  )
)'''

    intercepted_model_outputs = tb.cell_output_text("View intercepted outputs")
    intercepted_model_outputs = str(intercepted_model_outputs)
    intercepted_model_outputs = intercepted_model_outputs.replace("tensor(", "")
    intercepted_model_outputs = intercepted_model_outputs.replace(")", "")
    intercepted_model_outputs = eval(intercepted_model_outputs)

    assert str(list(intercepted_model_outputs.keys())) == "['linear1', 'hidden_layers.2', 'linear2']"
    for key in intercepted_model_outputs:
        value = np.array(intercepted_model_outputs[key])

        if key == "linear2":
            assert str(value.shape) == "(10, 1)"
        else:
            assert str(value.shape) == "(10, 3)"
         

    assert tb.cell_output_text("interceptor_clear example") == "{'linear1': None, 'hidden_layers.2': None, 'linear2': None}"

    assert tb.cell_output_text("Reduce example") == '''ExampleModel(
  (linear1): Linear(in_features=2, out_features=3, bias=True)
  (relu): ReLU()
  (hidden_layers): Sequential(
    (0): Linear(in_features=3, out_features=3, bias=True)
    (1): ReLU()
    (2): Linear(in_features=3, out_features=3, bias=True)
    (3): ReLU()
  )
  (linear2): Linear(in_features=3, out_features=1, bias=True)
)'''

@testbook(os.path.join(examples_path, "Prober.ipynb"), execute=True)
def test_prober(tb: TestbookNotebookClient):
    assert tb.cell_output_text("Create Prober") == '''Prober(
  (_module): ExampleModel(
    (linear1): InterceptorLayer(
      (_module): Linear(in_features=2, out_features=3, bias=True)
    )
    (relu): InterceptorLayer(
      (_module): ReLU()
    )
    (linear2): Linear(in_features=3, out_features=1, bias=True)
  )
  (_probes): ModuleDict(
    (linear1): Linear(in_features=3, out_features=2, bias=True)
    (relu): Identity()
  )
)'''

    assert tb.cell_output_text("View modified model") == '''ExampleModel(
  (linear1): InterceptorLayer(
    (_module): Linear(in_features=2, out_features=3, bias=True)
  )
  (relu): InterceptorLayer(
    (_module): ReLU()
  )
  (linear2): Linear(in_features=3, out_features=1, bias=True)
)'''

    outputs = tb.cell_output_text("View probes outputs")
    outputs = ref_to_dict(outputs)

    assert str(list(outputs.keys())) == "['linear1', 'relu']"
    for key in outputs:
        value = np.array(outputs[key])

        if key == "linear1":
            assert str(value.shape) == "(10, 2)"
        elif key == "relu":
            assert str(value.shape) == "(10, 3)"

    outputs2 = tb.cell_output_text("View probes outputs 2")
    outputs2 = eval(outputs2)

    assert str(list(outputs2.keys())) == "['linear1']"
    assert str(list(outputs2["linear1"].keys())) == "['probe1', 'probe2']"

    for key in outputs2["linear1"]:
        value = outputs2["linear1"][key].numpy()

        if key == "probe1":
            assert str(value.shape) == "(10, 2)"
        elif key == "probe2":
            assert str(value.shape) == "(10, 1)"


if __name__ == "__main__":
    test_collect()
    test_interceptor()
    test_prober()