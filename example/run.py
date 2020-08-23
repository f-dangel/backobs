"""Run BackpackRunner with SGD on a DeepOBS test problem."""

from runner import BackpackRunner
from torch.optim import SGD

from backobs.utils import SUPPORTED
from backpack import extensions
from deepobs.config import set_data_dir
from deepobs.pytorch.config import set_default_device

FORCE_CPU = False
if FORCE_CPU:
    set_default_device("cpu")

set_data_dir("~/tmp/data_deepobs")


def extensions_fn():
    return [
        extensions.BatchGrad(),
    ]


optimizer_class_sgd = SGD
hyperparams_sgd = {
    "lr": {"type": float, "default": 0.1,},
    "momentum": {"type": float, "default": 0.0,},
}

runner = BackpackRunner(optimizer_class_sgd, hyperparams_sgd, extensions_fn)
runner.run(num_epochs=1, batch_size=3)
