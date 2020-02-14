"""Run BackpackRunner on a DeepOBS test problem."""

from runner import BackpackRunner
from torch.optim import SGD


def make_backpack_runner_for_sgd():
    """Create a BackpackRunner for the SGD optimizer."""
    optimizer_class_sgd = SGD
    hyperparams_sgd = {
        "lr": {"type": float},
        "momentum": {"type": float, "default": 0.0,},
    }

    return BackpackRunner(optimizer_class_sgd, hyperparams_sgd)


if __name__ == "__main__":
    print("Running BackPACK runner with SGD:")
    runner = make_backpack_runner_for_sgd()
    runner.run()
