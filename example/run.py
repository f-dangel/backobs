"""Run BackpackRunner on a DeepOBS test problem."""

from runner import BackpackRunner
from torch.optim import SGD


def make_backpack_runner_for_sgd(check=True):
    """Create a BackpackRunner for the SGD optimizer."""
    optimizer_class_sgd = SGD
    hyperparams_sgd = {
        "lr": {"type": float, "default": 0.1,},
        "momentum": {"type": float, "default": 0.0,},
    }

    return BackpackRunner(optimizer_class_sgd, hyperparams_sgd, check_compatible=check)


if __name__ == "__main__":
    print("Running BackPACK runner with SGD:")
    runner = make_backpack_runner_for_sgd()
    runner.run()
