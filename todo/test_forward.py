"""Check that manual forward pass is same as in DeepOBS.

Background: Need access to input data for testing BackPACK extensions.
"""

import argparse
import contextlib
import random

import numpy
import torch

from backobs.integration import ALL
from backobs.integration import extend as backobs_extend
from deepobs.config import set_data_dir


def set_deepobs_seed(seed=0):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def set_up_problem(tproblem_cls, batch_size, seed=0, extend=False):
    """Create problem."""
    set_deepobs_seed(seed)

    tproblem = tproblem_cls(batch_size)

    with contextlib.redirect_stdout(None):
        tproblem.set_up()
    if extend:
        tproblem = backobs_extend(tproblem, check=False)
    tproblem.train_init_op()

    return tproblem


def manual_forward_pass(tproblem):
    """Reconstructed forward pass with explicit access to data."""
    X, y = tproblem._get_next_batch()
    X = X.to(tproblem._device)
    y = y.to(tproblem._device)

    reduction = "mean"
    assert tproblem.phase == "train"

    outputs = tproblem.net(X)
    loss = tproblem.loss_function(reduction=reduction)(outputs, y)

    return loss


def forward_pass(tproblem, add_regularization_if_available=False):
    """Forward pass in DeepOBS."""
    loss, _ = tproblem.get_batch_loss_and_accuracy(
        add_regularization_if_available=add_regularization_if_available
    )
    return loss


def manual_forward_pass_correct(
    tproblem_cls,
    batch_size,
    seed=0,
    verbose=True,
    add_regularization_if_available=False,
    extend=False,
):
    """Check if manual and DeepOBS forward pass match."""
    try:
        tproblem = set_up_problem(tproblem_cls, batch_size, seed=seed, extend=extend)
        manual_loss = forward_pass(tproblem).item()

        tproblem = set_up_problem(tproblem_cls, batch_size, seed=seed, extend=extend)
        loss = forward_pass(
            tproblem, add_regularization_if_available=add_regularization_if_available
        ).item()

        same = loss == manual_loss
        if verbose:
            name = tproblem_cls.__name__
            same_symbol = "✓" if same else "❌"
            print(
                "{} [{}, l2_reg: {},".format(
                    same_symbol, name, add_regularization_if_available
                ),
                " BackPACK: {}] DeepOBS: {:.5f}, manual: {:.5f}".format(
                    extend, loss, manual_loss,
                ),
            )

        return same

    except Exception as e:
        if verbose:
            name = tproblem_cls.__name__
            fail = "❌"
            print(
                "{} [{}, l2_reg: {}, BackPACK: {}] Raised exception: {}".format(
                    fail, name, add_regularization_if_available, extend, e
                )
            )
        return False


def tproblem_cls_from_str(tproblem_str):
    for tproblem_cls in ALL:
        if tproblem_cls.__name__ == tproblem_str:
            return tproblem_cls

    raise ValueError("Unknwn DeepOBS problem: {}".format(tproblem_str))


if __name__ == "__main__":
    set_data_dir("~/tmp/data_deepobs")

    parser = argparse.ArgumentParser(
        description="Compare manual forward pass with DeepOBS"
    )

    parser.add_argument(
        "tproblem_cls", type=str, help="Name of the DeepOBS testproblem",
    )
    parser.add_argument(
        "--add_regularization_if_available",
        action="store_true",
        help="Add regularization loss",
    )
    parser.add_argument(
        "--extend", action="store_true", help="Extend DeepOBS problem with BackPACK",
    )
    parser.add_argument(
        "--batch_size", type=int, default=3, help="Batch size",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output to command line",
    )

    kwargs = vars(parser.parse_args())
    kwargs["tproblem_cls"] = tproblem_cls_from_str(kwargs["tproblem_cls"])

    manual_forward_pass_correct(**kwargs)
