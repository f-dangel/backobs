"""Check that manual forward pass is same as in DeepOBS.

Background: Need access to input data for testing BackPACK extensions.
"""

import torch
from backobs.integration import integrate_backpack
from deepobs.pytorch.testproblems import mnist_logreg


def set_up_problem(tproblem_cls, batch_size, seed=0):
    """Create problem."""
    torch.manual_seed(seed)

    tproblem = tproblem_cls(batch_size)

    tproblem.set_up()
    tproblem = integrate_backpack(tproblem, check=False)
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


def forward_pass(tproblem):
    """Forward pass in DeepOBS."""
    loss, _ = tproblem.get_batch_loss_and_accuracy()
    return loss


def manual_forward_pass_correct(tproblem_cls, batch_size, seed=0, verbose=True):
    """Check if manual and DeepOBS forward pass match."""
    tproblem = set_up_problem(tproblem_cls, batch_size, seed=seed)
    loss = forward_pass(tproblem).item()

    tproblem = set_up_problem(tproblem_cls, batch_size, seed=seed)
    manual_loss = forward_pass(tproblem).item()

    same = loss == manual_loss

    if verbose:
        name = tproblem_cls.__name__
        same_symbol = "✓" if same else "❌"
        print(
            "{} [{}]\n\tDeepOBS: {:.5f}, manual: {:.5f}, same: {}".format(
                same_symbol, name, loss, manual_loss, same
            )
        )

    return same


if __name__ == "__main__":
    from backobs.integration import ALL_PROBLEMS, SUPPORTED_PROBLEMS
    from deepobs.config import set_data_dir

    set_data_dir("~/tmp/data_deepobs")
    batch_size = 4

    for tproblem_cls in SUPPORTED_PROBLEMS:
        manual_forward_pass_correct(tproblem_cls, batch_size)
