"""Check that manual mean over individual losses is same as loss in DeepOBS.

Background: Check structure of loss to be sum of individual losses.
"""
import torch

from backobs.utils import has_batchnorm, has_dropout
from deepobs.config import set_data_dir
from test_forward import forward_pass, set_up_problem, tproblem_cls_from_str


def manual_forward_pass_loop(tproblem):
    """Reconstructed forward pass (for loop over batch) with data access."""
    batch_size = tproblem._batch_size

    X, y = tproblem._get_next_batch()
    X = X.to(tproblem._device)
    y = y.to(tproblem._device)

    reduction = "mean"
    assert tproblem.phase == "train"

    losses = []

    for n in range(batch_size):
        x_n = X[n].unsqueeze(0)
        y_n = y[n].unsqueeze(0)

        f_n = tproblem.net(x_n)
        l_n = tproblem.loss_function(reduction=reduction)(f_n, y_n)
        losses.append(l_n)

    loss = torch.tensor(losses).mean()
    return loss


def manual_forward_pass_loop_correct(
    tproblem_cls,
    batch_size,
    seed=0,
    verbose=True,
    add_regularization_if_available=False,
    extend=False,
):
    """Check if manual for-loop and DeepOBS forward pass match."""
    try:
        tproblem = set_up_problem(tproblem_cls, batch_size, seed=seed, extend=extend)
        manual_loss = manual_forward_pass_loop(tproblem)

        tproblem = set_up_problem(tproblem_cls, batch_size, seed=seed, extend=extend)
        loss = forward_pass(
            tproblem, add_regularization_if_available=add_regularization_if_available
        )

        same = torch.allclose(manual_loss, loss, atol=1e-5)
        if verbose:
            name = tproblem_cls.__name__
            same_symbol = "✓" if same else "❌"
            print(
                "{} [{}, l2_reg: {}, BackPACK: {}] DeepOBS: {:.5f}, manual for-loop: {:.5f}".format(
                    same_symbol,
                    name,
                    add_regularization_if_available,
                    extend,
                    loss,
                    manual_loss,
                )
            )
            if not same:
                has_bn = has_batchnorm(tproblem.net)
                has_do = has_dropout(tproblem.net)
                print(", BatchNorm? {}, Dropout? {}".format(has_bn, has_do))

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


if __name__ == "__main__":
    import argparse

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

    manual_forward_pass_loop_correct(**kwargs)
