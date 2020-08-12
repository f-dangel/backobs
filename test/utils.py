import random

import numpy
import torch

from backobs.integration import extend as backobs_extend
from backobs.integration import \
    extend_with_access_unreduced_loss as \
    backobs_extend_with_access_unreduced_loss


def set_deepobs_seed(seed=0):
    """Set all seeds used by DeepOBS."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def set_up_problem(
    tproblem_cls,
    batch_size,
    force_no_l2_reg=True,
    seed=None,
    extend=False,
    unreduced_loss=False,
):
    """Create problem with neural network, and set to train mode."""
    if seed is not None:
        set_deepobs_seed(0)

    if force_no_l2_reg:
        tproblem = tproblem_cls(batch_size, l2_reg=0.0)
    else:
        tproblem = tproblem_cls(batch_size)

    tproblem.set_up()
    tproblem.train_init_op()

    if unreduced_loss and not extend:
        raise ValueError("To use unreduced_loss,  enable the extend option.")

    if extend:
        if unreduced_loss:
            backobs_extend_with_access_unreduced_loss(tproblem)
        else:
            tproblem = backobs_extend(tproblem)

    return tproblem


atol = 1e-5
rtol = 1e-5


def report_nonclose_values(x, y):
    x_numpy = x.data.cpu().numpy().flatten()
    y_numpy = y.data.cpu().numpy().flatten()

    close = numpy.isclose(x_numpy, y_numpy, atol=atol, rtol=rtol)
    where_not_close = numpy.argwhere(numpy.logical_not(close))
    for idx in where_not_close:
        x, y = x_numpy[idx], y_numpy[idx]
        print("{} versus {}. Ratio of {}".format(x, y, y / x))


def check_sizes_and_values(*plists, atol=atol, rtol=rtol):
    check_sizes(*plists)
    list1, list2 = plists
    check_values(list1, list2, atol=atol, rtol=rtol)


def check_sizes(*plists):
    for i in range(len(plists) - 1):
        assert len(plists[i]) == len(plists[i + 1])

    for params in zip(*plists):
        for i in range(len(params) - 1):
            assert params[i].size() == params[i + 1].size()


def check_values(list1, list2, atol=atol, rtol=rtol):
    for i, (g1, g2) in enumerate(zip(list1, list2)):
        print(i)
        print(g1.size())
        report_nonclose_values(g1, g2)
        assert torch.allclose(g1, g2, atol=atol, rtol=rtol)
