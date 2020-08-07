"""Accessing the unreduced loss should not change forward/backward."""

from test.utils import check_sizes_and_values, set_up_problem

import pytest
import torch

from backobs.utils import SUPPORTED, UNSUPPORTED
from backpack import backpack
from deepobs.config import set_data_dir

set_data_dir("~/tmp/data_deepobs")


@pytest.mark.parametrize("tproblem_cls", SUPPORTED, ids=[p.__name__ for p in SUPPORTED])
def test_unreduced_correct_reduction(tproblem_cls, batch_size=3, seed=0):
    """Test that extended problem with access to unreduced loss has same forward pass.

    Args:
        tproblem (TestProblem): DeepOBS test problem class.
    """
    tproblem = set_up_problem(
        tproblem_cls, batch_size=batch_size, seed=seed, extend=True, unreduced_loss=True
    )
    loss, _ = tproblem.get_batch_loss_and_accuracy()
    # (Memory leak) need manual triggering of BackPACK IO deletion
    with backpack():
        loss.backward()

    savefield = "_unreduced_loss"
    unreduced = getattr(loss, savefield)
    mean = unreduced.mean()

    check_sizes_and_values([loss], [mean])


@pytest.mark.parametrize("tproblem_cls", SUPPORTED, ids=[p.__name__ for p in SUPPORTED])
def test_same_forward(tproblem_cls, batch_size=3, seed=0):
    """Test that extended problem with access to unreduced loss has same forward pass.

    Args:
        tproblem (TestProblem): DeepOBS test problem class.
    """
    tproblem1 = set_up_problem(tproblem_cls, batch_size=batch_size, seed=seed)
    loss1, acc1 = tproblem1.get_batch_loss_and_accuracy()

    tproblem2 = set_up_problem(
        tproblem_cls, batch_size=batch_size, seed=seed, extend=True, unreduced_loss=True
    )
    loss2, acc2 = tproblem2.get_batch_loss_and_accuracy()
    # (Memory leak) need manual triggering of BackPACK IO deletion
    with backpack():
        loss2.backward()

    assert torch.allclose(loss1, loss2)
    assert torch.allclose(torch.tensor(acc1), torch.tensor(acc2))


@pytest.mark.parametrize("tproblem_cls", SUPPORTED, ids=[p.__name__ for p in SUPPORTED])
def test_same_backward(tproblem_cls, batch_size=3, seed=0):
    """Test that extended problem with access to unreduced los has the same backward pass.

    Args:
        tproblem (TestProblem): DeepOBS test problem class.
    """
    tproblem1 = set_up_problem(tproblem_cls, batch_size=batch_size, seed=seed)
    loss1, acc1 = tproblem1.get_batch_loss_and_accuracy()
    loss1.backward()
    grad1 = [p.grad for p in tproblem1.net.parameters() if p.requires_grad]

    tproblem2 = set_up_problem(
        tproblem_cls, batch_size=batch_size, seed=seed, extend=True, unreduced_loss=True
    )
    loss2, acc2 = tproblem2.get_batch_loss_and_accuracy()
    with backpack():
        loss2.backward()
        grad2 = [p.grad for p in tproblem2.net.parameters() if p.requires_grad]

    check_sizes_and_values(grad1, grad2)
