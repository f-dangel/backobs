"""Accessing the unreduced loss should not change forward/backward."""

from test.utils import check_sizes_and_values, get_reduction_factor, set_up_problem

import pytest
import torch

from backobs.utils import SUPPORTED, UNSUPPORTED
from backpack import backpack, extensions
from deepobs.config import set_data_dir
from deepobs.pytorch.config import set_default_device
from deepobs.pytorch.testproblems import quadratic_deep

FORCE_CPU = True
if FORCE_CPU:
    set_default_device("cpu")


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


@pytest.mark.parametrize("tproblem_cls", SUPPORTED, ids=[p.__name__ for p in SUPPORTED])
def test_same_batch_grad(tproblem_cls, batch_size=3, seed=0):
    """Test individual gradients from unreduced losses match with BackPACK.

    Args:
        tproblem (TestProblem): DeepOBS test problem class.
    """
    # via backpack
    tproblem1 = set_up_problem(
        tproblem_cls, batch_size=batch_size, seed=seed, extend=True
    )
    loss1, acc1 = tproblem1.get_batch_loss_and_accuracy()
    with backpack(extensions.BatchGrad()):
        loss1.backward()

    batch_grad1 = [p.grad_batch for p in tproblem1.net.parameters() if p.requires_grad]

    # via autograd
    tproblem2 = set_up_problem(
        tproblem_cls, batch_size=batch_size, seed=seed, extend=True, unreduced_loss=True
    )
    loss2, acc2 = tproblem2.get_batch_loss_and_accuracy()
    loss2_unreduced = loss2._unreduced_loss
    factor = get_reduction_factor(loss2, loss2_unreduced)

    # backpack assumes N individual losses, but MSELoss does not reduce non-batch
    # axes, so we have to do it manually
    if tproblem_cls == quadratic_deep:
        loss2_unreduced = loss2_unreduced.flatten(start_dim=1).sum(1)

    trainable_params = [p for p in tproblem2.net.parameters() if p.requires_grad]
    batch_grad2 = [
        torch.zeros(batch_size, *p.shape, device=p.device) for p in trainable_params
    ]

    for i in range(batch_size):
        retain_graph = True if i < batch_size - 1 else False
        l_i = loss2_unreduced[i]
        grad = torch.autograd.grad(l_i, trainable_params, retain_graph=retain_graph)
        for param_idx, g in enumerate(grad):
            batch_grad2[param_idx][i] = g * factor

    check_sizes_and_values(batch_grad1, batch_grad2)
