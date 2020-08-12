"""Extending a DeepOBS problem should not change forward/backward."""

from test.utils import check_sizes_and_values, set_up_problem

import pytest
import torch

from backobs import extend as backobs_extend
from backobs.utils import REGULARIZED, SUPPORTED, UNSUPPORTED
from backpack import backpack
from deepobs.config import set_data_dir
from deepobs.pytorch.config import set_default_device

set_data_dir("~/tmp/data_deepobs")

FORCE_CPU = True
if FORCE_CPU:
    set_default_device("cpu")


@pytest.mark.parametrize(
    "tproblem_cls", UNSUPPORTED, ids=[p.__name__ for p in UNSUPPORTED]
)
def test_unsupported_problems(tproblem_cls, batch_size=3):
    """Test that unsupported problems cannot be extended.

    Args:
        tproblem (TestProblem): DeepOBS test problem class.
    """
    with pytest.raises(NotImplementedError):
        set_up_problem(tproblem_cls, batch_size=batch_size, extend=True)


@pytest.mark.parametrize(
    "tproblem_cls", REGULARIZED, ids=[p.__name__ for p in REGULARIZED]
)
def test_no_l2_reg(tproblem_cls, batch_size=3):
    """ℓ₂ regularization is not supported.

    Args:
        tproblem (TestProblem): DeepOBS test problem class.
    """
    with pytest.raises(NotImplementedError):
        set_up_problem(
            tproblem_cls, batch_size=batch_size, force_no_l2_reg=False, extend=True
        )


@pytest.mark.parametrize("tproblem_cls", SUPPORTED, ids=[p.__name__ for p in SUPPORTED])
def test_already_extended(tproblem_cls, batch_size=3):
    """Extending an already extended test problem is not supported.

    Args:
        tproblem (TestProblem): DeepOBS test problem class.
    """
    tproblem = set_up_problem(tproblem_cls, batch_size=batch_size, extend=True)

    with pytest.raises(ValueError):
        tproblem = backobs_extend(tproblem)


@pytest.mark.parametrize("tproblem_cls", SUPPORTED, ids=[p.__name__ for p in SUPPORTED])
def test_same_forward(tproblem_cls, batch_size=3, seed=0):
    """Test that extended problem has the same forward pass.

    Args:
        tproblem (TestProblem): DeepOBS test problem class.
    """
    tproblem1 = set_up_problem(tproblem_cls, batch_size=batch_size, seed=seed)
    loss1, acc1 = tproblem1.get_batch_loss_and_accuracy()

    tproblem2 = set_up_problem(
        tproblem_cls, batch_size=batch_size, seed=seed, extend=True
    )
    loss2, acc2 = tproblem2.get_batch_loss_and_accuracy()
    # (Memory leak) need manual triggering of BackPACK IO deletion
    with backpack():
        loss2.backward()

    assert torch.allclose(loss1, loss2)
    assert torch.allclose(torch.tensor(acc1), torch.tensor(acc2))


@pytest.mark.parametrize("tproblem_cls", SUPPORTED, ids=[p.__name__ for p in SUPPORTED])
def test_same_backward(tproblem_cls, batch_size=3, seed=0):
    """Test that extended problem has the same backward pass.

    Args:
        tproblem (TestProblem): DeepOBS test problem class.
    """
    tproblem1 = set_up_problem(tproblem_cls, batch_size=batch_size, seed=seed)
    loss1, acc1 = tproblem1.get_batch_loss_and_accuracy()
    loss1.backward()
    grad1 = [p.grad for p in tproblem1.net.parameters() if p.requires_grad]

    tproblem2 = set_up_problem(
        tproblem_cls, batch_size=batch_size, seed=seed, extend=True
    )
    loss2, acc2 = tproblem2.get_batch_loss_and_accuracy()
    with backpack():
        loss2.backward()
        grad2 = [p.grad for p in tproblem2.net.parameters() if p.requires_grad]

    check_sizes_and_values(grad1, grad2)
