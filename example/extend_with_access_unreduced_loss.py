"""
Basic example how to use BackPACK with DeepOBS and get access to the unreduced loss.
"""

import os

from backobs import extend_with_access_unreduced_loss
from backpack import backpack
from backpack.extensions import (
    KFAC,
    KFLR,
    KFRA,
    BatchGrad,
    BatchL2Grad,
    DiagGGNExact,
    DiagGGNMC,
    DiagHessian,
    SumGradSquared,
    Variance,
)
from deepobs.config import set_data_dir
from deepobs.pytorch.testproblems import mnist_logreg

set_data_dir(os.path.expanduser("~/tmp/data_deepobs"))

# set up the neural net/dataset in DeepOBS
tproblem = mnist_logreg(batch_size=128)
tproblem.set_up()
tproblem.train_init_op()

# extend the problem with BackPACK using backobs
tproblem = extend_with_access_unreduced_loss(tproblem)

# forward pass
batch_loss, _ = tproblem.get_batch_loss_and_accuracy()

# individual loss
savefield = "_unreduced_loss"
individual_loss = getattr(batch_loss, savefield)

print("Individual loss shape:   ", individual_loss.shape)
print("Mini-batch loss:         ", batch_loss)
print("Averaged individual loss:", individual_loss.mean())

# It is still possible to use BackPACK in the backward pass
with backpack(
    BatchGrad(),
    Variance(),
    SumGradSquared(),
    BatchL2Grad(),
    DiagGGNExact(),
    DiagGGNMC(),
    KFAC(),
    KFLR(),
    KFRA(),
    DiagHessian(),
):
    batch_loss.backward()

# print info
for name, param in tproblem.net.named_parameters():
    print(name)
    print("\t.grad.shape:             ", param.grad.shape)
    print("\t.grad_batch.shape:       ", param.grad_batch.shape)
    print("\t.variance.shape:         ", param.variance.shape)
    print("\t.sum_grad_squared.shape: ", param.sum_grad_squared.shape)
    print("\t.batch_l2.shape:         ", param.batch_l2.shape)
    print("\t.diag_ggn_mc.shape:      ", param.diag_ggn_mc.shape)
    print("\t.diag_ggn_exact.shape:   ", param.diag_ggn_exact.shape)
    print("\t.diag_h.shape:           ", param.diag_h.shape)
    print("\t.kfac (shapes):          ", [kfac.shape for kfac in param.kfac])
    print("\t.kflr (shapes):          ", [kflr.shape for kflr in param.kflr])
    print("\t.kfra (shapes):          ", [kfra.shape for kfra in param.kfra])
