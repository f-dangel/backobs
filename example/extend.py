"""Basic example showing how to extend a DeepOBS problem with BackPACK."""

from backobs import extend
from backpack import backpack
from backpack.extensions import (KFAC, KFLR, KFRA, BatchGrad, BatchL2Grad,
                                 DiagGGNExact, DiagGGNMC, DiagHessian,
                                 SumGradSquared, Variance)
from deepobs.config import set_data_dir
from deepobs.pytorch.testproblems import mnist_logreg

set_data_dir("~/tmp/data_deepobs")

# set up the neural net/dataset in DeepOBS
tproblem = mnist_logreg(batch_size=128)
tproblem.set_up()
tproblem.train_init_op()

# extend the problem with BackPACK using backobs
tproblem = extend(tproblem)

# forward pass
batch_loss, _ = tproblem.get_batch_loss_and_accuracy()

# backward pass
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
