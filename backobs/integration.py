"""Custom Runner to track statistics. """

from backpack import extend
from deepobs.pytorch.testproblems import (cifar10_3c3d, cifar10_vgg16,
                                          cifar10_vgg19, cifar100_3c3d,
                                          cifar100_allcnnc, cifar100_vgg16,
                                          cifar100_vgg19, cifar100_wrn164,
                                          cifar100_wrn404, fmnist_2c2d,
                                          fmnist_logreg, fmnist_mlp,
                                          fmnist_vae, mnist_2c2d, mnist_logreg,
                                          mnist_mlp, mnist_vae, quadratic_deep,
                                          svhn_3c3d, svhn_wrn164)
from deepobs.pytorch.testproblems.testproblem import TestProblem

ALL_PROBLEMS = [
    cifar10_3c3d,
    cifar10_vgg16,
    cifar10_vgg19,
    cifar100_3c3d,
    cifar100_allcnnc,
    cifar100_vgg16,
    cifar100_vgg19,
    cifar100_wrn164,
    cifar100_wrn404,
    fmnist_2c2d,
    fmnist_logreg,
    fmnist_mlp,
    fmnist_vae,
    mnist_2c2d,
    mnist_logreg,
    mnist_mlp,
    mnist_vae,
    quadratic_deep,
    svhn_3c3d,
    svhn_wrn164,
]

SUPPORTED_PROBLEMS = [
    mnist_logreg,
    fmnist_2c2d,
    cifar10_3c3d,
    cifar100_allcnnc,
]


def integrate_backpack(testproblem, check=True):
    """Add BackPACK functionality to a DeepOBS test problem.

    Parameters:
    -----------
    testproblem : TestProblem instance from deepobs.pytorch
        The testproblem to be integrated.
    check: bool (optional)
        Verify that the testproblem is fully-supported by BackPACK.
        BackPACK does not fully support all testproblems.

    Returns:
    --------
    Extended testproblem.
    """
    if check:
        _check_can_be_integrated(testproblem)

    def extend_loss_func(testproblem):
        testproblem._old_loss = testproblem.loss_function

        def new_lossfunc(reduction="mean"):
            return extend(testproblem._old_loss(reduction=reduction))

        testproblem.loss_function = new_lossfunc

    extend(testproblem.net)
    extend_loss_func(testproblem)

    return testproblem


def _check_can_be_integrated(testproblem):
    """Check if the DeepOBS problem can be extended with BackPACK."""
    testproblem_class = testproblem.__class__

    def check_is_deepobs_problem():
        if not issubclass(testproblem_class, (TestProblem,)):
            raise ValueError("Expect TestProblem, got {}".format(testproblem_class))

    def check_supported_by_backpack(testproblem):
        if not isinstance(testproblem, SUPPORTED_PROBLEMS):
            raise ValueError(
                "{} currently not supported. Working problems: {}".format(
                    testproblem_class, SUPPORTED_PROBLEMS
                )
            )

    check_is_deepobs_problem()
    check_supported_by_backpack(testproblem)
