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


def integrate_backpack(tproblem, check=True):
    """Add BackPACK functionality to a DeepOBS test problem.

    Parameters:
    -----------
    tproblem : TestProblem instance from deepobs.pytorch
        The tproblem to be integrated.
    check: bool (optional)
        Verify that the tproblem is fully-supported by BackPACK.
        BackPACK does not fully support all tproblems.

    Returns:
    --------
    Extended tproblem.
    """
    original_loss_function_savefield = "_old_loss_function"

    if check:
        _check_can_be_integrated(tproblem)

    def already_integrated(tproblem):
        return hasattr(tproblem, original_loss_function_savefield)

    if already_integrated(tproblem):
        raise RuntimeError("Test problem is already extended")

    def extend_loss_function(tproblem):
        setattr(tproblem, original_loss_function_savefield, tproblem.loss_function)

        def new_loss_function(reduction="mean"):
            original_loss_function = getattr(tproblem, original_loss_function_savefield)
            return extend(original_loss_function(reduction=reduction))

        tproblem.loss_function = new_loss_function

    extend(tproblem.net)
    extend_loss_function(tproblem)

    return tproblem


def _check_can_be_integrated(tproblem):
    """Check if the DeepOBS problem can be extended with BackPACK."""
    tproblem_class = tproblem.__class__

    def check_is_deepobs_problem():
        if not issubclass(tproblem_class, (TestProblem,)):
            raise ValueError("Expect TestProblem, got {}".format(tproblem_class))

    def check_supported_by_backpack(tproblem):
        if not isinstance(tproblem, SUPPORTED_PROBLEMS):
            raise ValueError(
                "{} currently not supported. Working problems: {}".format(
                    tproblem_class, SUPPORTED_PROBLEMS
                )
            )

    check_is_deepobs_problem()
    check_supported_by_backpack(tproblem)
