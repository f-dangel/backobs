"""Utility functions and groups of testproblems."""

from deepobs.pytorch.testproblems import (
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
)
from deepobs.pytorch.testproblems.testproblem import (
    TestProblem,
    UnregularizedTestproblem,
)

ALL = (
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
)
BATCH_NORM = (
    cifar100_wrn164,
    cifar100_wrn404,
    svhn_wrn164,
)
VAE = (
    fmnist_vae,
    mnist_vae,
)
REGRESSION = (quadratic_deep,)
REGULARIZED = tuple(p for p in ALL if not issubclass(p, UnregularizedTestproblem))
SUPPORTED = tuple(p for p in ALL if not (p in BATCH_NORM or p in VAE))
UNSUPPORTED = tuple(p for p in ALL if (p in BATCH_NORM or p in VAE))


def has_no_accuracy(tproblem: TestProblem):
    """Return whether accuracy is defined for a DeepOBS task."""
    return isinstance(tproblem, REGRESSION)
