import os
import subprocess

PROBLEMS = [
    "cifar10_3c3d",
    "cifar10_vgg16",
    "cifar10_vgg19",
    "cifar100_3c3d",
    "cifar100_allcnnc",
    "cifar100_vgg16",
    "cifar100_vgg19",
    "cifar100_wrn164",
    "cifar100_wrn404",
    "fmnist_2c2d",
    "fmnist_logreg",
    "fmnist_mlp",
    "fmnist_vae",
    "mnist_2c2d",
    "mnist_logreg",
    "mnist_mlp",
    "mnist_vae",
    "quadratic_deep",
    "svhn_3c3d",
    "svhn_wrn164",
]
HERE = os.path.dirname(os.path.realpath(__file__))
FILE = os.path.join(HERE, "example/run.py")

HYPERPARAMS = "-N 1"

RESULTS = {}


def print_results():
    TEMPLATE = r"{:20}| {}"

    print("\nSUMMARY\n-------")
    print(TEMPLATE.format("PROBLEM", "SUPPORTED"))
    for problem, supported in RESULTS.items():
        print(TEMPLATE.format(problem, supported))


for testproblem in PROBLEMS:

    print("\nRunning {}".format(testproblem))

    exit_code = subprocess.call(["python", FILE, testproblem, HYPERPARAMS])
    crash = exit_code != 0

    RESULTS[testproblem] = not crash

    print_results()
