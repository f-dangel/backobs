"""Execute test for forward pass.

Runs a new Python interpreter for each test problem. This is because there
is a bug in BackPACK whenever a model is extended, but none of its extensions
are used. This leads to a memory leak which finally leads to a memory overflow,
as the test problems are not properly freed.
"""

import os
import subprocess

from backobs.integration import ALL_PROBLEMS

if __name__ == "__main__":
    HERE = os.path.abspath(__file__)
    SCRIPT = os.path.join(os.path.dirname(HERE), "test_forward.py")

    batch_size = 3
    verbose = True
    seed = 0

    for extend in [False, True]:
        for add_regularization_if_available in [False, True]:
            for tproblem_cls in ALL_PROBLEMS:

                tproblem_cls = tproblem_cls.__name__
                cmd = ["python"]
                cmd.append(SCRIPT)
                if verbose:
                    cmd.append("--verbose")
                if extend:
                    cmd.append("--extend")
                if add_regularization_if_available:
                    cmd.append("--add_regularization_if_available")
                cmd.append("--batch_size={}".format(batch_size))
                cmd.append("--seed={}".format(seed))
                cmd.append(tproblem_cls)

                # print(" ".join(cmd))
                result = subprocess.run(cmd, capture_output=True)
                print(result.stdout.decode().replace("\n", ""))

            print("\n")

    # os.system(cmd)

    # batch_size = 3

    # PROBLEMS = SUPPORTED_PROBLEMS
    # PROBLEMS = ALL_PROBLEMS

    # def segfault(tproblem_cls):
    #     crash = [
    #         # "cifar100_wrn404",
    #         # "cifar100_wrn404",
    #         # "cifar10_vgg19",
    #         # "cifar100_vgg19",
    #     ]
    #     return tproblem_cls.__name__ in crash

    # def vae(tproblem_cls):
    #     autoencoder = [
    #         # "fmnist_vae",
    #         # "mnist_vae",
    #     ]
    #     return tproblem_cls.__name__ in autoencoder

    # PROBLEMS = [p for p in PROBLEMS if not segfault(p)]
    # PROBLEMS = [p for p in PROBLEMS if not vae(p)]

    # print("==================")
    # print("No regularization:")
    # print("==================")
    # for tproblem_cls in PROBLEMS:
    #     manual_forward_pass_correct(
    #         tproblem_cls,
    #         batch_size,
    #         add_regularization_if_available=False,
    #         extend=False,
    #     )

    # print("====================")
    # print("With regularization:")
    # print("====================")
    # for tproblem_cls in PROBLEMS:
    #     manual_forward_pass_correct(
    #         tproblem_cls, batch_size, add_regularization_if_available=True, extend=False
    #     )
