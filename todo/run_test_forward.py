"""Execute test for forward pass.

Runs a new Python interpreter for each test problem. This is because there
is a bug in BackPACK whenever a model is extended, but none of its extensions
are used. This leads to a memory leak which finally leads to a memory overflow,
as the test problems are not properly freed.
"""

import os
import subprocess

from backobs.integration import ALL

if __name__ == "__main__":
    HERE = os.path.abspath(__file__)
    SCRIPT = os.path.join(os.path.dirname(HERE), "test_forward.py")

    batch_size = 3
    verbose = True
    seed = 0

    for extend in [False, True]:
        for add_regularization_if_available in [False, True]:
            for tproblem_cls in ALL:

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

                result = subprocess.run(cmd, capture_output=True)
                print(result.stdout.decode().replace("\n", ""))

            print("\n")
