"""Check individual gradient computation with BackPACK."""

import os
import subprocess

from backobs.integration import ALL_PROBLEMS

if __name__ == "__main__":
    HERE = os.path.abspath(__file__)
    SCRIPT = os.path.join(os.path.dirname(HERE), "test_batch_grad.py")

    batch_size = 2
    verbose = True
    seed = 0

    for tproblem_cls in ALL_PROBLEMS:
        tproblem_cls = tproblem_cls.__name__
        cmd = ["python"]
        cmd.append(SCRIPT)
        if verbose:
            cmd.append("--verbose")
        cmd.append("--batch_size={}".format(batch_size))
        cmd.append("--seed={}".format(seed))
        cmd.append(tproblem_cls)

        result = subprocess.run(cmd, capture_output=True)
        result = result.stdout.decode().replace("\n", "")
        if result == "":
            fail = "❌"
            print(
                "{} [{},individual gradients] Killed by OS".format(fail, tproblem_cls)
            )
        else:
            print(result)
