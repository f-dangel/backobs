"""Print supported and unsupported test problems."""

from backobs.utils import ALL_PROBLEMS, SUPPORTED_PROBLEMS

if __name__ == "__main__":
    print("Overview of DeepOBS problems")
    print("============================")
    print("Supported:")

    for problem_cls in SUPPORTED_PROBLEMS:
        print(f"\t✔ {problem_cls.__name__}")

    print("Not supported:")
    for problem_cls in ALL_PROBLEMS:
        if problem_cls not in SUPPORTED_PROBLEMS:
            print(f"\t❌ {problem_cls.__name__}")
