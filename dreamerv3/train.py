import os
import sys


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    entrypoint = os.path.join(repo_root, "dreamer.py")
    os.execv(sys.executable, [sys.executable, entrypoint, *sys.argv[1:]])


if __name__ == "__main__":
    main()
