"""``python -m torchslicer <command>``. Commands: ``launch`` (see
``torchslicer.deploy.launch``)."""

import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m torchslicer launch SCRIPT -c CONFIG --target "
              "{local,processes,compose,k8s,baremetal} [options]\n"
              "       python -m torchslicer launch --help")
        return 0 if argv else 2
    if argv[0] == "launch":
        from .deploy.launch import main as launch
        return launch(argv[1:])
    print(f"unknown command {argv[0]!r} (expected: launch)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
