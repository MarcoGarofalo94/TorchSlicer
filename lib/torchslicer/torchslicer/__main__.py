"""``python -m torchslicer <command>``. Commands: ``launch`` (see
``torchslicer.deploy.launch``) and ``relay`` (see ``torchslicer.transport.tunnel``)."""

import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m torchslicer launch SCRIPT -c CONFIG --target "
              "{local,processes,compose,k8s,baremetal} [options]\n"
              "       python -m torchslicer launch --help\n"
              "       python -m torchslicer relay [--port 50060] [--host [::]] [--max-tunnels 64]")
        return 0 if argv else 2
    if argv[0] == "launch":
        from .deploy.launch import main as launch
        return launch(argv[1:])
    if argv[0] == "relay":
        import argparse
        from .monitor.process_logger import configure
        from .transport.tunnel import run_relay
        p = argparse.ArgumentParser(prog="python -m torchslicer relay",
                                    description="Forward calls between TorchSlicer nodes behind NAT.")
        p.add_argument("--port", type=int, default=50060)
        p.add_argument("--host", default="[::]")
        p.add_argument("--max-tunnels", type=int, default=64,
                       help="open tunnels served at once (one server thread each)")
        args = p.parse_args(argv[1:])
        configure("INFO")
        run_relay(args.port, host=args.host, max_tunnels=args.max_tunnels)
        return 0
    print(f"unknown command {argv[0]!r} (expected: launch, relay)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
