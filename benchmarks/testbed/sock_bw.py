"""Raw TCP baseline between two hosts: what one socket moves, without TorchSlicer.

    python sock_bw.py serve PORT                 # on host B
    python sock_bw.py send HOST:PORT [MB] [N]    # on host A: N round trips of MB each way

Prints RTT (1-byte ping-pong, median of 200) and goodput per direction (median of N), so
the testbed's ms/batch can be read against the wire.
"""
import json
import socket
import statistics
import sys
import time


def _recv(s, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = s.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError("peer closed")
        got += k
    return buf


def serve(port):
    ls = socket.create_server(("0.0.0.0", port))
    c, _ = ls.accept()
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        head = c.recv(9)
        if not head:
            return
        op, n = head[:1], int.from_bytes(head[1:], "big")
        if op == b"p":
            c.sendall(b"p")
        elif op == b"u":          # upload: receive n bytes, ack
            _recv(c, n)
            c.sendall(b"k")
        elif op == b"d":          # download: send n bytes
            c.sendall(bytes(n))


def send(addr, mb=64, n=5):
    host, port = addr.rsplit(":", 1)
    s = socket.create_connection((host, int(port)))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    rtt = []
    for _ in range(200):
        t = time.perf_counter(); s.sendall(b"p" + (0).to_bytes(8, "big")); _recv(s, 1)
        rtt.append(time.perf_counter() - t)
    size, payload = mb * 2**20, bytes(mb * 2**20)
    up, down = [], []
    for _ in range(n):
        t = time.perf_counter(); s.sendall(b"u" + size.to_bytes(8, "big")); s.sendall(payload); _recv(s, 1)
        up.append(size / (time.perf_counter() - t))
        t = time.perf_counter(); s.sendall(b"d" + size.to_bytes(8, "big")); _recv(s, size)
        down.append(size / (time.perf_counter() - t))
    print("SOCK " + json.dumps({"rtt_ms": statistics.median(rtt) * 1e3,
                                "up_MBps": statistics.median(up) / 1e6,
                                "down_MBps": statistics.median(down) / 1e6, "mb": mb, "n": n}))


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        serve(int(sys.argv[2]))
    else:
        send(sys.argv[2], *(int(a) for a in sys.argv[3:]))
