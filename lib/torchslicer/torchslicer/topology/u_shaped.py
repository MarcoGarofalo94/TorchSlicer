"""
UShapedTopology — U-shaped split learning (Vepakomma et al., 2018).

Reference: "Split Learning for Health: Distributed Deep Learning without
Sharing Raw Patient Data", Vepakomma et al., 2018.
https://arxiv.org/abs/1812.00564

Design
------
The model is split into three logical sections:

    Client (bottom) ──► Server (middle) ──► Client (top / head)
                   ◄──────────────────── gradient

* The **client** holds the first ``n_client_bottom`` partitions (embeddings,
  early feature extraction) and the last ``n_client_head`` partitions
  (classification head, loss computation).
* The **server** holds all middle partitions.
* The server **never sees** raw inputs or labels — it only processes
  intermediate activations (smashed data).

In **local** execution (single process) the computation is identical to the
sequential pipeline; the topology is metadata that describes intent and
enables privacy analysis.  In future **distributed** execution the client–
server–client routing will be driven by this assignment.

Example::

    # 3-partition model: embed | blocks | head
    sliced = ts.slice(model, n=3, pack=pack_bert_seq_cls)
    sliced.topology = ts.UShapedTopology()
    roles = sliced.topology.assign(sliced.partitions)
    # {"client": [p0, p2], "server": [p1]}

    # Deeper model: embed | block0 | block1 | head
    sliced = ts.slice(model, n=4)
    sliced.topology = ts.UShapedTopology(n_client_bottom=1, n_client_head=1)
"""

from .base import BaseSplitTopology


class UShapedTopology(BaseSplitTopology):
    """U-shaped split: client owns bottom + head, server owns middle.

    Args:
        n_client_bottom: Number of partitions on the client's bottom section
                         (default 1 — just the embedding / first partition).
        n_client_head:   Number of partitions on the client's head section
                         (default 1 — just the last partition / classifier).

    Raises:
        ValueError: if ``n_client_bottom + n_client_head >= len(partitions)``
                    (no middle section would remain for the server).
    """

    def __init__(self, n_client_bottom: int = 1, n_client_head: int = 1):
        if n_client_bottom < 1 or n_client_head < 1:
            raise ValueError(
                "n_client_bottom and n_client_head must each be at least 1."
            )
        self.n_client_bottom = n_client_bottom
        self.n_client_head   = n_client_head

    def assign(self, partitions: list) -> dict[str, list]:
        n = len(partitions)
        n_server = n - self.n_client_bottom - self.n_client_head
        if n_server < 1:
            raise ValueError(
                f"UShapedTopology requires at least "
                f"{self.n_client_bottom + self.n_client_head + 1} partitions "
                f"(n_client_bottom={self.n_client_bottom}, "
                f"n_client_head={self.n_client_head}, need ≥1 server partition). "
                f"Got n={n}."
            )
        bottom = partitions[:self.n_client_bottom]
        middle = partitions[self.n_client_bottom: self.n_client_bottom + n_server]
        head   = partitions[self.n_client_bottom + n_server:]
        return {
            "client": list(bottom) + list(head),
            "server": list(middle),
        }

    def loss_owner(self, partitions: list) -> str:
        return "client"

    def describe(self) -> str:
        b, h = self.n_client_bottom, self.n_client_head
        return (
            f"client(bottom×{b}) → server(middle) → client(head×{h}, loss) "
            f"→ server(grad) → client(grad)"
        )
