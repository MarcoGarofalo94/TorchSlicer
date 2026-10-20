"""The tutorials' code blocks must keep running (docs/tutorials/*.md)."""
import pathlib
import re

import pytest

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs" / "tutorials"


def _blocks(name):
    src = (DOCS / name).read_text()
    # skip illustrations that need downloads or a user-defined pack function
    return [b for b in re.findall(r"```python\n(.*?)```", src, re.S)
            if "from_pretrained" not in b and "my_pack)" not in b]


@pytest.mark.parametrize("name", ["custom_strategy.md", "pack.md", "p2p.md", "sl_settings.md", "offload.md",
                                  "nat.md", "algorithm.md", "privacy.md"])
def test_tutorial_code_runs(name):
    pytest.importorskip("torchvision")
    ns = {}
    for i, block in enumerate(_blocks(name)):
        exec(compile(block, f"{name}#{i}", "exec"), ns)
