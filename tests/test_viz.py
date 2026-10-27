"""The slicing picture (torchslicer.viz): what it draws must be what ts.slice would do."""
import json
import re
import shutil
import subprocess

import pytest
import torch
import torch.nn as nn

import torchslicer as ts


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b, self.c = nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 4)

    def forward(self, x):
        h = self.a(x)
        return self.c(self.b(h) + h)          # the skip h spans b: no cut after a's successor


class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.mid = nn.Embedding(50, 8), nn.Linear(8, 8)
        self.head = nn.Linear(8, 50, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, t):
        return self.head(self.mid(self.emb(t)))


def _data(page):
    return json.loads(re.search(r"const D = (.*?);\n", page).group(1))


def test_view_draws_the_cuts_valid_gaps_and_skip_arcs_of_the_sliced_model():
    torch.manual_seed(0)
    s = ts.slice(Residual(), cuts=["a"])
    v = s.view(example_input=torch.randn(3, 8))
    d = _data(v.html())
    assert [n["name"] for n in d["nodes"]] == s.graph.names()
    assert d["cuts"] == [p.layer_indices[-1] for p in s.partitions[:-1]] == [0]
    assert d["valid"] == s.graph.valid_cuts()
    add = s.graph.index_of("add")
    assert [0, add] in d["arcs"]                                   # a -> add skips b
    assert d["nodes"][0]["shape"] == "[3, 8]" and d["nodes"][0]["out"] == 3 * 8 * 4
    assert 'class="gap ok on" data-g="0"' in v.svg()               # the cut is drawn statically
    assert "var(--" not in v.svg()                                 # colours written out


def test_view_of_a_graph_has_no_cuts_and_marks_tied_weights_without_changing_the_model():
    torch.manual_seed(0)
    model, x = Tied().train(), torch.randint(0, 50, (4,))
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    g = ts.trace(model)
    d = _data(g.view(example_input=x).html())
    assert d["cuts"] == [] and d["valid"] == g.valid_cuts()
    assert d["tied"] == [[g.index_of("emb"), g.index_of("head")]]
    assert d["nodes"][g.index_of("head")]["tied"]
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())


def test_view_of_packed_stages_cuts_by_index():
    stages = [nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)]
    s = ts.slice(nn.Sequential(*stages), cuts=[1], pack=lambda m: list(m))
    d = _data(s.view().html())
    assert d["packed"] and d["cuts"] == [1] and d["arcs"] == []


def test_view_page_escapes_names_and_its_script_parses(tmp_path):
    v = ts.slice(Residual(), cuts=["a"]).view(title="</script><b>x</b>")
    page = v.html()
    assert "</script><b>" not in page.split("<script>")[1].rsplit("</script>", 1)[0]
    assert v._repr_html_().startswith("<iframe srcdoc=")
    assert v.save(tmp_path / "p.html").read_text() == page
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed: script syntax not checked")
    js = tmp_path / "page.js"
    js.write_text(re.search(r"<script>(.*)</script>", page, re.S).group(1))
    subprocess.run([node, "--check", str(js)], check=True)
