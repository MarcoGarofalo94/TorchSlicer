"""Gallery of slicing cases: which model, which strategy, what you get.

    python examples/split_gallery.py            # from the repo root

Every case prints the partitions (``SlicedModel.summary()``) and checks that the
slices, chained, compute exactly what the unsplit model computes (eval mode).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torchslicer as ts
from torchslicer.core.split_layer import SplitLayer


def check(sliced, model, x):
    """max |sliced(x) - model(x)| with the slices chained in order."""
    model.eval()
    with torch.no_grad():
        out = x
        for p in sliced.partitions:
            out = SplitLayer([sliced.graph.nodes[i].module for i in p.layer_indices],
                             predecessors=p.predecessors).eval()(out)
        return (out - model(x)).abs().max().item()


def _show(v):
    if torch.is_tensor(v):
        return f"<tensor {tuple(v.shape)}>"
    if callable(v):
        return v.__name__
    return repr(v)


def case(title, model, x, **kwargs):
    print(f"\n### {title}\n    ts.slice({type(model).__name__}, "
          + ", ".join(f"{k}={_show(v)}" for k, v in kwargs.items()) + ")")
    try:
        sliced = ts.slice(model, **kwargs)
    except Exception as e:
        print(f"    -> {type(e).__name__}: {str(e).splitlines()[0]}")
        return
    for line in sliced.summary().splitlines():
        print("    " + line)
    print(f"    max|sliced - unsplit| = {check(sliced, model, x):.1e}")


# ── models ────────────────────────────────────────────────────────────────────

def dnn():
    return nn.Sequential(nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 10))


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.pool1 = nn.Conv2d(3, 16, 3, padding=1), nn.MaxPool2d(2)
        self.conv2, self.pool2 = nn.Conv2d(16, 64, 3, padding=1), nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))          # functional relu: a graph node since M1
        x = self.pool2(F.relu(self.conv2(x)))
        return self.fc(x.view(x.size(0), -1))         # dynamic view: kept exact since M1


class Untraceable(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(32, 4, 64, batch_first=True, dropout=0.0)
                                    for _ in range(4))
        self.head = nn.Linear(32, 100)

    def forward(self, ids):
        x = self.embed(ids)
        for blk in self.blocks:
            if x.abs().mean() > 0:                    # data-dependent control flow: fx can't trace
                x = blk(x)
        return self.head(x[:, -1])


def pack_untraceable(model):
    class LastToken(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head

        def forward(self, x):
            return self.head(x[:, -1])
    return [ts.SimpleEmbedStage(model.embed), *[ts.BlockStage(b) for b in model.blocks], LastToken(model.head)]


def main():
    torch.manual_seed(0)
    x_vec, x_img, x_ids = torch.randn(8, 32), torch.randn(8, 3, 32, 32), torch.randint(0, 100, (8, 16))

    print("=" * 78 + "\n1. DNN (nn.Sequential MLP): 7 nodes, every cut is valid")
    m = dnn()
    case("uniform: equal node count", m, x_vec, n=3)
    case("params: equal parameter count", m, x_vec, strategy="params", n=3)
    case("flops: equal forward FLOPs", m, x_vec, strategy="flops", n=3, example_input=x_vec)
    case("named cuts (children of a Sequential are named '0', '1', ...)", m, x_vec, cuts=["1", "3"])
    case("too many partitions for a cost strategy", m, x_vec, strategy="params", n=8)

    print("\n" + "=" * 78 + "\n2. Small CNN with functional ops (F.relu, x.view(x.size(0), -1))")
    m = SmallCNN()
    print("\n" + "\n".join("    " + l for l in ts.trace(m).summary().splitlines()))
    case("uniform", m, x_img, n=3)
    case("named cut after the first pooling layer", m, x_img, cuts=["pool1"])
    case("named cut between size() and view(): both read relu_1, so 2 values would cross", m, x_img, cuts=["size"])

    print("\n" + "=" * 78 + "\n3. ResNet-18: layer1..layer4 are atomic by default")
    m = torchvision.models.resnet18(num_classes=10)
    case("uniform n=4 over 11 top-level nodes", m, x_img, n=4)
    case("uniform n=6 (crashed in 0.2: a slice started with ReLU(inplace=True))", m, x_img, n=6)
    case("params n=4: layer4 alone holds ~75% of the weights", m, x_img, strategy="params", n=4)
    case("params n=4 + expand: blocks become nodes, so balance improves", m, x_img, strategy="params", n=4,
         expand=["layer3", "layer4"])
    case("flops n=3, fully expanded (only block boundaries are valid cuts)", m, x_img, strategy="flops", n=3,
         example_input=x_img, expand="all")
    case("profiled n=3 (measured fwd+bwd time per node)", m, x_img, strategy="profiled", n=3,
         example_input=x_img, expand=["layer1", "layer2", "layer3", "layer4"])
    case("named cuts with a dotted name: traces into layer3", m, x_img, cuts=["layer2", "layer3.0"])
    case("named cut inside a residual block: refused", m, x_img, cuts=["layer3.0.conv1"])

    print("\n" + "=" * 78 + "\n4. torchvision models that 0.2 split wrongly")
    for name in ["mobilenet_v2", "densenet121"]:
        m = getattr(torchvision.models, name)(num_classes=10)
        case(f"{name}, params n=2, expand='all'", m, x_img, strategy="params", n=2, expand="all")

    print("\n" + "=" * 78 + "\n5. A model fx can't trace -> TraceError, then pack=")
    m = Untraceable()
    case("no pack: refused, no silent fallback", m, x_ids, n=2)
    case("pack: embedding, 4 blocks, head = 6 stages, uniform", m, x_ids, n=3, pack=pack_untraceable)
    case("pack + params", m, x_ids, strategy="params", n=3, pack=pack_untraceable)
    case("pack + named cuts by stage index", m, x_ids, cuts=[2], pack=pack_untraceable)
    case("pack + expand: refused (packed stages are not traced)", m, x_ids, n=2, pack=pack_untraceable,
         expand="all")

    try:
        from transformers import GPT2Config, GPT2LMHeadModel
        from examples.pack.gpt2 import pack_gpt2
    except ImportError:
        return
    print("\n" + "=" * 78 + "\n6. Hugging Face GPT-2 (random init, 4 layers): needs pack=")
    m = GPT2LMHeadModel(GPT2Config(vocab_size=100, n_positions=16, n_embd=64, n_layer=4, n_head=4))
    ref = nn.Sequential(*pack_gpt2(m))

    def pack_this_gpt2(_):          # stages of `m`, so the check runs against the same weights
        return pack_gpt2(m)
    case("no pack", m, x_ids, n=2)
    print("    (checked against the chained pack stages, which equal the HF forward)")
    case("pack_gpt2 + uniform", ref, x_ids, n=3, pack=pack_this_gpt2)
    case("pack_gpt2 + params (tied embedding/head counted once)", ref, x_ids, strategy="params", n=3,
         pack=pack_this_gpt2)


if __name__ == "__main__":
    main()
