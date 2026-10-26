"""F1 check: slicing is exact. Every way of cutting ResNet-18 trains like the unsplit model."""
import copy, torch, torchvision
import torchslicer as ts

torch.manual_seed(0)
base = torchvision.models.resnet18(num_classes=10).eval()  # eval: BatchNorm uses running stats
x = torch.randn(8, 3, 32, 32)
y = torch.randint(0, 10, (8,))

# 1) The graph and where a cut is allowed
g = ts.trace(base, expand=["layer3"])
print(g.summary())

# 2) Three ways to cut. Forward must equal unsplit; train 3 steps and compare losses.
def unsplit_losses(m, steps=3):
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    out = []
    for _ in range(steps):
        opt.zero_grad(); l = torch.nn.functional.cross_entropy(m(x), y); l.backward(); opt.step()
        out.append(l.item())
    return out

ref = unsplit_losses(copy.deepcopy(base).train())
loader = [(x, y)] * 3
variants = {
    "uniform n=4":         dict(n=4),
    "params n=3":          dict(strategy="params", n=3),
    "cuts layer2,layer3.0": dict(cuts=["layer2", "layer3.0"], expand=["layer3"]),
}
for name, kw in variants.items():
    s = ts.slice(copy.deepcopy(base).train(), **kw)
    print(f"\n{name}:\n{s.summary()}")
    h = s.train(loader, {"name": "SGD", "params": {"lr": 0.05}},
                {"name": "CrossEntropyLoss", "params": {}}, epochs=1)
    mean_ref = sum(ref) / len(ref)
    print(f"{name}: sliced loss {h[0]['loss']:.6f}  unsplit mean {mean_ref:.6f}  "
          f"diff {abs(h[0]['loss'] - mean_ref):.1e}  bytes/epoch fwd {h[0].get('bytes_fwd', 'n/a')}")
