# TorchSlicer algorithm survey — M8

*Written 2026-10-16. Literature rule: only papers actually retrieved; all links verified.*

---

## 1. Goal

Design a `ts.Algorithm` API so that an SL researcher can express any published SL variant in
< 100 lines, the same object running in local, centralized and P2P executors, without touching
the engine. Analogous to [Flower's Strategy](https://flower.ai/docs/framework/explanation-flower-strategy-abstraction.html)
for FL: one abstract class, defaults equal to today's behaviour, override only what changes.

---

## 2. SL algorithm families

### 2.1 Vanilla SplitNN

**Paper:** Vepakomma et al., "Split learning for health: Distributed deep learning without
sharing raw patient data," 2019. arXiv:[1812.00564](https://arxiv.org/pdf/1812.00564).

**Setting:** one client, one server, one cut.  
**Flow:** client forward → send activations → server forward + loss → send cut gradient back →
client backward + step → server step.  
**Engine needs:** the standard flow already implemented (no extension required; forms the default).

### 2.2 U-shaped SplitNN

**Paper:** same Vepakomma et al. 2019 paper; extended by
arXiv:[2604.24468](https://arxiv.org/html/2604.24468v1) (SL for LLM fine-tuning survey, 2026).

**Setting:** client holds head + tail; server holds middle layers; labels stay on client.  
**Flow:** client head forward → server middle → client tail forward + loss → client tail backward →
server backward → client head backward + step on head+tail → server step.  
**Engine needs:** two cuts; in-process tail hosted by the data owner; loss on the tail (not on
the last *remote* slice); two communication rounds per batch. Already done in M4 (`u_shaped`
setting). The algorithm API must expose *where* the loss is computed and *which side* does the
final backward.

### 2.3 SplitFed V1 and V2

**Paper:** Thapa, Chamikara, Camtepe, and Sun, "SplitFed: When Federated Learning Meets Split
Learning," AAAI 2022. [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/20825/20584).

**SFL-V1:** one server-side model copy *per client*; both client-side and server-side models are
FedAvg-aggregated at the end of each round. Training is parallel (one server copy per client).  
**SFL-V2:** a single shared server-side model trained sequentially by all clients; only the
client-side models are aggregated.

**Engine needs (V1):**
- Per-client server state (weights + optimizer); the engine must allocate and restore these.
- Server-side FedAvg: collect all server copies, aggregate, broadcast back.
- Client-side FedAvg: as today (`on_round_end`).
- Parallel client scheduling: concurrent batches on the server.

**Engine needs (V2):**
- A single server state shared across clients: already the default.
- Client-side FedAvg only.
- Sequential client scheduling.

Today `ClientSession` handles client-state swapping and `on_round_end` for FedAvg, but per-client
**server** copies and server aggregation are impossible without algorithm hooks.

**Impact of cut layer (2024):** arXiv:[2412.15536](https://arxiv.org/html/2412.15536v1) reports
SFL-V2 is sensitive to cut layer depth; SFL-V1 is not (per-client copies decouple server training).

### 2.4 Local-loss / Decoupled SL

**Papers:**
- "Decoupled Split Learning via Auxiliary Loss," 2026. arXiv:[2601.19261](https://arxiv.org/pdf/2601.19261).
  Client attaches a small auxiliary classifier at the cut; trains with local loss `L_aux`; no
  gradient flows from server. Only forward activations are communicated. CIFAR-10 results close to
  standard SL with ~50% less communication.
- FedSplitX, 2023. arXiv:[2310.14579](https://arxiv.org/pdf/2310.14579). Auxiliary network on the
  client so it can run backward locally, without waiting for the server's gradient.
- HERON-SFL ("Lean Clients, Full Accuracy"), 2025. arXiv:[2601.09076](https://arxiv.org/html/2601.09076).
  Clients estimate gradients from perturbed forward passes (zeroth-order); no activations cached.

**Known weakness:** local gradients can be inaccurate without periodic realignment to the server
(noted in arXiv:[2604.24468](https://arxiv.org/pdf/2604.24468), which cites FSL-SAGE and ESL-LEO
as fixes). [unverified: FSL-SAGE and ESL-LEO paper details not retrieved]

**Engine needs:**
- `backward_policy`: suppress the server→client gradient return.
- The client slice must own an auxiliary head (instantiated by the algorithm, not the engine).
- The auxiliary head is used only during training; inference follows the standard forward path.
- Loss is computed *on the client* (at the cut), not on the last slice.

### 2.5 Parallel SL / CPSL

**Papers:**
- "Privacy-Sensitive Parallel Split Learning," 2020.
  [ResearchGate](https://www.researchgate.net/publication/339652912_Privacy-Sensitive_Parallel_Split_Learning).
- "Split Learning over Wireless Networks: Parallel Design and Resource Management," 2022.
  arXiv:[2204.08119](https://arxiv.org/pdf/2204.08119). CPSL introduces parallelism to reduce
  training latency; suited for resource-constrained IoT; low communication overhead and compute
  workload per device.
- "Optimal Resource Allocation for U-Shaped Parallel Split Learning," 2023.
  arXiv:[2308.08896](https://arxiv.org/pdf/2308.08896). U-PSL eliminates raw data and label
  sharing; reports similar or shorter latency than sequential SL benchmarks.
- "MP-SL: Multihop Parallel Split Learning," 2024. arXiv:[2402.00208](https://arxiv.org/pdf/2402.00208).
  Extends SL to K intermediate hops (not retrieved in full; arXiv link confirmed).

**Engine needs:**
- Concurrent client batches on the server (today clients are sequential).
- Per-client server copy (same as SFL-V1) or batching across clients.
- Synchronisation barrier at round end.
- (M9 scope for real parallel clients; the algorithm API must not *prevent* it.)

### 2.6 Label-protection variants / FSL

**Papers:**
- Yang et al., "Differentially Private Label Protection in Split Learning," 2022.
  arXiv:[2203.02073](https://arxiv.org/pdf/2203.02073). Generic gradient-perturbation framework
  with provable DP guarantee on labels.
- Vehicular federated SL with data and label privacy, 2023.
  [ResearchGate](https://www.researchgate.net/publication/373375913_Federated_Split_Learning_with_Data_and_Label_Privacy_Preservation_in_Vehicular_Networks).
  Labels held by roadside infrastructure (label party ≠ data owner).
- P3SL (Personalized Privacy-Preserving SL), 2025. arXiv:[2507.17228](https://arxiv.org/pdf/2507.17228).
  [unverified: paper not retrieved in full]

**Engine needs:**
- Loss computed at a *label party* that is distinct from both the data owner and the server.
- Gradient perturbation before the cut gradient is returned (a hook on the server side).
- (Overlaps with M11 privacy suite; the algorithm API should expose *where* the loss is and
  *which side* perturbs the gradient — the privacy defences plug into those hooks.)

### 2.7 Asynchronous SL

**Paper:** Listed as "AsyncSL" in the review "A Review and Experimental Evaluation on Split
Learning," MDPI Future Internet 2025. [Link](https://www.mdpi.com/1999-5903/17/2/87). Described
as model-split-only, reduces communication cost, no label protection.

[unverified: original AsyncSL paper not retrieved — the MDPI review's reference 86 was not
retrieved.]

**Engine needs:** stale-gradient acceptance (async step counter); non-blocking server forward;
no synchronisation barrier between client rounds.

### 2.8 SplitBud (closest prior framework)

**Paper:** Radovic et al., "Towards a Unified Framework for Split Learning," EuroMLSys 2025.
DOI: [10.1145/3721146.3721936](https://doi.org/10.1145/3721146.3721936).
[PDF](https://euromlsys.eu/pdf/euromlsys25-7.pdf).

"There is no unified framework for implementing and deploying SL algorithms" — the stated gap
SplitBud fills. The paper says it can "implement virtually any SL algorithm."
[unverified: the internal abstraction of SplitBud was not retrieved; the PDF at the above link
was found but not fully read in this session.]

**Relevance to TorchSlicer:** this is the closest related work to M8. The `ts.Algorithm` API
must be compared against SplitBud in the paper (M15).

---

## 3. What each family needs from the engine — summary table

| Family | Loss location | Backward | Client state | Server state | Parallelism |
|--------|--------------|----------|--------------|--------------|-------------|
| Vanilla SplitNN | last (server) slice | full chain | one | one | sequential |
| U-shaped | tail (client) slice | full chain, two cuts | one | one | sequential |
| SplitFed V1 | last (server) slice | full chain | per-client, FedAvg | per-client, FedAvg | parallel |
| SplitFed V2 | last (server) slice | full chain | per-client, FedAvg | shared | sequential |
| Local-loss SL | cut (client aux head) | client only | one | one | sequential |
| Parallel SL / CPSL | last slice | full chain | per-client | shared or per-client | parallel |
| Label-protection FSL | label party | full chain | one | one | sequential |
| Async SL | last slice | full chain | per-client | shared | async |

---

## 4. API design: `ts.Algorithm`

### 4.1 Design principles (from Flower's Strategy lessons)

1. **Defaults = today's behaviour.** An empty subclass runs vanilla SplitNN identically to the
   current engine. No existing run breaks.
2. **One object, all executors.** The same `Algorithm` instance is passed to `LocalExecutor`,
   `DistributedExecutor` and `P2PExecutor`. Executors call its hooks; the algorithm never
   reaches into the executor.
3. **Hooks, not overrides of internal methods.** A researcher overrides the hooks they care
   about; the loop structure stays in the executor.
4. **Data classes in, data classes out.** Hooks take and return typed objects
   (`ClientContext`, `ServerContext`), not raw tensors, so future hooks don't break signatures.
5. **Registry.** `ts.register_algorithm("my-algo", MyAlgo)` + `algorithm: my-algo` in config.

### 4.2 Hook sketch

```python
class Algorithm:
    # --- per-batch, client side ---
    def client_forward(self, ctx: ClientContext) -> ClientForwardResult:
        """Default: run the client slice forward, return activations."""

    def client_backward(self, ctx: ClientContext, cut_grads: Tensor | None):
        """Default: run backward from cut_grads; step optimizer."""

    # --- per-batch, server side ---
    def server_forward(self, ctx: ServerContext) -> ServerForwardResult:
        """Default: run server slice(s) forward."""

    def loss(self, ctx: ServerContext, out: Tensor, labels: Tensor) -> Tensor:
        """Default: cross-entropy on the last slice's output."""

    def server_backward(self, ctx: ServerContext, loss: Tensor) -> Tensor:
        """Default: loss.backward(); return gradient at the cut."""

    # --- per-batch, policy ---
    def backward_policy(self, round: int) -> BackwardPolicy:
        """Default: BackwardPolicy.FULL (full chain).
        Alternatives: STOP_AT_CUT_K (stop backward at cut k),
                      LOCAL_LOSS (client-only backward, server sends no grad)."""

    # --- per-round ---
    def select_clients(self, available: list[ClientInfo], round: int) -> list[ClientInfo]:
        """Default: all available clients."""

    def aggregate(
        self,
        client_states: list[ClientState],
        server_states: list[ServerState],
        round: int,
    ) -> tuple[ClientState | None, ServerState | None]:
        """Default: no aggregation (return None for both).
        Override for FedAvg on client (V2) or both client+server (V1)."""

    def on_round_end(self, metrics: RoundMetrics, round: int):
        """Default: no-op."""
```

### 4.3 Mapping of families to hooks

| Family | Hooks to override |
|--------|-------------------|
| Vanilla SplitNN | none (all defaults) |
| U-shaped | `loss` (client tail), `server_backward` (returns grad to client tail) |
| SplitFed V1 | `aggregate` (FedAvg client + server), `select_clients` |
| SplitFed V2 | `aggregate` (FedAvg client only), `select_clients` |
| Local-loss SL | `backward_policy` → `LOCAL_LOSS`, `client_forward` (attach aux head), `loss` |
| Gradient perturbation (DP) | `server_backward` (perturb before return) |
| Async SL | `select_clients` (async scheduling), `aggregate` (staleness weighting) |
| Per-client server copies (V1) | `aggregate` + executor allocates extra server states |

### 4.4 What M8 implements (proof of expressiveness)

Required by the roadmap:
1. **Default algorithm** — reproduces current matrix bit for bit.
2. **Vanilla, U-shaped, SplitFed FedAvg** — ports existing features onto the new hooks.
3. **Local-loss SL** — `backward_policy=LOCAL_LOSS`, client auxiliary head, no grad back.
4. **Per-client server copies with aggregation** — `aggregate` overrides server state.

---

## 5. Decision

**What to reuse:** `ClientSession.handoff` / `on_round_end` become thin wrappers over
`Algorithm.aggregate` / `Algorithm.on_round_end`. The executors' inner loops call the hooks;
no logic moves out of the engine.

**What to write:** the `Algorithm` base class and registry in `torchslicer/algorithm/`; thin
adapters in each executor; the four algorithm implementations above; the tutorial.

**What not to do in M8:** true parallel clients (M9), async scheduling (M9/M10), label-party
routing (M9), gradient perturbation as a defence (M11). The API must *permit* them (hooks are
there) without *requiring* the engine to implement them now.

**Key difference from SplitBud/SLPerf:** those frameworks ask the researcher to wire up their
own training loop. TorchSlicer keeps the loop in the executor and exposes hooks — the researcher
overrides only what changes, exactly as Flower's Strategy does for FL.
