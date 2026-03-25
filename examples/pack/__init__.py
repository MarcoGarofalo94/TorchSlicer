# Pack function examples for common model families.
#
# A pack function has signature:
#   pack(model: nn.Module) -> list[nn.Module]
#
# and is passed to ts.slice():
#   sliced = ts.slice(model, n=4, pack=pack_qwen)
#
# IMPORTANT — serialization constraint
# ------------------------------------
# Slices are shipped to workers via torch.save/load.  Every class
# instantiated inside a pack function must be importable on workers
# under the same module path.  There are three ways to satisfy this:
#
#   1. Use only torchslicer library classes (BlockStage, GPT2EmbedStage,
#      CausalLMHeadStage, MoEBlockStage, SimpleEmbedStage, AuxInputStage,
#      nn.Sequential, etc.) — these are always available.
#
#   2. Install your custom stage classes as a package on all workers:
#        pip install myproject   # on every worker
#      and import from there in your pack function.
#
#   3. For Docker-based setups, set PYTHONPATH on workers to expose
#      any local module:
#        environment:
#          - PYTHONPATH=/workspace
#
# The examples here follow rule 1 — no custom classes, only library
# primitives — so they work with any worker configuration.
#
# Available pack functions:
#   from examples.pack.gpt2         import pack_gpt2
#   from examples.pack.llama        import pack_llama
#   from examples.pack.bert         import pack_bert_seq_cls, pack_bert_mlm
#   from examples.pack.deepseek_moe import pack_moe
