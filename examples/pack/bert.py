"""
Pack functions for BERT family models.

Applies to: BertForSequenceClassification, BertForMaskedLM, DistilBERT,
            and any model with a ``model.bert.encoder.layer`` block list.

Usage
-----
Sequence classification::

    from transformers import BertForSequenceClassification
    import torchslicer as ts
    from examples.pack.bert import pack_bert_seq_cls

    model  = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    sliced = ts.slice(model, n=4, pack=pack_bert_seq_cls)
    sliced.train(loader, optimizer, criterion, epochs=3)

Masked LM::

    from transformers import BertForMaskedLM
    from examples.pack.bert import pack_bert_mlm

    model  = BertForMaskedLM.from_pretrained("bert-base-uncased")
    sliced = ts.slice(model, n=4, pack=pack_bert_mlm)

Notes
-----
- All stage classes used here are either from ``torchslicer`` or standard
  PyTorch (``nn.Sequential``) so workers can always deserialize slices
  without any extra imports.
- ``BertEmbeddings`` handles token + position + token_type embeddings
  internally, so ``SimpleEmbedStage`` just delegates to it.
- Phase 1: no padding mask — BERT encoder blocks compute full attention.
  Suitable for fixed-length sequences.
"""

import torch.nn as nn
import torchslicer as ts


def pack_bert_seq_cls(model: nn.Module) -> list[nn.Module]:
    """Return a flat stage list for BertForSequenceClassification.

    The classifier head (pooler → dropout → classifier) is wrapped as a single
    ``BlockStage`` around an ``nn.Sequential``, avoiding any custom class.
    """
    bert   = model.bert
    embed  = ts.SimpleEmbedStage(bert.embeddings)
    blocks = [ts.BlockStage(b) for b in bert.encoder.layer]
    head   = ts.BlockStage(nn.Sequential(bert.pooler, model.dropout, model.classifier))
    return [embed] + blocks + [head]


def pack_bert_mlm(model: nn.Module) -> list[nn.Module]:
    """Return a flat stage list for BertForMaskedLM."""
    bert   = model.bert
    embed  = ts.SimpleEmbedStage(bert.embeddings)
    blocks = [ts.BlockStage(b) for b in bert.encoder.layer]
    head   = ts.BlockStage(model.cls)
    return [embed] + blocks + [head]
