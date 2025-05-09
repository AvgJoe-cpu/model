VectorGate

Per-dimension exponential gating for residual connections—drop-in for any Transformer (or MLP) layer.

 

Features
	•	Vector-valued curvature c ∈ ℝᵈ for per-feature gating sharpness
	•	Learnable parameters for gating function and curvature
	•	Tiny parameter footprint (just O(d² + d) per gate)
	•	Interpretable outputs α(x) = exp(s(x)), β(x) = exp(-s(x)), s(x) = c ⊙ tanh(g(x))
	•	Plug-and-play with 🤗 Transformers, ViT, or your own residual blocks

Installation

pip install vectorgate
# or from GitHub
git clone https://github.com/yourorg/vectorgate.git
cd vectorgate
pip install -e .

Quickstart

import torch
from vectorgate import VectorGate, gated_residual

# x: (B, L, D), f_x = your MLP or attention output
gate = VectorGate(dim=D)
alpha, beta, s = gate(x)
y = gated_residual(x, f_x, alpha, beta)

BERT Example

from transformers import BertForSequenceClassification
from vectorgate import GatedBertOutput

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
# Replace layer 0's output block with gated version
gated = GatedBertOutput(model.config)
model.bert.encoder.layer[0].output = gated

See examples/bert_example.py for a full script.

Background & Related Work

The core idea of using an exponential gating mechanism traces back to the XLSTM (eXtra Long Short-Term Memory) architecture, which implemented symmetric exponential gates to control information flow in recurrent networks. We build upon this by:
	1.	Vector-valued curvature: instead of a shared scalar range bound, we introduce a learnable per-dimension c ∈ ℝᵈ to allow each feature channel its own gating sharpness.
	2.	Transformer integration: we drop the gate into residual connections of self-attention and feed-forward blocks, enabling fine-grained control with minimal code changes.
	3.	Parameter efficiency: only O(d² + d) new parameters per gated layer, compared to large adapter modules or low-rank updates.

Other influential works include:
	•	Highway Networks (Srivastava et al., 2015): introduced trainable gates in feed-forward nets, but used sigmoidal gates and shared range bounds.
	•	Gated Linear Units (GLU) (Dauphin et al., 2017): apply a sigmoid-based gate to half of the features in convolutional/MLP blocks.
	•	Mixture-of-Experts routing: dynamic gating across expert sub-networks, though often at a much larger scale and complexity.
	•	LoRA & Adapters: parameter-efficient tuning approaches that adapt weights or add bottleneck layers, but without interpretable gating dynamics.

By combining symmetric exponential gating with per-dimension curvature in Transformer residuals, VectorGate delivers an interpretable, general-purpose building block for modern architectures.

Development

git clone https://github.com/yourorg/vectorgate.git
cd vectorgate
pip install -e .
pytest
