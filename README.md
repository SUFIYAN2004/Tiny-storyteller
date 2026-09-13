# Tiny Storyteller

A ~27M-parameter decoder-only transformer built from scratch (RoPE, Grouped-Query Attention, RMSNorm, tied embeddings), trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) with **Active Learning** (entropy-based sample selection) and **MAML** (first-order meta-learning) layered into the training loop.

Live demo: a Streamlit chat app — give it a story opener, it streams back a short children's-style story.

## Architecture

- 9 decoder layers, embedding dim 576, 9 query heads / 3 KV heads (grouped-query attention)
- Rotary positional embeddings, RMSNorm pre-normalization, GELU feed-forward (576 → 1728 → 576)
- Custom 2,700-token byte-level BPE tokenizer trained on TinyStories
- 192-token context window
- Tied input/output embedding weights

## Training

- Single NVIDIA T4 GPU, ~3h40m, 37 epochs (early-stopped)
- Held-out validation perplexity: **8.2**
- Active learning: every training step, the 8 highest-entropy examples from a 64-example pool are added to the batch
- MAML: once per epoch, a first-order meta-gradient update runs across two support/query task pairs

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit chat app (chat_input/chat_message, streaming generation, temperature + max-length sliders) |
| `weights_fp16.npz` | Trained model weights, float16 |
| `tokenizer.json` | Custom BPE tokenizer |
| `config.json` | Model architecture config |
| `requirements.txt` | Python dependencies |

## Notes

This is a small research/learning project, not a production model. It reliably produces grammatically coherent short stories with consistent character tracking, and shows partial (inconsistent) short-range causal reasoning. It has no knowledge outside the TinyStories domain.
