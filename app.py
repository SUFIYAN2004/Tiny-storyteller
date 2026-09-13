import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tokenizers import Tokenizer

st.set_page_config(page_title="Tiny Storyteller", page_icon="📖")

# ── load config, tokenizer, weights (cached so it only runs once) ──
@st.cache_resource
def load_model():
    with open("config.json") as f:
        cfg = json.load(f)
    vocab_size, dim, n_layers = cfg["vocab_size"], cfg["dim"], cfg["n_layers"]
    n_q_heads, n_kv_heads, head_dim = cfg["n_q_heads"], cfg["n_kv_heads"], cfg["head_dim"]
    ff_hidden, seq = cfg["ff_hidden"], cfg["seq"]

    tok = Tokenizer.from_file("tokenizer.json")
    eos_id = tok.token_to_id("<eos>")

    weights = np.load("weights_fp16.npz")
    w_list = [weights[n].astype(np.float32) for n in weights.files]  # upcast fp16 -> fp32 for inference

    def build_rope_cache(t, dim, base=10000.0):
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        positions = np.arange(t).astype(np.float32)
        freqs = np.outer(positions, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=-1)
        return tf.constant(np.cos(emb), tf.float32), tf.constant(np.sin(emb), tf.float32)

    def rotate_half(x):
        d = x.shape[-1]
        x1, x2 = x[..., :d // 2], x[..., d // 2:]
        return tf.concat([-x2, x1], axis=-1)

    def apply_rope(x, cos, sin):
        cos = cos[tf.newaxis, tf.newaxis, :, :]
        sin = sin[tf.newaxis, tf.newaxis, :, :]
        return x * cos + rotate_half(x) * sin

    class RMSNorm(tf.keras.layers.Layer):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.scale = self.add_weight(shape=(dim,), initializer="ones", trainable=True)

        def call(self, x):
            var = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
            return x * tf.math.rsqrt(var + self.eps) * self.scale

    tok_emb = tf.keras.layers.Embedding(vocab_size, dim)

    class DecoderLayer(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
            self.wq = tf.keras.layers.Dense(n_q_heads * head_dim, use_bias=False)
            self.wk = tf.keras.layers.Dense(n_kv_heads * head_dim, use_bias=False)
            self.wv = tf.keras.layers.Dense(n_kv_heads * head_dim, use_bias=False)
            self.wo = tf.keras.layers.Dense(dim, use_bias=False)
            self.ff1 = tf.keras.layers.Dense(ff_hidden, activation="gelu")
            self.ff2 = tf.keras.layers.Dense(dim)

        def call(self, h, cos, sin, mask):
            b = tf.shape(h)[0]
            t = tf.shape(h)[1]
            hn = self.norm1(h)
            q = tf.transpose(tf.reshape(self.wq(hn), [b, t, n_q_heads, head_dim]), [0, 2, 1, 3])
            k = tf.transpose(tf.reshape(self.wk(hn), [b, t, n_kv_heads, head_dim]), [0, 2, 1, 3])
            v = tf.transpose(tf.reshape(self.wv(hn), [b, t, n_kv_heads, head_dim]), [0, 2, 1, 3])
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            rep = n_q_heads // n_kv_heads
            k = tf.repeat(k, rep, axis=1)
            v = tf.repeat(v, rep, axis=1)
            scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(head_dim, tf.float32))
            scores += mask
            w_ = tf.nn.softmax(scores, axis=-1)
            attn_out = tf.matmul(w_, v)
            attn_out = tf.reshape(tf.transpose(attn_out, [0, 2, 1, 3]), [b, t, n_q_heads * head_dim])
            attn_out = self.wo(attn_out)
            h = h + attn_out
            hn2 = self.norm2(h)
            ff_out = self.ff2(self.ff1(hn2))
            return h + ff_out

    decoder_layers = [DecoderLayer() for _ in range(n_layers)]
    final_norm = RMSNorm(dim)

    def causal_mask(t):
        m = 1 - tf.linalg.band_part(tf.ones((t, t)), -1, 0)
        return m * -1e9

    cos_cache, sin_cache = build_rope_cache(seq, head_dim)

    def forward(x):
        t = tf.shape(x)[1]
        h = tok_emb(x)
        mask = causal_mask(t)
        cos = cos_cache[:t]
        sin = sin_cache[:t]
        for layer in decoder_layers:
            h = layer(h, cos, sin, mask)
        h = final_norm(h)
        return tf.matmul(h, tok_emb.embeddings, transpose_b=True)

    def all_vars():
        v = list(tok_emb.trainable_variables) + list(final_norm.trainable_variables)
        for layer in decoder_layers:
            v += layer.trainable_variables
        return v

    _ = forward(tf.zeros([1, seq], dtype=tf.int32))
    vars_ = all_vars()
    for v, w in zip(vars_, w_list):
        v.assign(w)

    return forward, tok, eos_id, seq, vocab_size


forward, tok, eos_id, seq, vocab_size = load_model()


def generate_stream(seed, max_len, temp, top_k=40, rep_penalty=1.3, rep_window=20):
    """Yields one decoded token/piece at a time for streaming display."""
    ids = tok.encode(seed).ids
    ctx = tf.constant([ids[-seq:]], dtype=tf.int32)
    recent = []
    for _ in range(max_len):
        logit = forward(ctx)[0, -1].numpy().astype(np.float64) / max(temp, 1e-4)
        for tid in set(recent[-rep_window:]):
            logit[tid] /= rep_penalty
        k = min(top_k, vocab_size)
        top_idx = np.argpartition(logit, -k)[-k:]
        top_vals = logit[top_idx]
        probs = np.exp(top_vals - top_vals.max())
        probs /= probs.sum()
        next_id = int(np.random.choice(top_idx, p=probs))
        if next_id == eos_id:
            break
        recent.append(next_id)
        ctx = tf.concat([ctx[:, 1:], [[next_id]]], axis=1) if ctx.shape[1] >= seq else \
              tf.concat([ctx, [[next_id]]], axis=1)
        yield tok.decode([next_id]) + " "


# ── sidebar controls ────────────────────────────────────────────
st.sidebar.title("📖 Tiny Storyteller")
st.sidebar.caption("~27M params · RoPE + GQA + RMSNorm · trained with Active Learning + MAML")
temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.5, value=0.8, step=0.05)
max_len = st.sidebar.slider("Max length (tokens)", min_value=20, max_value=300, value=150, step=10)
streaming = True  # streaming generation is always on

st.title("Tiny Storyteller")
st.caption("A small transformer trained on TinyStories. Give it a story opener and see what it writes.")

# ── chat state ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Start a story... e.g. 'Once upon a time, there was a little fox'")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if streaming:
            full_response = st.write_stream(
                generate_stream(prompt, max_len=max_len, temp=temperature)
            )
        else:
            full_response = "".join(generate_stream(prompt, max_len=max_len, temp=temperature))
            st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
