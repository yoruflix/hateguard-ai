
"""hateguard_core.py — kode bersama Training, App, dan smoke test (SHA-256 dicatat di manifest).
Jangan diedit manual: perubahan apa pun mengubah SHA-256 dan akan ditolak oleh App (Hard Fail)."""
import re
import numpy as np
import pandas as pd

CORE_VERSION = "1.0.0"
POLICY_VERSION = "tier-4-softmax-0.50-0.75-0.90/v1"

RE_HEX_ESCAPE = re.compile(r"(\\x[0-9a-fA-F]{2})+")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MENTION = re.compile(r"@\w+")
RE_PLACEHOLDER = re.compile(r"\b(rt|user|url)\b", flags=re.IGNORECASE)
RE_MULTISPACE = re.compile(r"\s+")

def clean_text(text, kamus):
    """Pembersihan + normalisasi 4 bahasa (IDENTIK dengan revisi beku; menentukan preprocessing_parity_hash):
    1) case folding; 2) buang hex-escape, URL, mention, placeholder crawler (rt/user/url);
    3) normalisasi token per-kata memakai kamus (tanda baca tepi dipertahankan)."""
    if text is None or pd.isna(text):
        return ""
    text = str(text).lower()
    text = RE_HEX_ESCAPE.sub(" ", text)
    text = RE_URL.sub(" ", text)
    text = RE_MENTION.sub(" ", text)
    text = RE_PLACEHOLDER.sub(" ", text)
    normalized_tokens = []
    for raw_token in text.split():
        prefix_punct = ""
        suffix_punct = ""
        core_word = raw_token
        while len(core_word) > 1 and not core_word[0].isalnum():
            prefix_punct += core_word[0]
            core_word = core_word[1:]
        while len(core_word) > 1 and not core_word[-1].isalnum():
            suffix_punct = core_word[-1] + suffix_punct
            core_word = core_word[:-1]
        mapped_word = kamus.get(core_word, core_word)
        normalized_tokens.append(f"{prefix_punct}{mapped_word}{suffix_punct}")
    return RE_MULTISPACE.sub(" ", " ".join(normalized_tokens)).strip()

_CENSOR_CACHE = {}
def _mask_match(match):
    w = match.group(0)
    if len(w) <= 2:
        return "*" * len(w)
    return w[0] + ("*" * (len(w) - 2)) + w[-1]

def censor_sensitive_words(text, sensitive_words):
    """Sensor leksikal (huruf pertama & terakhir dipertahankan). Output tampilan, BUKAN input model."""
    out = text
    for word in sensitive_words:
        pat = _CENSOR_CACHE.get(word)
        if pat is None:
            pat = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
            _CENSOR_CACHE[word] = pat
        out = pat.sub(_mask_match, out)
    return out

def prob_to_level(p_hs, thresholds):
    """Ambang OPERASIONAL berbasis softmax (bukan ukuran keparahan linguistik terkalibrasi)."""
    if p_hs < thresholds["safe_max"]:
        return "safe"
    if p_hs < thresholds["weak_max"]:
        return "weak"
    if p_hs < thresholds["moderate_max"]:
        return "moderate"
    return "strong"

def predicted_label_from_p(p_hs, thresholds):
    return "HATE_SPEECH" if p_hs >= thresholds["safe_max"] else "NON_HATE_SPEECH"

def policy_decision(level, raw_text, masked_text):
    """Teks kebijakan guardrail — SATU definisi untuk Training dan App (tanpa efek samping/antrean)."""
    if level == "safe":
        return {"action": "tayang_normal",
                "status_label": "✅ NON-HATE SPEECH — DITAMPILKAN NORMAL (ACTION LEVEL: SAFE)",
                "public_output": raw_text}
    if level == "weak":
        return {"action": "flag_censor_ringan",
                "status_label": "⚠️ HATE SPEECH INDIKASI RENDAH — DISENSOR LEKSIKAL (ACTION LEVEL: WEAK)",
                "public_output": f"[⚠️ Disensor Leksikal] {masked_text}"}
    if level == "moderate":
        return {"action": "flag_spoiler_log",
                "status_label": "🟠 HATE SPEECH INDIKASI MENENGAH — SPOILER & DISENSOR (ACTION LEVEL: MODERATE)",
                "public_output": f"[🟠 Peringatan Konten Sensitif]: {masked_text}"}
    return {"action": "blokir_eskalasi",
            "status_label": "⛔ HATE SPEECH INDIKASI TINGGI — DIBLOKIR & MASUK ANTREAN MODERATOR (ACTION LEVEL: STRONG)",
            "public_output": "[⛔ KOMENTAR DIBLOKIR OTOMATIS OLEH HATEGUARD — MENUNGGU KEPUTUSAN MODERATOR MANUSIA (PENDING REVIEW)]"}

def logits_fp32(texts, tokenizer, model, device, max_length, padding, batch_size=64):
    """Jalur inferensi TUNGGAL (FP32, tanpa autocast, no_grad). Dipakai Training, App, dan smoke test."""
    import torch
    model.eval()
    outs = []
    texts = list(texts)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], return_tensors="pt", truncation=True,
                            padding=padding, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            outs.append(model(**enc).logits.to(torch.float32).cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 2), dtype=np.float32)

def softmax_hs(logits):
    """P(HATE_SPEECH) dari logit [N,2], stabil secara numerik (float64)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)
