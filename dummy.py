"""
Flask Web App — Network Traffic Analyser
=========================================
Matches your model_artifacts folder exactly:
  - best_model.keras
  - scaler.pkl
  - label_encoder.pkl      (single encoder, not cat/thr split)
  - metadata.json

Features:
  - Upload PCAP → category prediction + confidence
  - Automatic threat detection from traffic patterns
  - Plain-English explanation for each prediction
  - Top-3 probable categories shown
  - Protocol analysis (TCP vs QUIC breakdown)

Requirements:
    pip install flask tensorflow scikit-learn pandas numpy scapy

Run:
    python app.py
    Open: http://localhost:5000
"""

import os
import json
import pickle
import tempfile
import numpy as np
from collections import deque, defaultdict
from flask import Flask, request, jsonify, render_template_string

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_DIR      = os.environ.get("MODEL_DIR", "model_artifacts")
WINDOW_SIZE    = 100
N_FEATURES     = 12
QUIC_PORTS     = {443, 80}
LOCAL_PREFIXES = ("192.168.", "10.", "172.16.", "127.", "::1", "fe80:")

# ── Threat rules (applied on top of ML prediction) ───────────────────────────

THREAT_RULES = [
    {
        "name":        "C2 Beacon",
        "severity":    "HIGH",
        "color":       "#dc2626",
        "condition":   lambda s: s["mean_iat"] < 2.0 and s["std_len"] < 30 and s["mean_len"] < 200,
        "explanation": "Very regular packet timing with tiny payloads — matches automated command-and-control beacon behaviour.",
        "protocol":    "Any",
    },
    {
        "name":        "QUIC Evasion",
        "severity":    "MEDIUM",
        "color":       "#d97706",
        "condition":   lambda s: s["proto_mode"] == 1 and s["burst_max"] > 20 and 100 < s["mean_len"] < 600,
        "explanation": "QUIC/UDP traffic with large burst patterns — common technique to hide C2 or tunneling activity inside encrypted QUIC streams.",
        "protocol":    "QUIC (UDP 443)",
    },
    {
        "name":        "Data Exfiltration",
        "severity":    "CRITICAL",
        "color":       "#991b1b",
        "condition":   lambda s: s["total_bytes"] > 500_000 and s["direction_ratio"] < 0.3 and s["mean_len"] > 800,
        "explanation": "Large sustained outbound transfer with big packets — unusually high upload volume relative to download activity.",
        "protocol":    "TCP",
    },
    {
        "name":        "Port Scan",
        "severity":    "MEDIUM",
        "color":       "#d97706",
        "condition":   lambda s: s["mean_len"] < 80 and s["burst_max"] > 15 and s["direction_ratio"] > 0.8,
        "explanation": "Many tiny outbound packets in rapid succession — classic reconnaissance/port scanning pattern.",
        "protocol":    "TCP",
    },
    {
        "name":        "DNS Tunneling",
        "severity":    "HIGH",
        "color":       "#dc2626",
        "condition":   lambda s: s["proto_mode"] == 2 and s["mean_len"] < 120 and s["std_len"] < 25,
        "explanation": "Tiny, very regular UDP packets — may encode data inside DNS queries to bypass firewalls.",
        "protocol":    "UDP",
    },
]

CATEGORY_EXPLANATIONS = {
    "BBC News":   "HTTP/HTTPS article loads — medium packet sizes, moderate burst frequency, mostly download traffic.",
    "Coursera":   "Large video-streaming bursts mixed with smaller API and asset requests.",
    "Discord":    "Frequent small UDP/QUIC packets typical of real-time voice and chat — very bursty.",
    "Disney":     "Sustained high-bitrate video streaming — large TCP flows with adaptive bitrate switching.",
    "Dropbox":    "Mix of large file transfer bursts and small sync heartbeat packets.",
    "Facebook":   "HTTPS API calls, media loads, and WebSocket-style keep-alives.",
    "github":     "HTTPS API calls mixed with large git object transfers.",
    "pinterest":  "Image-heavy HTTPS traffic — many medium-sized download bursts.",
    "quora":      "Standard HTTPS browsing — medium packet sizes, low burst count.",
    "tumble":     "Mixed media content — images and video mixed with standard web requests.",
}

# ── Load model artefacts ──────────────────────────────────────────────────────

def load_artefacts():
    model_path = os.path.join(MODEL_DIR, "best_model.keras")
    enc_path   = os.path.join(MODEL_DIR, "label_encoder.pkl")
    scaler_path= os.path.join(MODEL_DIR, "scaler.pkl")
    meta_path  = os.path.join(MODEL_DIR, "metadata.json")

    missing = [p for p in [model_path, enc_path, scaler_path] if not os.path.exists(p)]
    if missing:
        print(f"[WARN] Missing artefacts: {missing} — running in demo mode")
        return None, None, None, None

    model  = tf.keras.models.load_model(model_path)
    with open(enc_path,    "rb") as f: enc    = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f: meta = json.load(f)

    print(f"[OK] Model loaded — classes: {list(enc.classes_)}")
    print(f"[OK] Test accuracy from training: {meta.get('test_accuracy', 'N/A')}")
    return model, enc, scaler, meta


MODEL, ENC, SCALER, META = load_artefacts()

# ── Feature extraction ────────────────────────────────────────────────────────

def is_local(ip):
    return any(ip.startswith(p) for p in LOCAL_PREFIXES)

def encode_tcp_flags(f):
    return round(
        ((f & 0x02) > 0) * 0.4 +
        ((f & 0x10) > 0) * 0.1 +
        ((f & 0x08) > 0) * 0.2 +
        ((f & 0x01) > 0) * 0.2 +
        ((f & 0x04) > 0) * 0.1, 4)

def extract_records(packets):
    from scapy.all import IP, TCP, UDP
    records = []
    flow_state = defaultdict(lambda: {
        "last_time": None, "burst_count": 0,
        "last_burst_time": None, "cumulative_bytes": 0,
        "recent_lengths": deque(maxlen=10),
    })
    for pkt in packets:
        if not pkt.haslayer(IP): continue
        ip_l = pkt[IP]; t = float(pkt.time)
        sp = dp = 0
        proto_str = "other"
        if pkt.haslayer(TCP):
            proto_str = "tcp"; sp, dp = pkt[TCP].sport, pkt[TCP].dport
        elif pkt.haslayer(UDP):
            proto_str = "udp"; sp, dp = pkt[UDP].sport, pkt[UDP].dport
        fkey = (ip_l.src, ip_l.dst, sp, dp, proto_str) if ip_l.src < ip_l.dst \
               else (ip_l.dst, ip_l.src, dp, sp, proto_str)
        state = flow_state[fkey]

        pkt_len = len(pkt); ip_len = len(ip_l); trans_len = 0
        if pkt.haslayer(TCP):   trans_len = len(pkt[TCP].payload)
        elif pkt.haslayer(UDP): trans_len = len(pkt[UDP].payload)

        direction = 1 if is_local(ip_l.src) else -1
        iat = max(0.0, t - state["last_time"]) if state["last_time"] else 0.0
        state["last_time"] = t

        if pkt.haslayer(TCP):
            proto_enc = 0
        elif pkt.haslayer(UDP):
            proto_enc = 1 if (sp in QUIC_PORTS or dp in QUIC_PORTS) else 2
        else:
            proto_enc = 2

        tcp_flags_enc = win_size = 0
        if pkt.haslayer(TCP):
            tcp_flags_enc = encode_tcp_flags(int(pkt[TCP].flags))
            win_size      = pkt[TCP].window

        if state["last_burst_time"] is None or iat > 0.050:
            state["burst_count"] = 1
        else:
            state["burst_count"] += 1
        state["last_burst_time"] = t

        state["cumulative_bytes"] += pkt_len
        rl = list(state["recent_lengths"]); state["recent_lengths"].append(pkt_len)
        roll_mean = float(np.mean(rl)) if rl else float(pkt_len)
        roll_std  = float(np.std(rl))  if len(rl) > 1 else 0.0

        records.append([
            pkt_len, ip_len, trans_len,
            direction, round(iat, 6),
            proto_enc, tcp_flags_enc, win_size,
            state["burst_count"], state["cumulative_bytes"],
            round(roll_mean, 2), round(roll_std, 2),
        ])
    return records


def pcap_to_windows(path):
    from scapy.all import rdpcap
    pkts    = rdpcap(path)
    records = extract_records(pkts)
    n       = len(records)
    if n < WINDOW_SIZE:
        return None, n

    windows = []
    step = WINDOW_SIZE // 2
    for s in range(0, n - WINDOW_SIZE + 1, step):
        windows.append(records[s: s + WINDOW_SIZE])
    return np.array(windows, dtype=np.float32), n


def compute_traffic_stats(X_raw):
    """Aggregate features across all windows for threat rule evaluation."""
    flat = X_raw.reshape(-1, N_FEATURES)
    proto_vals = flat[:, 5].astype(int)
    return {
        "mean_len":       float(flat[:, 0].mean()),
        "std_len":        float(flat[:, 0].std()),
        "mean_iat":       float(flat[:, 4].mean()),
        "proto_mode":     int(np.bincount(proto_vals).argmax()),
        "burst_max":      float(flat[:, 8].max()),
        "total_bytes":    float(flat[:, 9].max()),
        "direction_ratio":float((flat[:, 3] == 1).mean()),
        "tcp_pct":        float((proto_vals == 0).mean() * 100),
        "quic_pct":       float((proto_vals == 1).mean() * 100),
        "other_pct":      float((proto_vals == 2).mean() * 100),
    }


def detect_threats(stats):
    threats = []
    for rule in THREAT_RULES:
        if rule["condition"](stats):
            threats.append({
                "name":        rule["name"],
                "severity":    rule["severity"],
                "color":       rule["color"],
                "explanation": rule["explanation"],
                "protocol":    rule["protocol"],
            })
    return threats

# ── Prediction pipeline ───────────────────────────────────────────────────────

def predict_pcap(path):
    X_raw, n_packets = pcap_to_windows(path)
    if X_raw is None:
        return {"error": f"Too few packets ({n_packets}) — need at least {WINDOW_SIZE}"}

    stats   = compute_traffic_stats(X_raw)
    threats = detect_threats(stats)

    if MODEL is None:
        # Demo mode
        return {
            "demo_mode":    True,
            "n_packets":    n_packets,
            "n_windows":    len(X_raw),
            "category":     "Discord",
            "confidence":   0.84,
            "top3":         [
                {"label": "Discord",  "prob": 0.84},
                {"label": "quora",    "prob": 0.09},
                {"label": "Facebook", "prob": 0.04},
            ],
            "explanation":  CATEGORY_EXPLANATIONS.get("Discord", ""),
            "threats":      threats,
            "traffic_stats": stats,
        }

    N, W, F = X_raw.shape
    X_norm  = SCALER.transform(X_raw.reshape(-1, F)).reshape(N, W, F)
    probs_all = MODEL.predict(X_norm, verbose=0)

    # Handle both single-output and list output
    if isinstance(probs_all, list):
        probs_all = probs_all[0]

    probs    = probs_all.mean(axis=0)
    top_idx  = int(np.argmax(probs))
    top3_idx = np.argsort(probs)[-3:][::-1]

    category = ENC.classes_[top_idx]

    return {
        "n_packets":    int(n_packets),
        "n_windows":    int(N),
        "category":     category,
        "confidence":   round(float(probs[top_idx]), 4),
        "top3":         [
            {"label": ENC.classes_[i], "prob": round(float(probs[i]), 4)}
            for i in top3_idx
        ],
        "explanation":  CATEGORY_EXPLANATIONS.get(category, f"Traffic classified as {category}."),
        "threats":      threats,
        "traffic_stats": {
            "mean_packet_length": round(stats["mean_len"], 1),
            "mean_inter_arrival": round(stats["mean_iat"] * 1000, 2),
            "tcp_percent":        round(stats["tcp_pct"], 1),
            "quic_percent":       round(stats["quic_pct"], 1),
            "other_percent":      round(stats["other_pct"], 1),
            "total_bytes":        int(stats["total_bytes"]),
            "burst_max":          int(stats["burst_max"]),
        },
    }

# ── HTML UI ───────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Network Traffic Analyser</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#f0f2f5;color:#1a1a2e;min-height:100vh;padding:2rem 1rem}
.container{max-width:760px;margin:0 auto}
h1{font-size:1.6rem;font-weight:700;margin-bottom:.2rem;color:#1a1a2e}
.subtitle{color:#64748b;font-size:.9rem;margin-bottom:2rem}

.card{background:#fff;border-radius:14px;border:1px solid #e2e8f0;padding:1.5rem;margin-bottom:1.25rem;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.card-title{font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#94a3b8;margin-bottom:1rem}

/* Upload */
.drop-zone{border:2px dashed #cbd5e1;border-radius:10px;padding:2.5rem;text-align:center;cursor:pointer;transition:all .2s}
.drop-zone:hover,.drop-zone.drag{border-color:#6366f1;background:#f5f3ff}
.drop-icon{font-size:2.5rem;margin-bottom:.5rem}
.drop-label{color:#475569;font-size:.95rem}
.drop-hint{color:#94a3b8;font-size:.8rem;margin-top:.3rem}
.filename{margin-top:.6rem;color:#6366f1;font-weight:500;font-size:.9rem}
input[type=file]{display:none}
.btn{display:inline-flex;align-items:center;gap:.4rem;padding:.65rem 1.6rem;background:#6366f1;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:.9rem;font-weight:500;margin-top:1rem;transition:background .2s}
.btn:hover{background:#4f46e5}
.btn:disabled{background:#a5b4fc;cursor:not-allowed}

/* Confidence bar */
.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem}
.metric-box{background:#f8fafc;border-radius:10px;padding:1rem}
.metric-label{font-size:.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.25rem}
.metric-value{font-size:1.4rem;font-weight:700;color:#1a1a2e}
.metric-sub{font-size:.78rem;color:#94a3b8;margin-top:.2rem}
.bar-track{background:#e2e8f0;border-radius:4px;height:6px;margin-top:.5rem}
.bar-fill{height:6px;border-radius:4px;background:#6366f1;transition:width .5s ease}

/* Top 3 */
.top3-row{display:flex;align-items:center;gap:.75rem;padding:.4rem 0;border-bottom:1px solid #f1f5f9}
.top3-row:last-child{border-bottom:none}
.top3-label{flex:1;font-size:.875rem;color:#334155}
.top3-pct{font-size:.875rem;font-weight:600;color:#6366f1;min-width:45px;text-align:right}
.top3-bar{flex:2;background:#e2e8f0;border-radius:3px;height:5px}
.top3-bar-fill{height:5px;border-radius:3px;background:#a5b4fc}

/* Explanation */
.expl-box{background:#f0f4ff;border-left:3px solid #6366f1;padding:.8rem 1rem;border-radius:0 8px 8px 0;font-size:.88rem;color:#3730a3;margin-top:.75rem;line-height:1.6}

/* Threats */
.threat-badge{display:inline-flex;align-items:center;gap:.4rem;padding:.3rem .8rem;border-radius:20px;font-size:.78rem;font-weight:600;margin-bottom:.5rem}
.threat-crit{background:#fee2e2;color:#991b1b}
.threat-high{background:#fef3c7;color:#92400e}
.threat-med {background:#fff7ed;color:#9a3412}
.threat-none{background:#dcfce7;color:#166534}
.threat-card{border-left:3px solid;padding:.75rem 1rem;border-radius:0 8px 8px 0;margin-bottom:.75rem}
.threat-name{font-weight:600;font-size:.9rem;margin-bottom:.25rem}
.threat-expl{font-size:.82rem;color:#475569;line-height:1.5}
.threat-proto{font-size:.75rem;color:#94a3b8;margin-top:.25rem}

/* Stats grid */
.stats-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem}
.stat-item{background:#f8fafc;border-radius:8px;padding:.75rem;text-align:center}
.stat-val{font-size:1.05rem;font-weight:700;color:#1a1a2e}
.stat-lbl{font-size:.72rem;color:#94a3b8;margin-top:.15rem}

/* Spinner */
.spinner{display:none;text-align:center;padding:2rem;color:#94a3b8;font-size:.9rem}
.spin{display:inline-block;width:28px;height:28px;border:3px solid #e2e8f0;border-top-color:#6366f1;border-radius:50%;animation:spin .7s linear infinite;margin-bottom:.5rem}
@keyframes spin{to{transform:rotate(360deg)}}

#result{display:none}
.error-box{background:#fee2e2;border:1px solid #fecaca;border-radius:10px;padding:1rem;color:#991b1b;margin-bottom:1rem}
.demo-badge{background:#fef9c3;border:1px solid #fde047;border-radius:6px;padding:.4rem .8rem;font-size:.78rem;color:#854d0e;margin-bottom:1rem;display:inline-block}
</style>
</head>
<body>
<div class="container">
  <h1>🔍 Network Traffic Analyser</h1>
  <p class="subtitle">Upload a PCAP file — get website classification + cybersecurity threat detection</p>

  <div class="card" id="upload-card">
    <div class="card-title">Upload PCAP file</div>
    <label for="file-input">
      <div class="drop-zone" id="drop-zone">
        <div class="drop-icon">📂</div>
        <div class="drop-label">Drag &amp; drop your .pcap / .pcapng file here</div>
        <div class="drop-hint">or click to browse · max 50 MB</div>
        <div class="filename" id="filename"></div>
      </div>
    </label>
    <input type="file" id="file-input" accept=".pcap,.pcapng,.cap">
    <button class="btn" id="analyse-btn" onclick="analyse()">▶ Analyse Traffic</button>
  </div>

  <div class="spinner" id="spinner">
    <div class="spin"></div><br>Analysing packets… this may take a moment.
  </div>
  <div id="error-zone"></div>

  <div id="result">

    <div class="card">
      <div class="card-title">Classification result</div>
      <div class="metric-grid">
        <div class="metric-box">
          <div class="metric-label">Website / App</div>
          <div class="metric-value" id="r-category">—</div>
          <div class="metric-sub" id="r-conf"></div>
          <div class="bar-track"><div class="bar-fill" id="r-conf-bar" style="width:0%"></div></div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Packets analysed</div>
          <div class="metric-value" id="r-packets">—</div>
          <div class="metric-sub" id="r-windows"></div>
        </div>
      </div>
      <div class="expl-box" id="r-expl"></div>
    </div>

    <div class="card">
      <div class="card-title">Top 3 predictions</div>
      <div id="r-top3"></div>
    </div>

    <div class="card">
      <div class="card-title">Threat assessment</div>
      <div id="r-threats"></div>
    </div>

    <div class="card">
      <div class="card-title">Traffic statistics</div>
      <div class="stats-grid" id="r-stats"></div>
    </div>

  </div>
</div>

<script>
const inp = document.getElementById('file-input');
const dz  = document.getElementById('drop-zone');

inp.addEventListener('change', () => {
  if (inp.files[0]) document.getElementById('filename').textContent = inp.files[0].name;
});
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', ()=> dz.classList.remove('drag'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('drag');
  inp.files = e.dataTransfer.files;
  if (inp.files[0]) document.getElementById('filename').textContent = inp.files[0].name;
});

function analyse() {
  const f = inp.files[0];
  if (!f) { alert('Please select a PCAP file first.'); return; }
  document.getElementById('analyse-btn').disabled = true;
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('result').style.display  = 'none';
  document.getElementById('error-zone').innerHTML  = '';

  const fd = new FormData();
  fd.append('pcap', f);

  fetch('/predict', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(d => {
      document.getElementById('spinner').style.display = 'none';
      document.getElementById('analyse-btn').disabled  = false;
      if (d.error) {
        document.getElementById('error-zone').innerHTML =
          `<div class="error-box">⚠ ${d.error}</div>`;
        return;
      }
      render(d);
    })
    .catch(err => {
      document.getElementById('spinner').style.display = 'none';
      document.getElementById('analyse-btn').disabled  = false;
      document.getElementById('error-zone').innerHTML  =
        `<div class="error-box">Request failed: ${err}</div>`;
    });
}

function cap(s) { return s.replace(/_/g,' ').replace(/\\b\\w/g, c => c.toUpperCase()); }

function render(d) {
  document.getElementById('result').style.display = 'block';

  // Main result
  document.getElementById('r-category').textContent = d.category;
  const pct = Math.round(d.confidence * 100);
  document.getElementById('r-conf').textContent     = pct + '% confidence';
  document.getElementById('r-conf-bar').style.width = pct + '%';
  document.getElementById('r-packets').textContent  = d.n_packets.toLocaleString();
  document.getElementById('r-windows').textContent  = d.n_windows + ' windows evaluated';
  document.getElementById('r-expl').textContent     = d.explanation;

  // Demo badge
  if (d.demo_mode) {
    document.getElementById('result').insertAdjacentHTML('afterbegin',
      '<div class="demo-badge">⚠ Demo mode — place your model_artifacts/ folder next to app.py</div>');
  }

  // Top 3
  const top3html = (d.top3 || []).map(t => `
    <div class="top3-row">
      <span class="top3-label">${t.label}</span>
      <div class="top3-bar"><div class="top3-bar-fill" style="width:${Math.round(t.prob*100)}%"></div></div>
      <span class="top3-pct">${Math.round(t.prob*100)}%</span>
    </div>`).join('');
  document.getElementById('r-top3').innerHTML = top3html;

  // Threats
  const ts = d.threats || [];
  let thtml = '';
  if (ts.length === 0) {
    thtml = '<span class="threat-badge threat-none">✓ No threats detected — traffic appears benign</span>';
  } else {
    ts.forEach(t => {
      const badgeCls = t.severity === 'CRITICAL' ? 'threat-crit'
                     : t.severity === 'HIGH'     ? 'threat-high' : 'threat-med';
      thtml += `
        <div class="threat-card" style="border-color:${t.color};background:${t.color}11">
          <div class="threat-name" style="color:${t.color}">${t.name}
            <span class="threat-badge ${badgeCls}" style="margin-left:.5rem">${t.severity}</span>
          </div>
          <div class="threat-expl">${t.explanation}</div>
          <div class="threat-proto">Protocol: ${t.protocol}</div>
        </div>`;
    });
  }
  document.getElementById('r-threats').innerHTML = thtml;

  // Stats
  const st = d.traffic_stats || {};
  const statsItems = [
    { v: (st.mean_packet_length || 0) + ' B',   l: 'Avg packet size' },
    { v: (st.mean_inter_arrival || 0) + ' ms',  l: 'Avg inter-arrival' },
    { v: (st.tcp_percent || 0) + '%',            l: 'TCP traffic' },
    { v: (st.quic_percent || 0) + '%',           l: 'QUIC traffic' },
    { v: formatBytes(st.total_bytes || 0),       l: 'Total bytes' },
    { v: st.burst_max || 0,                      l: 'Max burst size' },
  ];
  document.getElementById('r-stats').innerHTML = statsItems.map(s =>
    `<div class="stat-item"><div class="stat-val">${s.v}</div><div class="stat-lbl">${s.l}</div></div>`
  ).join('');
}

function formatBytes(b) {
  if (b > 1e6) return (b/1e6).toFixed(1) + ' MB';
  if (b > 1e3) return (b/1e3).toFixed(1) + ' KB';
  return b + ' B';
}
</script>
</body>
</html>"""


# ── Flask routes ──────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    if "pcap" not in request.files or request.files["pcap"].filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["pcap"]
    suffix = os.path.splitext(f.filename)[1] or ".pcap"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = predict_pcap(tmp_path)
    except Exception as e:
        result = {"error": str(e)}
    finally:
        os.unlink(tmp_path)

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": MODEL is not None,
        "classes":      list(ENC.classes_) if ENC else [],
        "test_accuracy": META.get("test_accuracy") if META else None,
    })


if __name__ == "__main__":
    print("Starting Network Traffic Analyser...")
    print(f"Model directory : {MODEL_DIR}")
    print(f"Model loaded    : {MODEL is not None}")
    if META:
        print(f"Classes         : {META.get('classes', [])}")
        print(f"Training acc    : {META.get('test_accuracy', 'N/A')}")
    print("\nOpen: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)