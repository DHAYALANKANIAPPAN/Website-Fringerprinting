"""
Network Traffic Analyser — Flask Web App
=========================================
Run:
    python app.py
Open:
    http://localhost:5000
"""

import os, json, pickle, tempfile, traceback
import numpy as np
from collections import deque, defaultdict
from flask import Flask, request, jsonify, render_template_string

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_DIR   = "model_artifacts"
WINDOW_SIZE = 100
N_FEATURES  = 12
QUIC_PORTS  = {443, 80}
LOCAL_PFX   = ("192.168.", "10.", "172.16.", "172.17.", "172.18.",
               "172.19.", "172.2", "127.", "::1", "fe80:")

# Correct class names in alphabetical order (how LabelEncoder assigns indices)
# Index: 0=BBC News, 1=Coursera, 2=Discord, 3=Disney, 4=Dropbox,
#        5=Facebook, 6=github, 7=pinterest, 8=quora, 9=tumble
KNOWN_CLASSES = [
    'BBC News', 'Coursera', 'Discord', 'Disney', 'Dropbox',
    'Facebook', 'github', 'pinterest', 'quora', 'tumble'
]

FEATURE_NAMES = [
    "pkt_len","ip_len","trans_len","direction","iat",
    "protocol","tcp_flags","win_size","burst_size","flow_bytes",
    "roll_mean","roll_std"
]

THREAT_MAP = {
    "benign":            "✅ Normal traffic — no suspicious patterns detected.",
    "c2_beacon":         "⚠️  Possible C2 beacon: very regular timing + tiny packets suggest automated callbacks to a remote server.",
    "dns_tunneling":     "⚠️  Possible DNS tunneling: tiny regular UDP packets may encode data inside DNS queries.",
    "data_exfiltration": "🚨 Possible data exfiltration: large outbound bursts — unusual upload volume detected.",
    "port_scan":         "⚠️  Possible port scan: many small outbound packets — reconnaissance pattern.",
    "quic_evasion":      "⚠️  QUIC evasion pattern: encrypted QUIC bursts may conceal C2 or tunneling activity.",
}

CAT_MAP = {
    "bbc news":  "BBC News — HTTPS page loads, moderate-size packets, regular intervals.",
    "coursera":  "Coursera — video streaming bursts mixed with API calls.",
    "discord":   "Discord — small frequent UDP/QUIC packets (voice + chat), very bursty.",
    "youtube":   "YouTube — large sustained TCP flows, adaptive bitrate video.",
    "github":    "GitHub — HTTPS API calls + occasional large git object transfers.",
    "netflix":   "Netflix — very large sustained TCP flows, adaptive bitrate.",
    "dropbox":   "Dropbox — large TCP uploads/downloads, sustained flows.",
    "tumble":    "Tumblr — HTTP/HTTPS media-heavy loads with image bursts.",
    "quora":     "Quora — typical HTTPS browsing, medium frequency, moderate packet size.",
    "disney":    "Disney+ — large video streaming flows, similar to Netflix.",
    "facebook":  "Facebook — frequent small HTTPS requests mixed with media loads.",
    "pinterest": "Pinterest — image-heavy HTTPS traffic, bursty load patterns.",
}

# ── Load model artifacts ──────────────────────────────────────────────────────

def load_artifacts():
    import tensorflow as tf
    result = {"model": None, "scaler": None, "classes": KNOWN_CLASSES,
              "thr_enc": None, "loaded": False}

    # Model
    for name in ("best_model.keras", "final_model.keras"):
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p):
            try:
                result["model"] = tf.keras.models.load_model(p)
                print(f"[OK]  Model: {name}")
                break
            except Exception as e:
                print(f"[ERR] Could not load {name}: {e}")

    if result["model"] is None:
        print(f"[WARN] No model in {MODEL_DIR}/ — DEMO mode")
        return result

    # Scaler
    p = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            result["scaler"] = pickle.load(f)
        print(f"[OK]  Scaler loaded")

    # Class names — try encoder first, then metadata, then KNOWN_CLASSES fallback
    classes = None

    for enc_name in ("label_encoder.pkl", "cat_encoder.pkl", "encoder.pkl"):
        p = os.path.join(MODEL_DIR, enc_name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                enc = pickle.load(f)
            if hasattr(enc, "classes_") and len(enc.classes_) > 1:
                classes = list(enc.classes_)
                print(f"[OK]  Encoder: {enc_name} → {classes}")
                break
            else:
                print(f"[WARN] {enc_name} only has {len(enc.classes_)} class — skipping")

    if classes is None:
        for meta_name in ("metadata.json", "model_meta.json"):
            p = os.path.join(MODEL_DIR, meta_name)
            if os.path.exists(p):
                with open(p) as f:
                    m = json.load(f)
                clist = m.get("class_names", m.get("classes", m.get("category_classes", [])))
                if len(clist) > 1:
                    classes = clist
                    print(f"[OK]  Classes from {meta_name}: {classes}")
                    break

    if classes is None:
        # Check if model output size matches KNOWN_CLASSES
        try:
            out = result["model"].output
            n_out = (out[0] if isinstance(out, list) else out).shape[-1]
            if n_out == len(KNOWN_CLASSES):
                classes = KNOWN_CLASSES
                print(f"[OK]  Using hardcoded KNOWN_CLASSES ({n_out} classes): {classes}")
            else:
                classes = [f"class_{i}" for i in range(n_out)]
                print(f"[WARN] Model has {n_out} outputs but KNOWN_CLASSES has "
                      f"{len(KNOWN_CLASSES)} — using generic names. "
                      f"Run fix_encoder.py to fix this.")
        except Exception:
            classes = KNOWN_CLASSES

    result["classes"] = classes
    result["loaded"]  = result["model"] is not None and result["scaler"] is not None
    return result


ART = load_artifacts()

# ── Feature extraction ────────────────────────────────────────────────────────

def is_local(ip):
    return any(ip.startswith(p) for p in LOCAL_PFX)

def safe_flags(f):
    try:
        fi = int(f)
        return round(
            ((fi&0x02)>0)*0.4+((fi&0x10)>0)*0.1+
            ((fi&0x08)>0)*0.2+((fi&0x01)>0)*0.2+((fi&0x04)>0)*0.1, 4)
    except Exception:
        return 0.0

def extract_records(packets):
    from scapy.all import IP, TCP, UDP
    records = []
    state = defaultdict(lambda: {
        "last_t":None,"burst":0,"last_bt":None,
        "cum":0,"recent":deque(maxlen=10)
    })
    for pkt in packets:
        try:
            if not pkt.haslayer(IP): continue
            ip  = pkt[IP]
            t   = float(pkt.time)
            src, dst = str(ip.src), str(ip.dst)
            fkey = (min(src,dst), max(src,dst))
            s = state[fkey]

            pkt_len=int(len(pkt)); ip_len=int(len(ip))
            trans_len=proto_enc=0; flags_enc=win_size=sp=dp=0

            if pkt.haslayer(TCP):
                tcp=pkt[TCP]; sp,dp=int(tcp.sport),int(tcp.dport)
                trans_len=int(len(tcp.payload)); proto_enc=0
                flags_enc=safe_flags(tcp.flags); win_size=int(tcp.window)
            elif pkt.haslayer(UDP):
                udp=pkt[UDP]; sp,dp=int(udp.sport),int(udp.dport)
                trans_len=int(len(udp.payload))
                proto_enc=1 if (sp in QUIC_PORTS or dp in QUIC_PORTS) else 2

            direction = 1 if is_local(src) else -1
            iat = max(0.0, t-s["last_t"]) if s["last_t"] else 0.0
            s["last_t"] = t

            if s["last_bt"] is None or iat>0.05: s["burst"]=1
            else: s["burst"]+=1
            s["last_bt"]=t; s["cum"]+=pkt_len; s["recent"].append(pkt_len)

            rl=list(s["recent"])
            records.append([
                float(pkt_len), float(ip_len),  float(trans_len),
                float(direction), float(iat),   float(proto_enc),
                float(flags_enc), float(win_size),
                float(s["burst"]), float(s["cum"]),
                float(np.mean(rl)),
                float(np.std(rl)) if len(rl)>1 else 0.0
            ])
        except Exception:
            continue
    return records

def to_windows(records):
    if len(records) < WINDOW_SIZE: return None
    step = WINDOW_SIZE // 2
    wins = [records[i:i+WINDOW_SIZE]
            for i in range(0, len(records)-WINDOW_SIZE+1, step)
            if len(records[i:i+WINDOW_SIZE])==WINDOW_SIZE]
    return np.array(wins, dtype=np.float32) if wins else None

def rule_threat(X):
    flat=X.mean(axis=1)
    ml=float(flat[:,0].mean()); sl=float(flat[:,0].std())
    mi=float(flat[:,4].mean()); pr=int(round(float(flat[:,5].mean())))
    bm=float(X[:,:,8].max()); tb=float(X[:,:,9].max())
    dr=float((flat[:,3]==1).mean())
    if mi<2.0 and sl<30 and ml<200:         return "c2_beacon",0.72
    if pr==1 and bm>20 and 100<ml<600:      return "quic_evasion",0.68
    if tb>500000 and dr<0.3 and ml>800:     return "data_exfiltration",0.75
    if ml<80 and bm>15 and dr>0.8:          return "port_scan",0.70
    if pr==2 and ml<120 and sl<25:          return "dns_tunneling",0.65
    return "benign",0.85

# ── Prediction ────────────────────────────────────────────────────────────────

def predict_pcap(path):
    from scapy.all import rdpcap
    try:
        packets = rdpcap(path)
    except Exception as e:
        return {"error": f"Cannot read PCAP: {e}"}

    n_pkts = len(packets)
    if n_pkts < 10:
        return {"error": f"Only {n_pkts} packets — need at least {WINDOW_SIZE}."}

    records = extract_records(packets)
    if len(records) < WINDOW_SIZE:
        return {"error": (f"Only {len(records)} usable IP packets "
                          f"(need {WINDOW_SIZE}). PCAP may be too short.")}

    X_raw = to_windows(records)
    if X_raw is None:
        return {"error": "Could not build input windows."}

    n_win = len(X_raw)
    threat_label, threat_conf = rule_threat(X_raw)

    # Protocol summary
    pv = X_raw[:,:,5].flatten(); tot=len(pv)
    proto_summary = {
        "TCP":   f"{(pv==0).sum()/tot*100:.1f}%",
        "QUIC":  f"{(pv==1).sum()/tot*100:.1f}%",
        "Other": f"{(pv==2).sum()/tot*100:.1f}%",
    }

    flat = X_raw.mean(axis=1)
    flow_stats = {
        "mean_pkt_len": f"{flat[:,0].mean():.0f} bytes",
        "mean_iat_ms":  f"{flat[:,4].mean()*1000:.1f} ms",
        "max_burst":    f"{X_raw[:,:,8].max():.0f} packets",
        "total_kb":     f"{X_raw[:,:,9].max()/1024:.1f} KB",
        "pct_outgoing": f"{(flat[:,3]==1).mean()*100:.0f}%",
    }

    if not ART["loaded"]:
        return {
            "demo_mode":True, "n_packets":n_pkts, "n_windows":n_win,
            "category":"No model loaded", "category_confidence":0.0,
            "category_explanation":"Place model_artifacts/ next to app.py and restart.",
            "category_top3":[], "threat":threat_label,
            "threat_confidence":threat_conf,
            "threat_explanation":THREAT_MAP.get(threat_label,""),
            "protocol_summary":proto_summary, "flow_stats":flow_stats,
        }

    # Normalize
    try:
        N,W,F = X_raw.shape
        X_norm = ART["scaler"].transform(X_raw.reshape(-1,F)).reshape(N,W,F)
    except Exception as e:
        return {"error": f"Normalization failed: {e}"}

    # Inference
    try:
        raw_out = ART["model"].predict(X_norm, verbose=0)
    except Exception as e:
        return {"error": f"Model inference failed: {e}"}

    cat_probs_all = np.array(raw_out[0] if isinstance(raw_out,(list,tuple)) else raw_out)
    thr_probs_all = np.array(raw_out[1]) if isinstance(raw_out,(list,tuple)) and len(raw_out)>1 else None

    cat_probs = cat_probs_all.mean(axis=0)
    n_out     = len(cat_probs)
    classes   = ART["classes"]

    # If model output count doesn't match class list, fix it
    if n_out != len(classes):
        print(f"[WARN] Model outputs {n_out} classes, encoder has {len(classes)} — using KNOWN_CLASSES")
        if n_out == len(KNOWN_CLASSES):
            classes = KNOWN_CLASSES
        else:
            classes = [f"class_{i}" for i in range(n_out)]

    cat_idx   = int(np.argmax(cat_probs))
    cat_label = classes[cat_idx] if cat_idx < len(classes) else f"class_{cat_idx}"
    cat_conf  = round(float(cat_probs[cat_idx]), 4)

    top3_idx = np.argsort(cat_probs)[::-1][:3]
    cat_top3 = [
        {"label": classes[i] if i<len(classes) else f"class_{i}",
         "prob":  round(float(cat_probs[i]),4)}
        for i in top3_idx
    ]

    if thr_probs_all is not None and ART.get("thr_enc"):
        tp = thr_probs_all.mean(axis=0)
        ti = int(np.argmax(tp))
        tc = list(ART["thr_enc"].classes_)
        if ti < len(tc):
            threat_label = tc[ti]
            threat_conf  = round(float(tp[ti]),4)

    return {
        "n_packets":           n_pkts,
        "n_windows":           n_win,
        "category":            cat_label,
        "category_confidence": cat_conf,
        "category_explanation":CAT_MAP.get(cat_label.lower(),
                                f"Traffic classified as '{cat_label}'."),
        "category_top3":       cat_top3,
        "threat":              threat_label,
        "threat_confidence":   round(float(threat_conf),4),
        "threat_explanation":  THREAT_MAP.get(threat_label,""),
        "protocol_summary":    proto_summary,
        "flow_stats":          flow_stats,
    }

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Network Traffic Analyser</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#f0f2f5;color:#111;min-height:100vh;padding:2rem 1rem}
.wrap{max-width:740px;margin:0 auto}
h1{font-size:1.55rem;font-weight:700;margin-bottom:.2rem}
.sub{color:#666;font-size:.9rem;margin-bottom:1.75rem}
.card{background:#fff;border-radius:14px;border:1px solid #e2e2e2;padding:1.4rem;margin-bottom:1.2rem}
.ct{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#999;margin-bottom:.9rem}
.upload-box{border:2px dashed #d0d0d0;border-radius:10px;padding:2.5rem 1rem;text-align:center;cursor:pointer;transition:.2s}
.upload-box:hover{border-color:#555;background:#fafafa}
.icon{font-size:2rem;margin-bottom:.4rem}
.hint{color:#aaa;font-size:.82rem;margin-top:.3rem}
#fname{margin-top:.55rem;font-weight:600;color:#333;font-size:.88rem}
input[type=file]{display:none}
.btn{display:inline-block;margin-top:.9rem;padding:.65rem 1.7rem;background:#111;color:#fff;border-radius:9px;border:none;cursor:pointer;font-size:.9rem;transition:.15s}
.btn:hover{background:#333}.btn:disabled{background:#aaa;cursor:not-allowed}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.metric{background:#f7f7f7;border-radius:10px;padding:.95rem}
.ml{font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:#aaa;margin-bottom:.3rem}
.mv{font-size:1.2rem;font-weight:700}
.mc{font-size:.78rem;color:#bbb;margin-top:.2rem}
.bg{background:#eee;border-radius:4px;height:7px;margin-top:.5rem}
.bf{height:7px;border-radius:4px;background:#2563eb;transition:width .5s}
.bt{background:#ef4444}
.expl{border-left:3px solid #3b82f6;background:#eff6ff;padding:.7rem 1rem;border-radius:0 8px 8px 0;font-size:.87rem;color:#1e3a6e;margin-top:.9rem;line-height:1.55}
.expl-t{border-left-color:#f59e0b;background:#fff7ed;color:#78350f}
.badge{display:inline-block;padding:.22rem .8rem;border-radius:20px;font-size:.78rem;font-weight:600}
.bs{background:#dcfce7;color:#166534}.bw{background:#fef9c3;color:#854d0e}.bd{background:#fee2e2;color:#991b1b}
.row{display:flex;justify-content:space-between;align-items:center;padding:.42rem 0;border-bottom:1px solid #f3f3f3;font-size:.86rem}
.row:last-child{border:none}.rn{color:#555}.rv{font-weight:600;color:#111}
.t3r{display:flex;align-items:center;gap:.65rem;padding:.35rem 0;font-size:.86rem}
.t3b{flex:1;background:#eee;border-radius:3px;height:6px}
.t3f{height:6px;border-radius:3px;background:#2563eb}
.t3p{min-width:38px;text-align:right;color:#aaa;font-size:.78rem}
.chips{display:flex;gap:.6rem;flex-wrap:wrap;margin-top:.5rem}
.chip{background:#f0f0f0;border-radius:7px;padding:.28rem .7rem;font-size:.82rem}
.spin{display:none;text-align:center;padding:1.5rem;color:#999;font-size:.88rem}
.spin::after{content:'';display:block;width:26px;height:26px;border:3px solid #ddd;border-top-color:#555;border-radius:50%;margin:.6rem auto 0;animation:sp .7s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
#result{display:none}
.err{background:#fff0f0;border:1px solid #fca5a5;border-radius:10px;padding:1rem;color:#b91c1c;font-size:.88rem;margin-top:.75rem;white-space:pre-wrap}
.demo-tag{display:inline-block;background:#fef9c3;color:#854d0e;border-radius:5px;padding:.15rem .5rem;font-size:.72rem;margin-left:.5rem;vertical-align:middle}
</style>
</head>
<body>
<div class="wrap">
  <h1>Network Traffic Analyser <span id="dtag" style="display:none" class="demo-tag">demo</span></h1>
  <p class="sub">Upload a PCAP file to classify website traffic and detect potential threats.</p>

  <div class="card">
    <div class="ct">Upload PCAP file</div>
    <label for="fi">
      <div class="upload-box">
        <div class="icon">📂</div>
        <div>Drag &amp; drop <strong>.pcap</strong> / <strong>.pcapng</strong> here</div>
        <div class="hint">or click to browse &nbsp;·&nbsp; max 50 MB</div>
        <div id="fname"></div>
      </div>
    </label>
    <input type="file" id="fi" accept=".pcap,.pcapng,.cap">
    <br>
    <button class="btn" id="btn" onclick="go()">Analyse Traffic</button>
  </div>

  <div class="spin" id="spin">Analysing packets — please wait…</div>
  <div id="err"></div>

  <div id="result">
    <div class="card">
      <div class="ct">Classification result</div>
      <div class="g2">
        <div class="metric">
          <div class="ml">Website / App</div>
          <div class="mv" id="rc">—</div>
          <div class="mc" id="rcc"></div>
          <div class="bg"><div class="bf" id="bc" style="width:0"></div></div>
        </div>
        <div class="metric">
          <div class="ml">Threat level</div>
          <div class="mv" id="rt">—</div>
          <div class="mc" id="rtc"></div>
          <div class="bg"><div class="bf bt" id="bth" style="width:0"></div></div>
        </div>
      </div>
      <div class="expl" id="rce"></div>
    </div>

    <div class="card">
      <div class="ct">Top 3 predictions</div>
      <div id="t3"></div>
    </div>

    <div class="card">
      <div class="ct">Threat analysis</div>
      <div id="tbw"></div>
      <div class="expl expl-t" id="rte" style="margin-top:.8rem"></div>
    </div>

    <div class="card">
      <div class="ct">Protocol breakdown</div>
      <div class="chips" id="proto"></div>
    </div>

    <div class="card">
      <div class="ct">Flow statistics</div>
      <div id="fstats"></div>
    </div>

    <div class="card" style="font-size:.78rem;color:#bbb">
      <div id="meta"></div>
    </div>
  </div>
</div>

<script>
document.getElementById('fi').addEventListener('change',function(){
  const f=this.files[0];
  if(f) document.getElementById('fname').textContent='📄 '+f.name;
});

function cap(s){return String(s).replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());}

function go(){
  const f=document.getElementById('fi').files[0];
  if(!f){alert('Please select a PCAP file.');return;}
  const fd=new FormData(); fd.append('pcap',f);
  document.getElementById('spin').style.display='block';
  document.getElementById('result').style.display='none';
  document.getElementById('err').innerHTML='';
  document.getElementById('btn').disabled=true;

  fetch('/predict',{method:'POST',body:fd})
    .then(r=>{
      const ct=r.headers.get('content-type')||'';
      if(!ct.includes('json'))
        return r.text().then(t=>{throw new Error('Server error:\n'+t.substring(0,400));});
      return r.json();
    })
    .then(d=>{
      document.getElementById('spin').style.display='none';
      document.getElementById('btn').disabled=false;
      if(d.error){
        document.getElementById('err').innerHTML='<div class="err">⚠️ '+d.error+'</div>';
        return;
      }
      render(d);
    })
    .catch(e=>{
      document.getElementById('spin').style.display='none';
      document.getElementById('btn').disabled=false;
      document.getElementById('err').innerHTML='<div class="err">'+e.message+'</div>';
    });
}

function render(d){
  document.getElementById('result').style.display='block';
  if(d.demo_mode) document.getElementById('dtag').style.display='inline-block';

  const cp=Math.round((d.category_confidence||0)*100);
  const tp=Math.round((d.threat_confidence||0)*100);

  document.getElementById('rc').textContent=cap(d.category||'—');
  document.getElementById('rcc').textContent=cp+'% confidence';
  document.getElementById('bc').style.width=cp+'%';
  document.getElementById('rce').textContent=d.category_explanation||'';

  document.getElementById('rt').textContent=cap(d.threat||'—');
  document.getElementById('rtc').textContent=tp+'% confidence';
  document.getElementById('bth').style.width=tp+'%';
  document.getElementById('rte').textContent=d.threat_explanation||'';

  const t=d.threat||'benign';
  const bc=t==='benign'?'bs':t.includes('exfil')?'bd':'bw';
  document.getElementById('tbw').innerHTML=
    '<span class="badge '+bc+'">'+cap(t)+'</span>';

  document.getElementById('t3').innerHTML=(d.category_top3||[]).map(x=>{
    const p=Math.round((x.prob||0)*100);
    return '<div class="t3r"><span style="min-width:120px">'+cap(x.label)+
      '</span><div class="t3b"><div class="t3f" style="width:'+p+'%"></div></div>'+
      '<span class="t3p">'+p+'%</span></div>';
  }).join('')||'—';

  document.getElementById('proto').innerHTML=
    Object.entries(d.protocol_summary||{}).map(
      ([k,v])=>'<div class="chip"><strong>'+k+'</strong>: '+v+'</div>').join('');

  const fl=d.flow_stats||{};
  const lb={mean_pkt_len:'Mean packet length',mean_iat_ms:'Mean inter-arrival',
            max_burst:'Max burst size',total_kb:'Total flow volume',pct_outgoing:'Outgoing traffic'};
  document.getElementById('fstats').innerHTML=
    Object.entries(fl).map(([k,v])=>
      '<div class="row"><span class="rn">'+(lb[k]||k)+'</span>'+
      '<span class="rv">'+v+'</span></div>').join('');

  document.getElementById('meta').innerHTML=
    'Packets: <strong>'+d.n_packets+'</strong> &nbsp;|&nbsp; '+
    'Windows: <strong>'+d.n_windows+'</strong>'+
    (d.demo_mode?' &nbsp;|&nbsp; <em>No model loaded</em>':'');
}
</script>
</body>
</html>"""

# ── Flask ─────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "pcap" not in request.files:
            return jsonify({"error":"No file uploaded."}),400
        f = request.files["pcap"]
        if not f or not f.filename:
            return jsonify({"error":"Empty file."}),400
        suffix = os.path.splitext(f.filename)[1].lower() or ".pcap"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            f.save(tmp.name); tmp_path=tmp.name
        try:
            result = predict_pcap(tmp_path)
        finally:
            try: os.unlink(tmp_path)
            except: pass
        return jsonify(result)
    except Exception as e:
        return jsonify({"error":str(e),"detail":traceback.format_exc()}),500

@app.route("/health")
def health():
    return jsonify({"status":"ok","model_loaded":ART["loaded"],"classes":ART["classes"]})

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error":"File too large — max 50 MB."}),413

@app.errorhandler(500)
def err500(e):
    return jsonify({"error":str(e)}),500

if __name__=="__main__":
    print("\n"+"="*50)
    print("  Network Traffic Analyser")
    print(f"  Model loaded : {ART['loaded']}")
    print(f"  Classes      : {ART['classes']}")
    print("  URL          : http://localhost:5000")
    print("="*50+"\n")
    app.run(host="0.0.0.0", port=5000, debug=False)