"""
PCAP Feature Extractor — 12 features per packet
Converts raw PCAP files into ML-ready CSV datasets.

Requirements:
    pip install scapy pandas numpy tqdm

Usage:
    python feature_extractor.py --pcap_dir ./pcaps --label bbc_news --output dataset.csv
    python feature_extractor.py --pcap_dir ./pcaps --label discord   --output dataset.csv --append

Features extracted per packet:
    1.  packet_length       — total bytes
    2.  ip_length           — IP payload size
    3.  transport_length    — TCP/UDP payload size
    4.  direction           — +1 outgoing (src=local), -1 incoming
    5.  iat                 — inter-arrival time (seconds)
    6.  protocol            — 0=TCP, 1=QUIC/UDP, 2=other
    7.  tcp_flags           — encoded (SYN/ACK/FIN/RST/PSH)
    8.  window_size         — TCP window size (0 for UDP/QUIC)
    9.  burst_size          — running packets in current burst (<50ms gap)
    10. flow_bytes_so_far   — cumulative bytes in this flow
    11. pkt_len_rolling_mean— rolling mean of last 10 packet lengths
    12. pkt_len_rolling_std — rolling std  of last 10 packet lengths
"""

import argparse
import os
import csv
import time
import numpy as np
from collections import defaultdict, deque

try:
    from scapy.all import rdpcap, IP, TCP, UDP, IPv6
    from scapy.layers.inet import ICMP
except ImportError:
    raise ImportError("Install scapy:  pip install scapy")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ── Helpers ──────────────────────────────────────────────────────────────────

LOCAL_PREFIXES = ("192.168.", "10.", "172.16.", "172.17.", "172.18.",
                  "172.19.", "172.2",  "127.",    "::1",    "fe80:")

QUIC_PORTS = {443, 80}          # QUIC typically runs on UDP 443/80

WINDOW_SIZE = 100               # packets per sample window


def is_local(ip: str) -> bool:
    return any(ip.startswith(p) for p in LOCAL_PREFIXES)


def encode_tcp_flags(flags_int: int) -> float:
    """
    Map TCP flag bitmask to a single float in [0,1].
    Bits: FIN=0x01 SYN=0x02 RST=0x04 PSH=0x08 ACK=0x10 URG=0x20
    We weight the most security-relevant flags more heavily.
    """
    fin = (flags_int & 0x01) > 0
    syn = (flags_int & 0x02) > 0
    rst = (flags_int & 0x04) > 0
    psh = (flags_int & 0x08) > 0
    ack = (flags_int & 0x10) > 0
    # Weighted sum, normalised to roughly [0,1]
    score = (syn * 0.4) + (ack * 0.1) + (psh * 0.2) + (fin * 0.2) + (rst * 0.1)
    return round(score, 4)


def extract_flow_key(pkt):
    """Return a canonical (src_ip, dst_ip, src_port, dst_port, proto) tuple."""
    if not pkt.haslayer(IP):
        return None
    ip = pkt[IP]
    proto = "tcp" if pkt.haslayer(TCP) else ("udp" if pkt.haslayer(UDP) else "other")
    src_port = dst_port = 0
    if pkt.haslayer(TCP):
        src_port, dst_port = pkt[TCP].sport, pkt[TCP].dport
    elif pkt.haslayer(UDP):
        src_port, dst_port = pkt[UDP].sport, pkt[UDP].dport
    # Canonical: lower IP first so fwd/rev share same key
    if ip.src < ip.dst:
        return (ip.src, ip.dst, src_port, dst_port, proto)
    else:
        return (ip.dst, ip.src, dst_port, src_port, proto)


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_features_from_pcap(pcap_path: str, label: str) -> list:
    """
    Returns a list of sample rows.
    Each sample = WINDOW_SIZE packets × 12 features, flattened + label.
    """
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"  [WARN] Could not read {pcap_path}: {e}")
        return []

    if len(packets) < WINDOW_SIZE:
        print(f"  [WARN] Only {len(packets)} packets in {pcap_path} — skipping")
        return []

    # ── Per-packet feature extraction ────────────────────────────────────────
    records = []
    flow_state = defaultdict(lambda: {
        "last_time": None,
        "burst_count": 0,
        "last_burst_time": None,
        "cumulative_bytes": 0,
        "recent_lengths": deque(maxlen=10),
    })

    for pkt in packets:
        if not pkt.haslayer(IP):
            continue

        ip_layer = pkt[IP]
        pkt_time = float(pkt.time)
        fkey = extract_flow_key(pkt)
        if fkey is None:
            continue

        state = flow_state[fkey]

        # 1. Lengths
        pkt_len   = len(pkt)
        ip_len    = len(ip_layer)
        trans_len = 0
        if pkt.haslayer(TCP):
            trans_len = len(pkt[TCP].payload)
        elif pkt.haslayer(UDP):
            trans_len = len(pkt[UDP].payload)

        # 2. Direction  (+1 = outgoing from local, -1 = incoming)
        direction = 1 if is_local(ip_layer.src) else -1

        # 3. Inter-arrival time
        iat = 0.0
        if state["last_time"] is not None:
            iat = max(0.0, pkt_time - state["last_time"])
        state["last_time"] = pkt_time

        # 4. Protocol encoding
        if pkt.haslayer(TCP):
            proto_enc = 0
        elif pkt.haslayer(UDP):
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
            proto_enc = 1 if (sport in QUIC_PORTS or dport in QUIC_PORTS) else 2
        else:
            proto_enc = 2

        # 5. TCP flags
        tcp_flags_enc = 0.0
        win_size      = 0
        if pkt.haslayer(TCP):
            tcp_flags_enc = encode_tcp_flags(int(pkt[TCP].flags))
            win_size      = pkt[TCP].window

        # 6. Burst tracking  (burst = gap < 50 ms)
        burst_threshold = 0.050
        if state["last_burst_time"] is None or (iat > burst_threshold):
            state["burst_count"] = 1
        else:
            state["burst_count"] += 1
        state["last_burst_time"] = pkt_time
        burst_size = state["burst_count"]

        # 7. Cumulative bytes in flow
        state["cumulative_bytes"] += pkt_len
        flow_bytes = state["cumulative_bytes"]

        # 8. Rolling statistics on packet length
        state["recent_lengths"].append(pkt_len)
        recent = list(state["recent_lengths"])
        roll_mean = float(np.mean(recent))
        roll_std  = float(np.std(recent)) if len(recent) > 1 else 0.0

        records.append([
            pkt_len, ip_len, trans_len,
            direction, round(iat, 6),
            proto_enc, tcp_flags_enc, win_size,
            burst_size, flow_bytes,
            round(roll_mean, 2), round(roll_std, 2),
        ])

    # ── Sliding-window sampling ───────────────────────────────────────────────
    samples = []
    step = WINDOW_SIZE // 2          # 50% overlap between windows

    for start in range(0, len(records) - WINDOW_SIZE + 1, step):
        window = records[start: start + WINDOW_SIZE]
        flat   = [val for pkt_row in window for val in pkt_row]
        flat.append(label)
        samples.append(flat)

    return samples


def build_header(n_features: int = 12, window: int = WINDOW_SIZE) -> list:
    feature_names = [
        "pkt_len", "ip_len", "trans_len",
        "direction", "iat",
        "protocol", "tcp_flags", "win_size",
        "burst_size", "flow_bytes",
        "roll_mean", "roll_std",
    ]
    cols = []
    for i in range(window):
        for fn in feature_names:
            cols.append(f"p{i:03d}_{fn}")
    cols.append("label")
    return cols


# ── CLI ───────────────────────────────────────────────────────────────────────

def process_directory(pcap_dir: str, label: str, output_csv: str, append: bool):
    pcap_files = [
        os.path.join(pcap_dir, f)
        for f in os.listdir(pcap_dir)
        if f.lower().endswith((".pcap", ".pcapng", ".cap"))
    ]

    if not pcap_files:
        print(f"No PCAP files found in {pcap_dir}")
        return

    print(f"Found {len(pcap_files)} PCAP file(s) for label '{label}'")

    all_samples = []
    iterator = tqdm(pcap_files, desc="Extracting") if HAS_TQDM else pcap_files
    for fp in iterator:
        samples = extract_features_from_pcap(fp, label)
        all_samples.extend(samples)
        if not HAS_TQDM:
            print(f"  {fp}: {len(samples)} samples")

    if not all_samples:
        print("No samples extracted — check PCAP size or network layer presence.")
        return

    mode = "a" if append and os.path.exists(output_csv) else "w"
    write_header = (mode == "w")

    with open(output_csv, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(build_header())
        writer.writerows(all_samples)

    print(f"\n✓ Saved {len(all_samples)} samples → {output_csv}  (mode={mode})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ML features from PCAP files")
    parser.add_argument("--pcap_dir", required=True,
                        help="Directory containing .pcap / .pcapng files")
    parser.add_argument("--label",    required=True,
                        help="Class label for this capture (e.g. discord, bbc_news)")
    parser.add_argument("--output",   default="dataset.csv",
                        help="Output CSV file path")
    parser.add_argument("--append",   action="store_true",
                        help="Append to existing CSV (for adding multiple classes)")
    args = parser.parse_args()

    process_directory(args.pcap_dir, args.label, args.output, args.append)