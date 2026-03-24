from scapy.all import rdpcap
import numpy as np
from collections import Counter

# 🔹 Detect client IP automatically
def get_client_ip(packets):
    src_ips = []

    for pkt in packets:
        if 'IP' in pkt:
            src_ips.append(pkt['IP'].src)

    return Counter(src_ips).most_common(1)[0][0]


# 🔹 Feature extraction
def extract_features(pcap_file, max_len=3000):
    packets = rdpcap(pcap_file)

    my_ip = get_client_ip(packets)
    print(f"[INFO] File: {pcap_file}")
    print(f"[INFO] Detected client IP: {my_ip}")

    features = []
    prev_time = None

    for pkt in packets:
        if 'IP' not in pkt:
            continue

        # Direction
        if pkt['IP'].src == my_ip:
            direction = 1
        else:
            direction = -1

        # Time difference
        if prev_time is None:
            delta_time = 0.0
        else:
            delta_time = float(pkt.time - prev_time)

        prev_time = pkt.time

        # Packet size
        size = len(pkt)

        # 🔥 NEW: Protocol feature
        if pkt.haslayer("TCP"):
            protocol = 0
        elif pkt.haslayer("UDP"):
            protocol = 1
        else:
            protocol = 0

        features.append([direction, delta_time, size, protocol])

    # Fix length
    if len(features) > max_len:
        features = features[:max_len]
    else:
        features += [[0, 0, 0, 0]] * (max_len - len(features))

    return np.array(features)