#!/usr/bin/env python3
"""
Κρατά τις πρώτες N ροές ενός pcap (προεπιλογή 5.000), ΟΠΟΙΟΥΔΗΠΟΤΕ πρωτοκόλλου
(TCP, UDP, ICMP, ICMPv6, GRE, ESP, κίνηση εκτός IP κ.λπ.), τις γράφει σε CSV
και φτιάχνει ένα PNG με το είδος τους (IPv4 / IPv6, TCP / UDP / ICMP ...).

Χρήση:
    python3 pcap_first_flows.py part_00000_*.pcap first_5000_flows.csv
    python3 pcap_first_flows.py part_00000_*.pcap first_5000_flows.csv --n 5000 --png flows.png
    python3 pcap_first_flows.py 202606101400.pcap.gz first_5000_flows.csv --until 60

Χρειάζεται Python 3, matplotlib και tshark.

Πώς ορίζεται η ροή:
  - TCP: μία ροή ανά tcp.stream (όπως στο Statistics -> Conversations -> TCP του Wireshark).
  - UDP: μία ροή ανά udp.stream (διευθύνσεις και θύρες).
  - Τα υπόλοιπα πρωτόκολλα πάνω από IP (ICMP, ICMPv6, GRE, ESP, tunnels κ.ά.):
    μία ροή ανά ζεύγος διευθύνσεων IP και πρωτόκολλο, γιατί δεν έχουν θύρες.
  - Κίνηση εκτός IP (π.χ. ARP): μία ροή ανά ζεύγος διευθύνσεων MAC και πρωτόκολλο.
  - Μετράει η εξωτερική επικεφαλίδα του πακέτου. Ένα μήνυμα σφάλματος ICMP που
    μεταφέρει μέσα του επικεφαλίδα TCP είναι ICMP, και ένα tunnel (π.χ. IPv6 μέσα
    σε IPv4) είναι μία ροή του tunnel.

Οι «πρώτες N ροές» είναι οι N ροές που ξεκινούν πρώτες. Τα πακέτα τους μετρώνται
μέχρι το τέλος του αρχείου (ή μέχρι το --until), ώστε κάθε ροή να είναι πλήρης.
Τα υπόλοιπα πεδία υπολογίζονται όπως στο TCP.csv: A = πηγή του πρώτου πακέτου,
Bytes = frame.len, Bits/s κενό όταν Duration < 0,005 s. Οι γραμμές γράφονται με
σειρά έναρξης (Rel Start).
"""
import argparse
import csv
import os
import subprocess
import sys

HEADER = ["Network", "Protocol", "Address A", "Port A", "Address B", "Port B",
          "Packets", "Bytes", "Stream ID", "Packets A → B", "Bytes A → B",
          "Packets B → A", "Bytes B → A", "Rel Start", "Duration",
          "Bits/s A → B", "Bits/s B → A"]

FIELDS = ["frame.time_relative", "frame.len", "frame.protocols",
          "eth.src", "eth.dst",
          "ip.src", "ip.dst", "ip.proto", "ip.frag_offset",
          "ipv6.src", "ipv6.dst", "ipv6.nxt",
          "tcp.srcport", "tcp.dstport", "tcp.stream",
          "udp.srcport", "udp.dstport", "udp.stream"]
(T, LEN, PROTOS, ESRC, EDST, ISRC, IDST, IPROTO, IFRAG,
 V6SRC, V6DST, V6NXT, TSPORT, TDPORT, TSTREAM, USPORT, UDPORT, USTREAM) = range(len(FIELDS))

# Ονόματα του frame.protocols -> όνομα πρωτοκόλλου στο CSV
NAMES = {"tcp": "TCP", "udp": "UDP", "icmp": "ICMP", "icmpv6": "ICMPv6", "gre": "GRE",
         "esp": "ESP", "ah": "AH", "sctp": "SCTP", "igmp": "IGMP", "pim": "PIM",
         "ospf": "OSPF", "vrrp": "VRRP", "udplite": "UDP-Lite", "dccp": "DCCP",
         "l2tp": "L2TP", "eigrp": "EIGRP", "carp": "CARP"}
# Αριθμός πρωτοκόλλου IP -> όνομα, όταν το tshark δεν αναγνωρίζει το περιεχόμενο
PROTO_NUM = {"1": "ICMP", "2": "IGMP", "4": "IP-in-IP", "6": "TCP", "17": "UDP",
             "41": "6in4", "47": "GRE", "50": "ESP", "51": "AH", "58": "ICMPv6",
             "89": "OSPF", "103": "PIM", "112": "VRRP", "132": "SCTP"}
TUNNELS = {("IPv4", "ip"): "IP-in-IP", ("IPv4", "ipv6"): "6in4",
           ("IPv6", "ip"): "4in6", ("IPv6", "ipv6"): "IPv6-in-IPv6"}


def tshark_lines(pcap, tshark="tshark"):
    cmd = [tshark, "-r", pcap, "-n",
           "-o", "tcp.desegment_tcp_streams:FALSE",
           "-o", "tcp.analyze_sequence_numbers:FALSE",
           "-o", "ip.defragment:FALSE",
           "-o", "ipv6.defragment:FALSE",
           "-T", "fields", "-E", "separator=/t", "-E", "occurrence=f"]
    for f in FIELDS:
        cmd += ["-e", f]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    return proc


def classify(p):
    """Από ένα πακέτο: (δίκτυο, πρωτόκολλο, πηγή, θύρα πηγής, προορισμός, θύρα προορισμού, κλειδί ροής, Stream ID)."""
    toks = p[PROTOS].split(":")
    i = next((k for k, t in enumerate(toks) if t in ("ip", "ipv6")), None)

    if i is None:  # κίνηση εκτός IP (ARP, LLDP, STP ...)
        proto = (toks[-1] if toks and toks[-1] else "unknown").upper()
        src, dst = p[ESRC], p[EDST]
        return ("Non-IP", proto, src, "", dst, "",
                ("Non-IP", proto) + tuple(sorted((src, dst))), "")

    net = "IPv4" if toks[i] == "ip" else "IPv6"
    src, dst = (p[ISRC], p[IDST]) if net == "IPv4" else (p[V6SRC], p[V6DST])
    j = i + 1
    while j < len(toks) and toks[j].startswith("ipv6."):  # επεκτάσεις επικεφαλίδας IPv6
        j += 1
    nxt = toks[j] if j < len(toks) else ""

    if nxt == "tcp" and p[TSTREAM]:
        return (net, "TCP", src, p[TSPORT], dst, p[TDPORT], ("TCP", p[TSTREAM]), p[TSTREAM])
    if nxt == "udp" and p[USTREAM]:
        return (net, "UDP", src, p[USPORT], dst, p[UDPORT], ("UDP", p[USTREAM]), p[USTREAM])

    if nxt in ("ip", "ipv6"):
        proto = TUNNELS[(net, nxt)]
    elif nxt in NAMES:
        proto = NAMES[nxt]
    elif net == "IPv4" and p[IFRAG] not in ("", "0"):
        proto = "Fragment"
    elif net == "IPv6" and "ipv6.fraghdr" in toks[i:j]:
        proto = "Fragment"
    elif nxt and nxt != "data":
        proto = nxt.upper()
    else:
        num = p[IPROTO] if net == "IPv4" else p[V6NXT]
        proto = PROTO_NUM.get(num, f"IP proto {num}" if num else "Unknown")
    return (net, proto, src, "", dst, "", (net, proto) + tuple(sorted((src, dst))), "")


def collect(lines, n_flows, until=None, progress=True):
    """Κρατά τις πρώτες n_flows ροές και μετράει όλα τα πακέτα τους."""
    flows = {}
    read = used = 0
    for line in lines:
        p = line.rstrip("\n").split("\t")
        if len(p) < len(FIELDS):
            p += [""] * (len(FIELDS) - len(p))
        t = float(p[T])
        if until is not None and t > until:
            break
        read += 1
        if progress and read % 1_000_000 == 0:
            print(f"{read:,} πακέτα διαβάστηκαν", file=sys.stderr)
        net, proto, src, sport, dst, dport, key, sid = classify(p)
        f = flows.get(key)
        if f is None:
            if len(flows) >= n_flows:
                continue
            # [δίκτυο, πρωτόκολλο, A, θύρα A, B, θύρα B, Stream ID, πρώτο, τελευταίο,
            #  πακέτα A→B, byte A→B, πακέτα B→A, byte B→A]
            f = [net, proto, src, sport, dst, dport, sid, t, t, 0, 0, 0, 0]
            flows[key] = f
        ln = int(p[LEN])
        if t < f[7]:
            f[7] = t
        if t > f[8]:
            f[8] = t
        if src == f[2] and sport == f[3]:
            f[9] += 1
            f[10] += ln
        else:
            f[11] += 1
            f[12] += ln
        used += 1
    return list(flows.values()), read, used


def write_csv(flows, out, min_bps_duration=0.005):
    flows.sort(key=lambda f: f[7])
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, lineterminator="\r\n")
        w.writerow(HEADER)
        for net, proto, a, pa, b, pb, sid, first, last, pab, bab, pba, bba in flows:
            dur = last - first
            if dur >= min_bps_duration:
                bps_ab, bps_ba = int(bab * 8 / dur), int(bba * 8 / dur)
            else:
                bps_ab = bps_ba = ""
            w.writerow([net, proto, a, pa, b, pb, pab + pba, bab + bba, sid, pab, bab,
                        pba, bba, f"{first:.6f}", f"{dur:.6f}", bps_ab, bps_ba])


def label(net, proto):
    return f"{proto} (non-IP)" if net == "Non-IP" else f"{net} / {proto}"


def summarize(flows):
    """Ανά κατηγορία (δίκτυο / πρωτόκολλο): ροές, πακέτα, byte. Με φθίνουσα σειρά ροών."""
    cat = {}
    for f in flows:
        k = label(f[0], f[1])
        c = cat.setdefault(k, [0, 0, 0])
        c[0] += 1
        c[1] += f[9] + f[11]
        c[2] += f[10] + f[12]
    return sorted(cat.items(), key=lambda kv: (-kv[1][0], kv[0]))


def plot(flows, png, source_name, max_bars=12, max_segments=7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    INK, INK2, GRID, OTHER = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]

    cats = summarize(flows)
    n = len(flows)
    tot_p = sum(c[1] for _, c in cats)
    tot_b = sum(c[2] for _, c in cats)
    nets = {}
    for f in flows:
        nets[f[0]] = nets.get(f[0], 0) + 1
    net_text = " · ".join(f"{k} {100 * v / n:.1f}%" for k, v in
                          sorted(nets.items(), key=lambda kv: -kv[1]))
    start, end = min(f[7] for f in flows), max(f[7] for f in flows)

    # Πάνω διάγραμμα: ροές ανά κατηγορία (οι μικρές κατηγορίες ενώνονται στο «Other»)
    bars = cats[:max_bars]
    if len(cats) > max_bars:
        rest = cats[max_bars - 1:]
        bars = cats[:max_bars - 1] + [("Other", [sum(c[i] for _, c in rest) for i in range(3)])]
    # Κάτω διάγραμμα: μερίδιο σε ροές, πακέτα, byte
    segs = cats[:max_segments]
    if len(cats) > max_segments:
        rest = cats[max_segments - 1:]
        segs = cats[:max_segments - 1] + [("Other", [sum(c[i] for _, c in rest) for i in range(3)])]
    colors = [OTHER if name == "Other" else SERIES[k] for k, (name, _) in enumerate(segs)]

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK, "text.color": INK})
    fig = plt.figure(figsize=(11, 4.2 + 0.38 * len(bars)), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[max(len(bars), 3) * 0.38 + 0.6, 1.9], hspace=0.55)

    ax = fig.add_subplot(gs[0])
    names = [b[0] for b in bars][::-1]
    counts = [b[1][0] for b in bars][::-1]
    ax.barh(names, counts, color=SERIES[0], height=0.62, edgecolor="white", linewidth=2)
    xmax = max(counts)
    for y, c in enumerate(counts):
        ax.text(c + xmax * 0.01, y, f"{c:,}  ({100 * c / n:.1f}%)", va="center", color=INK2)
    ax.set_xlim(0, xmax * 1.22)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Number of flows")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"Flows per protocol: first {n:,} flows of {source_name}",
                 loc="left", fontsize=13, fontweight="bold", pad=24)
    ax.text(0, 1.02, f"Flow starts {start:.3f}–{end:.3f} s  ·  {net_text}  ·  "
                     f"{tot_p:,} packets  ·  {tot_b:,} bytes",
            transform=ax.transAxes, color=INK2, fontsize=9.5)

    ax2 = fig.add_subplot(gs[1])
    rows = [("Flows", 0, n), ("Packets", 1, tot_p), ("Bytes", 2, tot_b)][::-1]
    for y, (rname, idx, total) in enumerate(rows):
        left = 0.0
        for (name, c), col in zip(segs, colors):
            share = 100 * c[idx] / total if total else 0
            ax2.barh(y, share, left=left, color=col, height=0.62, edgecolor="white", linewidth=2,
                     label=name if y == 0 else None)
            if share >= 6:
                ax2.text(left + share / 2, y, f"{share:.0f}%", ha="center", va="center",
                         color="white", fontsize=9, fontweight="bold")
            left += share
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels([r[0] for r in rows])
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Share (%)")
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("Share of flows, packets and bytes per protocol", loc="left",
                  fontsize=11, fontweight="bold")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=min(4, len(segs)),
               frameon=False, handlelength=1.2, columnspacing=1.6)

    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcap", help="αρχείο .pcap ή .pcap.gz")
    ap.add_argument("out", help="αρχείο CSV εξόδου")
    ap.add_argument("--n", type=int, default=5000, help="πόσες ροές (προεπιλογή 5000)")
    ap.add_argument("--png", help="αρχείο PNG (προεπιλογή: ίδιο όνομα με το CSV)")
    ap.add_argument("--until", type=float,
                    help="σταματά το διάβασμα μετά από τόσα δευτερόλεπτα (π.χ. 60)")
    ap.add_argument("--tshark", default="tshark", help="διαδρομή του tshark")
    ap.add_argument("--fields-file", help=argparse.SUPPRESS)  # για δοκιμές
    a = ap.parse_args()

    proc = None
    if a.fields_file:
        lines = open(a.fields_file, encoding="utf-8")
    else:
        proc = tshark_lines(a.pcap, a.tshark)
        lines = proc.stdout
    flows, read, used = collect(lines, a.n, a.until)
    if proc:
        if a.until is not None and proc.poll() is None:
            proc.terminate()  # σταματήσαμε νωρίτερα σκόπιμα
            proc.wait()
        elif proc.wait() != 0:
            sys.exit(f"Το tshark σταμάτησε με κωδικό {proc.returncode}")
    if not flows:
        sys.exit("Δεν βρέθηκε καμία ροή.")

    write_csv(flows, a.out)
    png = a.png or os.path.splitext(a.out)[0] + ".png"
    plot(flows, png, os.path.basename(a.pcap))

    n = len(flows)
    print(f"\nΔιαβάστηκαν {read:,} πακέτα· {used:,} ανήκουν στις {n:,} πρώτες ροές.")
    print(f"\n{'Κατηγορία':<24}{'Ροές':>8}{'%':>8}{'Πακέτα':>12}{'Byte':>16}")
    for name, (fl, pk, by) in summarize(flows):
        print(f"{name:<24}{fl:>8,}{100 * fl / n:>7.1f}%{pk:>12,}{by:>16,}")
    if n < a.n:
        print(f"\nΠΡΟΣΟΧΗ: το αρχείο έχει μόνο {n:,} ροές.")
    print(f"\nCSV: {a.out}\nPNG: {png}")


if __name__ == "__main__":
    main()
