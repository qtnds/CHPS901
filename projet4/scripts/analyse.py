#!/usr/bin/env python3
"""
analyse.py — Analyse complète des benchmarks Barnes-Hut parallèle

Figures générées dans results/figures/ :
  fig01_ref_breakdown.png          Décomposition tree/force/integ de la référence vs N
  fig02_omp_speedup_vs_N.png       Speedup OMP vs N (T=max)
  fig03_omp_speedup_vs_T.png       Speedup OMP vs threads (N=max)
  fig04_omp_efficiency.png         Efficacité OMP
  fig05_omp_breakdown.png          Barres empilées tree/force/integ par version OMP
  fig06_mpi_speedup_vs_ranks.png   Speedup MPI vs rangs
  fig07_mpi_breakdown.png          Barres empilées MPI par version
  fig08_hybrid_speedup.png         Speedup hybride vs procs totaux
  fig09_hybrid_breakdown.png       Barres empilées hybride + comm_us par N
  fig10_cuda_breakdown_vs_N.png    Barres empilées CUDA (tree/serial/gpu/integ) vs N
  fig11_cuda_crossover.png         Crossover GPU/CPU : temps absolu CUDA vs séq vs hybride
  fig12_cuda_breakdown_pct.png     Pourcentage de chaque phase CUDA vs N (stacked area)
  fig13_global_comparison.png      Vue globale meilleur de chaque phase

Usage :
    python3 scripts/analyse.py [--results results/]
"""

import argparse, csv, os, sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib absent — tableaux seulement", file=sys.stderr)

# ── Palette ───────────────────────────────────────────────────
C = {
    "seq":    "#555555",
    "v1":     "#4C72B0", "v2": "#DD8452", "v3": "#55A868", "v3s": "#C44E52",
    "mpi_v1": "#4C72B0", "mpi_v2": "#DD8452", "mpi_v3": "#55A868",
    "hybrid": "#8172B3",
    "cuda":   "#C44E52",
    # Composantes de temps
    "tree":      "#4C72B0",
    "force":     "#DD8452",
    "integrate": "#55A868",
    "comm":      "#E377C2",
    "serial":    "#9ECAE1",
    "gpu":       "#C44E52",
}
MK = {"seq":"x","v1":"o","v2":"s","v3":"^","v3s":"D",
      "mpi_v1":"o","mpi_v2":"s","mpi_v3":"^","hybrid":"P","cuda":"*"}
OMP_LABELS = {"v1":"static","v2":"dynamic(32)","v3":"guided(16)","v3s":"guided+sort"}
MPI_LABELS = {"v1":"bandes-Y 7f","v2":"blocs-idx 7f","v3":"bandes-Y 3f"}

DPI = 150
W, H = 9, 5

# ── I/O ───────────────────────────────────────────────────────

def load(path):
    rows = []
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                out = {}
                for k, v in r.items():
                    try:    out[k] = int(v)
                    except:
                        try:    out[k] = float(v)
                        except: out[k] = v
                rows.append(out)
    except FileNotFoundError:
        pass
    return rows

def save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")

# ── Helpers console ───────────────────────────────────────────

def sep(n=70): print("─"*n)
def hdr(t): print(); sep(); print(f"  {t}"); sep()

def fmt_us(v):
    if v is None or v == 0: return "—"
    if v >= 1_000_000: return f"{v/1e6:.2f}s"
    if v >= 1_000:     return f"{v/1e3:.1f}ms"
    return f"{int(v)}µs"

def spd(ref, t):
    if not ref or not t or t == 0: return "—"
    return f"{ref/t:.2f}x"

# ── Fig helpers ───────────────────────────────────────────────

def stacked_bars(ax, x_labels, series, colors, title, ylabel="Temps (ms)"):
    """series = list of (label, values_in_ms)"""
    x = np.arange(len(x_labels))
    bottom = np.zeros(len(x_labels))
    for label, vals in series:
        vals = np.array(vals, dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label, color=colors[label], width=0.5)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(x_labels, rotation=10, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left"); ax.grid(axis="y", alpha=0.3)

def stacked_area_pct(ax, x_vals, series_dict, colors, title):
    """series_dict = {label: [pct values]}. X axis log scale."""
    labels  = list(series_dict.keys())
    data    = np.array([series_dict[l] for l in labels], dtype=float)
    # Normaliser en pourcentage
    total   = data.sum(axis=0)
    total[total == 0] = 1
    pct     = data / total * 100
    bottom  = np.zeros(len(x_vals))
    for i, label in enumerate(labels):
        ax.fill_between(x_vals, bottom, bottom + pct[i],
                        alpha=0.75, color=colors[label], label=label)
        bottom += pct[i]
    ax.set_xscale("log"); ax.set_ylim(0, 100)
    ax.set_xlabel("N particules"); ax.set_ylabel("% du temps total")
    ax.set_title(title); ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.3)

# ══════════════════════════════════════════════════════════════
#  Analyse référence
# ══════════════════════════════════════════════════════════════

def analyse_ref(seq_rows):
    hdr("Référence séquentielle (BH.cpp original)")
    print(f"  {'N':>8}  {'total':>9}  {'tree':>9}  {'force':>9}  {'integ':>9}  "
          f"{'tree%':>6}  {'force%':>6}")
    sep()
    ref = {}
    for r in sorted(seq_rows, key=lambda x: x["n_particles"]):
        N = r["n_particles"]
        ref[N] = r["total_us"]
        tot = r["total_us"] or 1
        print(f"  {N:>8}  {fmt_us(r['total_us']):>9}  "
              f"{fmt_us(r.get('tree_us')):>9}  "
              f"{fmt_us(r.get('force_us')):>9}  "
              f"{fmt_us(r.get('integrate_us')):>9}  "
              f"{r.get('tree_us',0)/tot*100:>5.1f}%  "
              f"{r.get('force_us',0)/tot*100:>5.1f}%")
    return ref

def fig_ref_breakdown(seq_rows, figdir):
    if not HAS_MPL or not seq_rows: return
    rows = sorted(seq_rows, key=lambda x: x["n_particles"])
    labels = [str(r["n_particles"]) for r in rows]
    series = [
        ("tree",      [r.get("tree_us",0)/1e3      for r in rows]),
        ("force",     [r.get("force_us",0)/1e3     for r in rows]),
        ("integrate", [r.get("integrate_us",0)/1e3 for r in rows]),
    ]
    fig, ax = plt.subplots(figsize=(W, H))
    stacked_bars(ax, labels, series, C,
                 "Référence BH.cpp — décomposition tree/force/intégration")
    save(fig, f"{figdir}/fig01_ref_breakdown.png")

# ══════════════════════════════════════════════════════════════
#  Phase 1 : OMP
# ══════════════════════════════════════════════════════════════

def analyse_omp(omp_rows, ref):
    hdr("Phase 1 — OpenMP : speedup vs référence BH.cpp")
    data = defaultdict(dict)
    for r in omp_rows:
        data[r["version"]][(r["n_particles"], r["n_threads"])] = r
    all_N = sorted({r["n_particles"] for r in omp_rows})
    all_T = sorted({r["n_threads"]   for r in omp_rows})
    for N in all_N:
        print(f"\n  N={N:>7}  (ref={fmt_us(ref.get(N))})")
        print(f"  {'version':<16}" + "".join(f"  T={T:<4}" for T in all_T))
        for ver in sorted(data):
            row = f"  {OMP_LABELS.get(ver,ver):<16}"
            for T in all_T:
                r = data[ver].get((N, T))
                row += f"  {spd(ref.get(N), r['total_us'] if r else None):>6}"
            print(row)
    return data, all_N, all_T

def figs_omp(omp_data, ref, all_N, all_T, figdir):
    if not HAS_MPL: return
    T_max = max(all_T) if all_T else 1
    N_max = max(all_N) if all_N else 1

    # Fig02 : speedup vs N
    fig, ax = plt.subplots(figsize=(W, H))
    for ver in sorted(omp_data):
        xs, ys = [], []
        for N in all_N:
            r = omp_data[ver].get((N, T_max))
            if r and ref.get(N):
                xs.append(N); ys.append(ref[N]/r["total_us"])
        if xs:
            ax.plot(xs, ys, marker=MK.get(ver,"o"), color=C.get(ver,"k"),
                    label=OMP_LABELS.get(ver,ver), lw=2)
    ax.axhline(T_max, color="grey", ls="--", lw=1, label=f"idéal {T_max}x")
    ax.set_xscale("log"); ax.set_xlabel("N particules"); ax.set_ylabel("Speedup")
    ax.set_title(f"OMP — Speedup vs N  (T={T_max}, ref=BH.cpp)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, f"{figdir}/fig02_omp_speedup_vs_N.png")

    # Fig03 : speedup vs T
    fig, ax = plt.subplots(figsize=(7, H))
    for ver in sorted(omp_data):
        xs, ys = [], []
        for T in all_T:
            r = omp_data[ver].get((N_max, T))
            if r and ref.get(N_max):
                xs.append(T); ys.append(ref[N_max]/r["total_us"])
        if xs:
            ax.plot(xs, ys, marker=MK.get(ver,"o"), color=C.get(ver,"k"),
                    label=OMP_LABELS.get(ver,ver), lw=2)
    ax.plot(all_T, all_T, "k--", lw=1, label="idéal")
    ax.set_xlabel("Threads OMP"); ax.set_ylabel("Speedup")
    ax.set_title(f"OMP — Speedup vs T  (N={N_max})"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, f"{figdir}/fig03_omp_speedup_vs_T.png")

    # Fig04 : efficacité
    fig, ax = plt.subplots(figsize=(7, H))
    for ver in sorted(omp_data):
        xs, ys = [], []
        for T in all_T:
            r = omp_data[ver].get((N_max, T))
            if r and ref.get(N_max) and T > 0:
                xs.append(T); ys.append(ref[N_max]/r["total_us"]/T*100)
        if xs:
            ax.plot(xs, ys, marker=MK.get(ver,"o"), color=C.get(ver,"k"),
                    label=OMP_LABELS.get(ver,ver), lw=2)
    ax.axhline(100, color="grey", ls="--", lw=1, label="idéal 100%")
    ax.set_ylim(0, 115); ax.set_xlabel("Threads OMP"); ax.set_ylabel("Efficacité (%)")
    ax.set_title(f"OMP — Efficacité parallèle (N={N_max})")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, f"{figdir}/fig04_omp_efficiency.png")

    # Fig05 : breakdown barres empilées (T=max, toutes versions)
    fig, ax = plt.subplots(figsize=(W, H))
    vers = sorted(omp_data)
    labels = [OMP_LABELS.get(v,v) for v in vers]
    series = []
    for comp, field in [("tree","tree_us"),("force","force_us"),("integrate","integrate_us")]:
        vals = []
        for ver in vers:
            r = omp_data[ver].get((N_max, T_max))
            vals.append(r.get(field,0)/1e3 if r else 0)
        series.append((comp, vals))
    stacked_bars(ax, labels, series, C,
                 f"OMP — Décomposition  (N={N_max}, T={T_max})")
    save(fig, f"{figdir}/fig05_omp_breakdown.png")

# ══════════════════════════════════════════════════════════════
#  Phase 2 : MPI
# ══════════════════════════════════════════════════════════════

def analyse_mpi(mpi_rows, ref):
    hdr("Phase 2 — MPI × OMP : speedup vs référence BH.cpp")
    data = defaultdict(dict)
    for r in mpi_rows:
        data[r["version"]][(r["n_particles"],r["n_ranks"],r["n_threads"])] = r
    all_N = sorted({r["n_particles"] for r in mpi_rows})
    all_R = sorted({r["n_ranks"]     for r in mpi_rows})
    all_T = sorted({r["n_threads"]   for r in mpi_rows})
    for N in all_N:
        for T in all_T[:1]:  # résumé T=min
            print(f"\n  N={N:>7}  T_omp={T}  (ref={fmt_us(ref.get(N))})")
            print(f"  {'version':<18}" + "".join(f"  R={R:<3}" for R in all_R))
            for ver in sorted(data):
                row = f"  {MPI_LABELS.get(ver,ver):<18}"
                for R in all_R:
                    r = data[ver].get((N,R,T))
                    row += f"  {spd(ref.get(N), r['total_us'] if r else None):>6}"
                print(row)
    return data, all_N, all_R, all_T

def figs_mpi(mpi_data, ref, all_N, all_R, all_T, figdir):
    if not HAS_MPL: return
    T_mpi = all_T[0] if all_T else 1
    N_max = max(all_N) if all_N else 1
    R_max = max(all_R) if all_R else 1

    # Fig06 : speedup vs rangs
    fig, ax = plt.subplots(figsize=(W, H))
    for ver in sorted(mpi_data):
        xs, ys = [], []
        for R in all_R:
            r = mpi_data[ver].get((N_max, R, T_mpi))
            if r and ref.get(N_max):
                xs.append(R); ys.append(ref[N_max]/r["total_us"])
        if xs:
            ax.plot(xs, ys, marker=MK.get(ver,"o"), color=C.get(f"mpi_{ver}","k"),
                    label=MPI_LABELS.get(ver,ver), lw=2)
    if all_R: ax.plot(all_R, all_R, "k--", lw=1, label="idéal")
    ax.set_xlabel("Rangs MPI"); ax.set_ylabel("Speedup vs ref BH.cpp")
    ax.set_title(f"MPI — Speedup vs rangs  (N={N_max}, T_omp={T_mpi})")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, f"{figdir}/fig06_mpi_speedup_vs_ranks.png")

    # Fig07 : breakdown
    fig, ax = plt.subplots(figsize=(W, H))
    vers = sorted(mpi_data)
    labels = [MPI_LABELS.get(v,v) for v in vers]
    series = []
    for comp, field in [("tree","tree_us"),("force","force_us"),("integrate","integrate_us")]:
        vals = []
        for ver in vers:
            r = mpi_data[ver].get((N_max, R_max, T_mpi))
            vals.append(r.get(field,0)/1e3 if r else 0)
        series.append((comp, vals))
    stacked_bars(ax, labels, series, C,
                 f"MPI — Décomposition  (N={N_max}, R={R_max}, T={T_mpi})")
    save(fig, f"{figdir}/fig07_mpi_breakdown.png")

# ══════════════════════════════════════════════════════════════
#  Phase 3 : Hybride
# ══════════════════════════════════════════════════════════════

def analyse_hybrid(hyb_rows, ref):
    if not hyb_rows: return {}, [], [], []
    hdr("Phase 3 — Hybride MPI+OMP : speedup vs référence BH.cpp")
    data = {}
    for r in hyb_rows:
        data[(r["n_particles"],r["n_ranks"],r["n_threads"])] = r
    all_N = sorted({r["n_particles"] for r in hyb_rows})
    all_R = sorted({r["n_ranks"]     for r in hyb_rows})
    all_T = sorted({r["n_threads"]   for r in hyb_rows})
    for N in all_N:
        print(f"\n  N={N:>7}  (ref={fmt_us(ref.get(N))})")
        hdr_rt = f"{'R/T':<5}"
        print(f"  {hdr_rt}" + "".join(f"  T={T:<5}" for T in all_T))
        for R in all_R:
            row = f"  R={R:<3}"
            for T in all_T:
                r = data.get((N,R,T))
                row += f"  {spd(ref.get(N), r['total_us'] if r else None):>7}"
            print(row)
    return data, all_N, all_R, all_T

def figs_hybrid(hyb_data, ref, all_N, all_R, all_T, figdir):
    if not HAS_MPL or not hyb_data: return
    N_max = max(all_N) if all_N else 1
    R_max = max(all_R) if all_R else 1

    # Fig08 : speedup vs procs totaux
    fig, ax = plt.subplots(figsize=(W, H))
    all_T_h = sorted({k[2] for k in hyb_data})
    for T in all_T_h:
        xs, ys = [], []
        for R in all_R:
            r = hyb_data.get((N_max, R, T))
            if r and ref.get(N_max):
                xs.append(R*T); ys.append(ref[N_max]/r["total_us"])
        if xs:
            pts = sorted(zip(xs,ys)); xx,yy = zip(*pts)
            ax.plot(xx, yy, marker="P", color=C["hybrid"], label=f"T={T}", lw=2)
    procs = sorted({k[1]*k[2] for k in hyb_data})
    if procs: ax.plot(procs, procs, "k--", lw=1, label="idéal")
    ax.set_xlabel("Procs totaux (rangs × threads)")
    ax.set_ylabel("Speedup vs ref BH.cpp")
    ax.set_title(f"Hybride — Speedup  (N={N_max})")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, f"{figdir}/fig08_hybrid_speedup.png")

    # Fig09 : breakdown + comm par N (R=max, T=max)
    T_h_max = max(all_T_h) if all_T_h else 1
    fig, ax = plt.subplots(figsize=(W, H))
    sorted_N = sorted({k[0] for k in hyb_data})
    labels = [str(n) for n in sorted_N]
    series = []
    for comp, field in [("tree","tree_us"),("force","force_us"),
                         ("integrate","integrate_us"),("comm","comm_us")]:
        vals = []
        for N in sorted_N:
            r = hyb_data.get((N, R_max, T_h_max))
            vals.append(r.get(field,0)/1e3 if r else 0)
        series.append((comp, vals))
    stacked_bars(ax, labels, series, C,
                 f"Hybride — Décomposition  (R={R_max}, T={T_h_max})")
    save(fig, f"{figdir}/fig09_hybrid_breakdown.png")

# ══════════════════════════════════════════════════════════════
#  Phase 4 : CUDA
# ══════════════════════════════════════════════════════════════

def analyse_cuda(cuda_rows, ref):
    if not cuda_rows: return {}, []
    hdr("Phase 4 — CUDA : décomposition et crossover")
    data = {}
    for r in cuda_rows:
        data[(r["n_particles"], r["n_threads_cpu"])] = r
    all_N = sorted({r["n_particles"]    for r in cuda_rows})
    all_T = sorted({r["n_threads_cpu"]  for r in cuda_rows})

    print(f"\n  {'N':>8}  {'T':>3}  {'total':>9}  {'tree':>8}  {'serial':>8}  "
          f"{'gpu':>8}  {'integ':>8}  {'speedup':>8}")
    sep()
    for N in all_N:
        for T in all_T:
            r = data.get((N, T))
            if not r: continue
            tot = r.get("total_us",1)
            seq = r.get("seq_us") or ref.get(N, 0)
            s = spd(seq, tot)
            print(f"  {N:>8}  {T:>3}  {fmt_us(tot):>9}  "
                  f"{fmt_us(r.get('tree_us')):>8}  "
                  f"{fmt_us(r.get('serial_us')):>8}  "
                  f"{fmt_us(r.get('gpu_us')):>8}  "
                  f"{fmt_us(r.get('integrate_us')):>8}  "
                  f"{s:>8}")
    return data, all_N

def figs_cuda(cuda_data, ref, hyb_data, all_N_cuda, figdir):
    if not HAS_MPL or not cuda_data: return

    T_cuda = min({k[1] for k in cuda_data})

    # Fig10 : breakdown absolu (barres empilées) vs N
    fig, ax = plt.subplots(figsize=(W, H))
    sorted_N = sorted(all_N_cuda)
    labels = [str(n) for n in sorted_N]
    series = []
    for comp, field in [("tree","tree_us"),("serial","serial_us"),
                         ("gpu","gpu_us"),("integrate","integrate_us")]:
        vals = [cuda_data.get((N, T_cuda),{}).get(field,0)/1e3 for N in sorted_N]
        series.append((comp, vals))
    stacked_bars(ax, labels, series, C,
                 f"CUDA — Décomposition absolue (tree/sérialisation/GPU/intég)")
    save(fig, f"{figdir}/fig10_cuda_breakdown_vs_N.png")

    # Fig11 : crossover — temps absolu CUDA vs séquentiel vs hybride best
    fig, ax = plt.subplots(figsize=(W, H))

    # Séquentiel
    xs_seq = sorted(r for r in ref if ref[r] > 0)
    ys_seq = [ref[n]/1e3 for n in xs_seq]
    if xs_seq:
        ax.plot(xs_seq, ys_seq, marker="x", color=C["seq"],
                label="Séquentiel (BH.cpp)", lw=2, ls="--")

    # CUDA
    xs_c, ys_c = [], []
    for N in sorted(all_N_cuda):
        r = cuda_data.get((N, T_cuda))
        if r:
            xs_c.append(N); ys_c.append(r["total_us"]/1e3)
    if xs_c:
        ax.plot(xs_c, ys_c, marker="*", color=C["cuda"],
                label=f"CUDA (T_cpu={T_cuda})", lw=2.5, markersize=9)

    # Hybride meilleur (R=max, T=max)
    if hyb_data:
        all_keys = list(hyb_data.keys())
        R_max = max(k[1] for k in all_keys)
        T_max = max(k[2] for k in all_keys)
        xs_h, ys_h = [], []
        for N in sorted({k[0] for k in all_keys}):
            r = hyb_data.get((N, R_max, T_max))
            if r:
                xs_h.append(N); ys_h.append(r["total_us"]/1e3)
        if xs_h:
            ax.plot(xs_h, ys_h, marker="P", color=C["hybrid"],
                    label=f"Hybride R={R_max}×T={T_max}", lw=2.5)

    # Annoter le crossover CUDA/séq s'il existe
    if xs_c and xs_seq:
        # Interpoler le point de croisement
        for i in range(len(xs_c)-1):
            N0, N1 = xs_c[i], xs_c[i+1]
            s0 = ref.get(N0, 0)/1e3; s1 = ref.get(N1, 0)/1e3
            c0, c1 = ys_c[i], ys_c[i+1]
            if s0 and s1 and ((c0-s0)*(c1-s1) < 0):
                # Croisement entre N0 et N1
                Ncross = int(N0 + (N1-N0) * (s0-c0) / ((c1-s1)-(c0-s0) + 1e-9))
                ax.axvline(Ncross, color="grey", ls=":", lw=1.5, alpha=0.7)
                ax.text(Ncross, ax.get_ylim()[1]*0.9 if ax.get_ylim()[1] > 0 else 1,
                        f"  crossover\n  N≈{Ncross:,}", fontsize=8, color="grey",
                        va="top")
                break

    ax.set_xscale("log")
    ax.set_xlabel("N particules"); ax.set_ylabel("Temps total (ms)")
    ax.set_title("CUDA vs séquentiel vs hybride — crossover GPU/CPU")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, f"{figdir}/fig11_cuda_crossover.png")

    # Fig12 : pourcentage de chaque phase CUDA vs N (stacked area)
    # Montre comment le GPU devient dominant à grand N
    x_vals = sorted(all_N_cuda)
    series_pct = {}
    for comp, field in [("tree","tree_us"),("serial","serial_us"),
                         ("gpu","gpu_us"),("integrate","integrate_us")]:
        series_pct[comp] = [cuda_data.get((N, T_cuda),{}).get(field,0) for N in x_vals]

    fig, ax = plt.subplots(figsize=(W, H))
    stacked_area_pct(ax, x_vals, series_pct, C,
                     f"CUDA — Part de chaque phase vs N\n"
                     f"(à grand N, GPU doit dominer ; tree+serial doivent reculer)")
    save(fig, f"{figdir}/fig12_cuda_breakdown_pct.png")

# ══════════════════════════════════════════════════════════════
#  Fig13 : Vue globale
# ══════════════════════════════════════════════════════════════

def fig_global(omp_data, mpi_data, hyb_data, cuda_data,
               ref, all_N, all_T_omp, all_R, all_N_cuda, figdir):
    if not HAS_MPL: return
    T_max = max(all_T_omp) if all_T_omp else 1
    R_max = max(all_R)     if all_R     else 1
    N_all = sorted(set(all_N) | set(all_N_cuda) | set(ref.keys()))

    fig, ax = plt.subplots(figsize=(W+1, H))

    # Séq
    xs = sorted(ref); ys = [1.0]*len(xs)
    ax.plot(xs, ys, marker="x", color=C["seq"], label="séquentiel (1x)",
            lw=1.5, ls="--", alpha=0.6)

    # OMP v2, T=max
    if omp_data:
        xs, ys = [], []
        for N in all_N:
            r = omp_data.get("v2",{}).get((N, T_max))
            if r and ref.get(N): xs.append(N); ys.append(ref[N]/r["total_us"])
        if xs: ax.plot(xs, ys, marker="s", color=C["v2"],
                       label=f"OMP dynamic32 T={T_max}", lw=2)

    # MPI v3, R=max
    if mpi_data:
        xs, ys = [], []
        for N in all_N:
            r = mpi_data.get("v3",{}).get((N, R_max, 1))
            if r and ref.get(N): xs.append(N); ys.append(ref[N]/r["total_us"])
        if xs: ax.plot(xs, ys, marker="^", color=C["mpi_v3"],
                       label=f"MPI v3 R={R_max}", lw=2)

    # Hybride R=max T=max
    if hyb_data:
        T_h = max(k[2] for k in hyb_data)
        xs, ys = [], []
        for N in sorted({k[0] for k in hyb_data}):
            r = hyb_data.get((N, R_max, T_h))
            if r and ref.get(N): xs.append(N); ys.append(ref[N]/r["total_us"])
        if xs: ax.plot(xs, ys, marker="P", color=C["hybrid"],
                       label=f"Hybride R={R_max}×T={T_h}", lw=2.5)

    # CUDA
    if cuda_data:
        T_c = min(k[1] for k in cuda_data)
        xs, ys = [], []
        for N in sorted(all_N_cuda):
            r = cuda_data.get((N, T_c))
            if r and ref.get(N): xs.append(N); ys.append(ref[N]/r["total_us"])
        if xs: ax.plot(xs, ys, marker="*", color=C["cuda"],
                       label="CUDA", lw=2.5, markersize=9)

    ax.set_xscale("log")
    ax.set_xlabel("N particules"); ax.set_ylabel("Speedup vs BH.cpp séquentiel")
    ax.set_title("Comparaison globale — meilleure config par phase\n"
                 "(ref = BH.cpp original)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, f"{figdir}/fig13_global_comparison.png")

# ══════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    rdir   = Path(args.results)
    figdir = str(rdir / "figures")

    # Charger tous les CSV
    seq_rows  = load(rdir / "seq_results.csv")
    # Fusionner avec seq_cuda_range si présent
    seq_rows += load(rdir / "seq_cuda_range.csv")
    omp_rows  = load(rdir / "omp_results.csv")
    mpi_rows  = load(rdir / "mpi_results.csv")
    hyb_rows  = load(rdir / "hybrid_results.csv")
    cuda_rows = load(rdir / "cuda_results.csv")

    if not seq_rows:
        print(f"[ERR] {rdir}/seq_results.csv introuvable.")
        print("      Lancez : make benchmark  (ou make benchmark_quick)")
        sys.exit(1)

    # Dédupliquer seq_rows sur (n_particles, iters) — garder dernier
    seen = {}
    for r in seq_rows:
        seen[r["n_particles"]] = r
    seq_rows_dedup = list(seen.values())

    ref = analyse_ref(seq_rows_dedup)

    omp_data, all_N_omp, all_T_omp = {}, [], []
    if omp_rows:
        omp_data, all_N_omp, all_T_omp = analyse_omp(omp_rows, ref)

    mpi_data, all_N_mpi, all_R, all_T_mpi = {}, [], [], []
    if mpi_rows:
        mpi_data, all_N_mpi, all_R, all_T_mpi = analyse_mpi(mpi_rows, ref)

    hyb_data = {}
    if hyb_rows:
        hyb_data, _, _, _ = analyse_hybrid(hyb_rows, ref)

    cuda_data, all_N_cuda = {}, []
    if cuda_rows:
        cuda_data, all_N_cuda = analyse_cuda(cuda_rows, ref)

    if HAS_MPL:
        os.makedirs(figdir, exist_ok=True)
        print(); sep()
        print("  Génération des figures...")
        sep()

        fig_ref_breakdown(seq_rows_dedup, figdir)
        if omp_data:
            figs_omp(omp_data, ref, all_N_omp, all_T_omp, figdir)
        if mpi_data:
            figs_mpi(mpi_data, ref, all_N_mpi, all_R, all_T_mpi, figdir)
        if hyb_data:
            figs_hybrid(hyb_data, ref, all_N_omp or all_N_mpi, all_R, all_T_mpi, figdir)
        if cuda_data:
            figs_cuda(cuda_data, ref, hyb_data, all_N_cuda, figdir)

        all_N_union = sorted(set(all_N_omp) | set(all_N_mpi))
        fig_global(omp_data, mpi_data, hyb_data, cuda_data,
                   ref, all_N_union, all_T_omp, all_R, all_N_cuda, figdir)

        print(f"\n  {figdir}/")
    print()

if __name__ == "__main__":
    main()