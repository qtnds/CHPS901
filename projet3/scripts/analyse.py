#!/usr/bin/env python3
"""
analyse.py — Analyse complète des benchmarks Barnes-Hut parallèle
Génère tableaux console + figures PNG dans results/figures/

Usage :
    python3 scripts/analyse.py [--results results/]
"""

import argparse, csv, os, sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib absent — tableaux seulement", file=sys.stderr)

# ── Palette & styles ──────────────────────────────────────────
C = {
    "seq":   "#555555",
    "v1":    "#4C72B0", "v2": "#DD8452", "v3": "#55A868", "v3s": "#C44E52",
    "mpi_v1":"#4C72B0", "mpi_v2":"#DD8452", "mpi_v3":"#55A868",
    "hybrid":"#8172B3",
    "cuda":  "#C44E52",
}
MK = {"seq":"x","v1":"o","v2":"s","v3":"^","v3s":"D",
      "mpi_v1":"o","mpi_v2":"s","mpi_v3":"^","hybrid":"P","cuda":"*"}
OMP_LABELS  = {"v1":"static","v2":"dynamic(32)","v3":"guided(16)","v3s":"guided+sort"}
MPI_LABELS  = {"v1":"bandes-Y 7f","v2":"blocs-idx 7f","v3":"bandes-Y 3f"}
HYBRID_LABEL = "hybrid (MPI 3f + OMP dyn32)"
CUDA_LABEL   = "CUDA (arbre CPU + kernel GPU)"

FIGSIZE_WIDE = (9, 5)
FIGSIZE_SQ   = (7, 5)
DPI = 150

# ── Utilitaires ───────────────────────────────────────────────

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

def sep(n=68): print("─"*n)
def hdr(t):    print(); sep(); print(f"  {t}"); sep()

def fmt_us(v):
    if v is None: return "—"
    if v >= 1_000_000: return f"{v/1e6:.2f}s"
    if v >= 1_000:     return f"{v/1e3:.1f}ms"
    return f"{int(v)}µs"

def spd(ref, t):
    if not ref or not t or t==0: return "—"
    return f"{ref/t:.2f}x"

def eff(ref, t, p):
    if not ref or not t or not p or t==0 or p==0: return "—"
    return f"{ref/t/p*100:.0f}%"

def savefig(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")

def median_of(rows, key):
    vals = [r[key] for r in rows if key in r and r[key]]
    if not vals: return None
    return sorted(vals)[len(vals)//2]

# ── Séquentiel ────────────────────────────────────────────────

def analyse_seq(seq_rows):
    hdr("Référence séquentielle")
    print(f"  {'N':>8}  {'total':>9}  {'tree':>9}  {'force':>9}  {'integr':>9}")
    sep()
    ref = {}
    for r in sorted(seq_rows, key=lambda x: x["n_particles"]):
        N = r["n_particles"]
        ref[N] = r["total_us"]
        print(f"  {N:>8}  {fmt_us(r['total_us']):>9}  "
              f"{fmt_us(r.get('tree_us')):>9}  "
              f"{fmt_us(r.get('force_us')):>9}  "
              f"{fmt_us(r.get('integrate_us')):>9}")
    return ref

# ── Phase 1 : OMP ─────────────────────────────────────────────

def analyse_omp(omp_rows, ref):
    hdr("Phase 1 — OpenMP : speedup vs séquentiel")
    data  = defaultdict(dict)   # data[ver][(N,T)] = row
    for r in omp_rows:
        data[r["version"]][(r["n_particles"], r["n_threads"])] = r

    all_N = sorted({r["n_particles"] for r in omp_rows})
    all_T = sorted({r["n_threads"]   for r in omp_rows})

    for N in all_N:
        print(f"\n  N={N:>7}  (séq={fmt_us(ref.get(N))})")
        hdr_row = f"  {'version':<16}" + "".join(f"  T={T:<5}" for T in all_T)
        print(hdr_row)
        for ver in sorted(data.keys()):
            row = f"  {OMP_LABELS.get(ver,ver):<16}"
            for T in all_T:
                r = data[ver].get((N,T))
                row += f"  {spd(ref.get(N), r['total_us'] if r else None):>7}"
            print(row)

    hdr("Phase 1 — Efficacité (speedup/threads × 100%)")
    for N in all_N:
        print(f"\n  N={N:>7}")
        hdr_row = f"  {'version':<16}" + "".join(f"  T={T:<5}" for T in all_T)
        print(hdr_row)
        for ver in sorted(data.keys()):
            row = f"  {OMP_LABELS.get(ver,ver):<16}"
            for T in all_T:
                r = data[ver].get((N,T))
                row += f"  {eff(ref.get(N), r['total_us'] if r else None, T):>7}"
            print(row)
    return data, all_N, all_T

# ── Phase 2 : MPI ─────────────────────────────────────────────

def analyse_mpi(mpi_rows, ref):
    hdr("Phase 2 — MPI × OMP : speedup vs séquentiel")
    data = defaultdict(dict)
    for r in mpi_rows:
        data[r["version"]][(r["n_particles"], r["n_ranks"], r["n_threads"])] = r

    all_N = sorted({r["n_particles"] for r in mpi_rows})
    all_R = sorted({r["n_ranks"]     for r in mpi_rows})
    all_T = sorted({r["n_threads"]   for r in mpi_rows})

    for N in all_N:
        for T in all_T:
            print(f"\n  N={N:>7}  T_omp={T}  (séq={fmt_us(ref.get(N))})")
            hdr_row = f"  {'version':<18}" + "".join(f"  R={R:<4}" for R in all_R)
            print(hdr_row)
            for ver in sorted(data.keys()):
                row = f"  {MPI_LABELS.get(ver,ver):<18}"
                for R in all_R:
                    r = data[ver].get((N,R,T))
                    row += f"  {spd(ref.get(N), r['total_us'] if r else None):>6}"
                print(row)
    return data, all_N, all_R, all_T

# ── Phase 3 : Hybride ─────────────────────────────────────────

def analyse_hybrid(hyb_rows, ref):
    if not hyb_rows: return {}, [], [], []
    hdr("Phase 3 — Hybride final : speedup vs séquentiel")
    data = {}
    for r in hyb_rows:
        data[(r["n_particles"], r["n_ranks"], r["n_threads"])] = r

    all_N = sorted({r["n_particles"] for r in hyb_rows})
    all_R = sorted({r["n_ranks"]     for r in hyb_rows})
    all_T = sorted({r["n_threads"]   for r in hyb_rows})

    for N in all_N:
        print(f"\n  N={N:>7}  (séq={fmt_us(ref.get(N))})")
        hdr_row = f"  {'R\\T':<6}" + "".join(f"  T={T:<5}" for T in all_T)
        print(hdr_row)
        for R in all_R:
            row = f"  R={R:<4}"
            for T in all_T:
                r = data.get((N,R,T))
                row += f"  {spd(ref.get(N), r['total_us'] if r else None):>7}"
            print(row)

        print(f"  {'comm_us breakdown (R=max):'}")
        R_max = max(all_R)
        for T in all_T:
            r = data.get((N, R_max, T))
            if r:
                tot = r.get("total_us",1)
                comm = r.get("comm_us",0)
                pct = comm/tot*100 if tot else 0
                print(f"    T={T}: comm={fmt_us(comm)} ({pct:.1f}% du total)")
    return data, all_N, all_R, all_T

# ── Phase 4 : CUDA ────────────────────────────────────────────

def analyse_cuda(cuda_rows, ref):
    if not cuda_rows: return {}, []
    hdr("Phase 4 — CUDA : speedup vs séquentiel")
    data = {}
    for r in cuda_rows:
        data[(r["n_particles"], r["n_threads_cpu"])] = r

    all_N = sorted({r["n_particles"]    for r in cuda_rows})
    all_T = sorted({r["n_threads_cpu"]  for r in cuda_rows})

    print(f"  {'N':>8}  {'T_cpu':>6}  {'total':>9}  {'tree':>9}  {'serial':>9}  "
          f"{'gpu':>9}  {'integr':>9}  {'speedup':>8}")
    sep()
    for N in all_N:
        for T in all_T:
            r = data.get((N,T))
            if not r: continue
            tot = r.get("total_us",1)
            print(f"  {N:>8}  {T:>6}  {fmt_us(tot):>9}  "
                  f"{fmt_us(r.get('tree_us')):>9}  "
                  f"{fmt_us(r.get('serial_us')):>9}  "
                  f"{fmt_us(r.get('gpu_us')):>9}  "
                  f"{fmt_us(r.get('integrate_us')):>9}  "
                  f"{spd(ref.get(N), tot):>8}")
    return data, all_N

# ═══════════════════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════════════════

def make_figures(omp_data, mpi_data, hyb_data, cuda_data,
                 ref, all_N, all_T_omp, all_R, all_T_mpi,
                 cuda_all_N, figdir):
    if not HAS_MPL:
        return
    os.makedirs(figdir, exist_ok=True)
    T_max = max(all_T_omp) if all_T_omp else 1
    N_max = max(all_N)     if all_N     else 1
    R_max = max(all_R)     if all_R     else 1

    # ── Fig 1 : OMP speedup vs N ──────────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
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
    ax.set_title(f"OMP — Speedup vs N  (T={T_max} threads)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    savefig(fig, f"{figdir}/fig1_omp_speedup_vs_N.png")

    # ── Fig 2 : OMP speedup vs threads ────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    for ver in sorted(omp_data):
        xs, ys = [], []
        for T in all_T_omp:
            r = omp_data[ver].get((N_max, T))
            if r and ref.get(N_max):
                xs.append(T); ys.append(ref[N_max]/r["total_us"])
        if xs:
            ax.plot(xs, ys, marker=MK.get(ver,"o"), color=C.get(ver,"k"),
                    label=OMP_LABELS.get(ver,ver), lw=2)
    ax.plot(all_T_omp, all_T_omp, "k--", lw=1, label="idéal")
    ax.set_xlabel("Threads OMP"); ax.set_ylabel("Speedup")
    ax.set_title(f"OMP — Speedup vs T  (N={N_max})"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    savefig(fig, f"{figdir}/fig2_omp_speedup_vs_T.png")

    # ── Fig 3 : OMP efficacité ────────────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    for ver in sorted(omp_data):
        xs, ys = [], []
        for T in all_T_omp:
            r = omp_data[ver].get((N_max, T))
            if r and ref.get(N_max) and T>0:
                xs.append(T); ys.append(ref[N_max]/r["total_us"]/T*100)
        if xs:
            ax.plot(xs, ys, marker=MK.get(ver,"o"), color=C.get(ver,"k"),
                    label=OMP_LABELS.get(ver,ver), lw=2)
    ax.axhline(100, color="grey", ls="--", lw=1, label="idéal 100%")
    ax.set_ylim(0,115); ax.set_xlabel("Threads OMP"); ax.set_ylabel("Efficacité (%)")
    ax.set_title(f"OMP — Efficacité  (N={N_max})"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    savefig(fig, f"{figdir}/fig3_omp_efficiency.png")

    # ── Fig 4 : OMP décomposition du temps (barres empilées) ──
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    versions = sorted(omp_data.keys())
    T_bar = T_max
    labels, trees, forces, ints = [], [], [], []
    for ver in versions:
        r = omp_data[ver].get((N_max, T_bar))
        if r:
            labels.append(OMP_LABELS.get(ver, ver))
            trees.append(r.get("tree_us",0)/1e3)
            forces.append(r.get("force_us",0)/1e3)
            ints.append(r.get("integrate_us",0)/1e3)
    x = np.arange(len(labels))
    w = .5
    ax.bar(x, trees,  w, label="Construction arbre", color="#4C72B0")
    ax.bar(x, forces, w, bottom=trees,               label="Calcul forces",    color="#DD8452")
    ax.bar(x, ints,   w, bottom=[t+f for t,f in zip(trees,forces)],
           label="Intégration", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=9)
    ax.set_ylabel("Temps (ms)"); ax.set_title(f"OMP — Décomposition  (N={N_max}, T={T_bar})")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    savefig(fig, f"{figdir}/fig4_omp_breakdown.png")

    # ── Fig 5 : MPI speedup vs rangs ──────────────────────────
    T_mpi = all_T_mpi[0] if all_T_mpi else 1
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
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
    ax.set_xlabel("Rangs MPI"); ax.set_ylabel("Speedup vs séquentiel")
    ax.set_title(f"MPI — Speedup vs rangs  (N={N_max}, T_omp={T_mpi})")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    savefig(fig, f"{figdir}/fig5_mpi_speedup_vs_ranks.png")

    # ── Fig 6 : MPI décomposition du temps ────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    vlabels, vtrees, vforces, vints = [], [], [], []
    for ver in sorted(mpi_data):
        r = mpi_data[ver].get((N_max, R_max, T_mpi))
        if r:
            vlabels.append(MPI_LABELS.get(ver,ver))
            vtrees.append(r.get("tree_us",0)/1e3)
            vforces.append(r.get("force_us",0)/1e3)
            vints.append(r.get("integrate_us",0)/1e3)
    x = np.arange(len(vlabels)); w=.5
    ax.bar(x, vtrees,  w, label="Arbre",       color="#4C72B0")
    ax.bar(x, vforces, w, bottom=vtrees,        label="Forces",    color="#DD8452")
    ax.bar(x, vints,   w, bottom=[t+f for t,f in zip(vtrees,vforces)],
           label="Intégration", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(vlabels, rotation=10, ha="right", fontsize=9)
    ax.set_ylabel("Temps (ms)")
    ax.set_title(f"MPI — Décomposition  (N={N_max}, R={R_max}, T={T_mpi})")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    savefig(fig, f"{figdir}/fig6_mpi_breakdown.png")

    # ── Fig 7 : Hybride — speedup vs procs totaux ─────────────
    if hyb_data:
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        all_T_hyb = sorted({k[2] for k in hyb_data})
        for T in all_T_hyb:
            xs, ys = [], []
            for R in all_R:
                r = hyb_data.get((N_max, R, T))
                if r and ref.get(N_max):
                    xs.append(R*T); ys.append(ref[N_max]/r["total_us"])
            if xs:
                pts = sorted(zip(xs,ys))
                xx,yy = zip(*pts)
                ax.plot(xx, yy, marker="P", color=C["hybrid"],
                        label=f"hybrid T={T}", lw=2)
        # Superposer la meilleure version MPI pour comparaison
        best_mpi = "v3"
        if best_mpi in mpi_data:
            xs, ys = [], []
            for R in all_R:
                r = mpi_data[best_mpi].get((N_max, R, T_mpi))
                if r and ref.get(N_max):
                    xs.append(R); ys.append(ref[N_max]/r["total_us"])
            if xs:
                ax.plot(xs, ys, marker=MK["v3"], color=C["mpi_v3"],
                        ls="--", label=f"MPI-v3 (T=1)", lw=1.5, alpha=.7)
        procs = sorted({k[1]*k[2] for k in hyb_data})
        if procs: ax.plot(procs, procs, "k--", lw=1, label="idéal")
        ax.set_xlabel("Procs totaux (rangs × threads)")
        ax.set_ylabel("Speedup vs séquentiel")
        ax.set_title(f"Hybride — Speedup  (N={N_max})")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        savefig(fig, f"{figdir}/fig7_hybrid_speedup.png")

    # ── Fig 8 : Hybride — décomposition avec comm_us ──────────
    if hyb_data:
        R_bar = max(all_R) if all_R else 1
        T_bar2 = max({k[2] for k in hyb_data}) if hyb_data else 1
        fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
        # N sur l'axe X, barres empilées
        sorted_N = sorted({k[0] for k in hyb_data})
        tr_l, fo_l, it_l, co_l = [], [], [], []
        for N in sorted_N:
            r = hyb_data.get((N, R_bar, T_bar2))
            if r:
                tr_l.append(r.get("tree_us",0)/1e3)
                fo_l.append(r.get("force_us",0)/1e3)
                it_l.append(r.get("integrate_us",0)/1e3)
                co_l.append(r.get("comm_us",0)/1e3)
            else:
                tr_l.append(0); fo_l.append(0); it_l.append(0); co_l.append(0)
        x = np.arange(len(sorted_N)); w=.5
        ax.bar(x, tr_l, w, label="Arbre",        color="#4C72B0")
        ax.bar(x, fo_l, w, bottom=tr_l,           label="Forces",     color="#DD8452")
        b2 = [t+f for t,f in zip(tr_l,fo_l)]
        ax.bar(x, it_l, w, bottom=b2,             label="Intégration",color="#55A868")
        b3 = [a+b for a,b in zip(b2,it_l)]
        ax.bar(x, co_l, w, bottom=b3,             label="Comm MPI",   color="#E377C2")
        ax.set_xticks(x); ax.set_xticklabels([str(n) for n in sorted_N], fontsize=9)
        ax.set_xlabel("N particules"); ax.set_ylabel("Temps (ms)")
        ax.set_title(f"Hybride — Décomposition  (R={R_bar}, T={T_bar2})")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
        savefig(fig, f"{figdir}/fig8_hybrid_breakdown.png")

    # ── Fig 9 : CUDA décomposition ────────────────────────────
    if cuda_data:
        cuda_N = sorted({k[0] for k in cuda_data})
        T_cuda = min({k[1] for k in cuda_data})
        tr_l, se_l, gu_l, it_l = [], [], [], []
        for N in cuda_N:
            r = cuda_data.get((N, T_cuda))
            if r:
                tr_l.append(r.get("tree_us",0)/1e3)
                se_l.append(r.get("serial_us",0)/1e3)
                gu_l.append(r.get("gpu_us",0)/1e3)
                it_l.append(r.get("integrate_us",0)/1e3)
            else:
                tr_l.append(0); se_l.append(0); gu_l.append(0); it_l.append(0)
        x = np.arange(len(cuda_N)); w=.5
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.bar(x, tr_l, w, label="Arbre CPU",      color="#4C72B0")
        ax.bar(x, se_l, w, bottom=tr_l,             label="Sérialis.+H2D",color="#9ECAE1")
        b1 = [t+s for t,s in zip(tr_l,se_l)]
        ax.bar(x, gu_l, w, bottom=b1,               label="Kernel GPU+D2H",color="#C44E52")
        b2 = [a+b for a,b in zip(b1,gu_l)]
        ax.bar(x, it_l, w, bottom=b2,               label="Intégration CPU",color="#55A868")
        ax.set_xticks(x); ax.set_xticklabels([str(n) for n in cuda_N], fontsize=9)
        ax.set_xlabel("N particules"); ax.set_ylabel("Temps (ms)")
        ax.set_title(f"CUDA — Décomposition  (T_cpu={T_cuda})")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
        savefig(fig, f"{figdir}/fig9_cuda_breakdown.png")

    # ── Fig 10 : Vue globale — meilleur de chaque phase ───────
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Séq
    if ref:
        xs = sorted(ref.keys()); ys = [ref[n]/ref[n] for n in xs]
        ax.plot(xs, ys, marker=MK["seq"], color=C["seq"],
                label="séquentiel (1x)", lw=1.5, ls="--")

    # Meilleur OMP : v2, T=max
    if omp_data and all_T_omp:
        xs, ys = [], []
        for N in all_N:
            r = omp_data.get("v2",{}).get((N, T_max))
            if r and ref.get(N):
                xs.append(N); ys.append(ref[N]/r["total_us"])
        if xs:
            ax.plot(xs, ys, marker=MK["v2"], color=C["v2"],
                    label=f"OMP v2 T={T_max}", lw=2)

    # Meilleur MPI : v3, R=max, T=1
    if mpi_data and all_R:
        xs, ys = [], []
        for N in all_N:
            r = mpi_data.get("v3",{}).get((N, R_max, 1))
            if r and ref.get(N):
                xs.append(N); ys.append(ref[N]/r["total_us"])
        if xs:
            ax.plot(xs, ys, marker=MK["v3"], color=C["mpi_v3"],
                    label=f"MPI v3 R={R_max}", lw=2)

    # Hybride : R=max, T=max
    if hyb_data:
        all_T_h = sorted({k[2] for k in hyb_data})
        T_h_max = max(all_T_h) if all_T_h else 1
        xs, ys = [], []
        for N in all_N:
            r = hyb_data.get((N, R_max, T_h_max))
            if r and ref.get(N):
                xs.append(N); ys.append(ref[N]/r["total_us"])
        if xs:
            ax.plot(xs, ys, marker="P", color=C["hybrid"],
                    label=f"Hybrid R={R_max}×T={T_h_max}", lw=2.5)

    # CUDA
    if cuda_data:
        xs, ys = [], []
        for N in sorted(cuda_all_N):
            r = cuda_data.get((N, 1))
            if r and ref.get(N):
                xs.append(N); ys.append(ref[N]/r["total_us"])
        if xs:
            ax.plot(xs, ys, marker="*", color=C["cuda"],
                    label="CUDA", lw=2.5, markersize=10)

    ax.set_xscale("log")
    ax.set_xlabel("N particules"); ax.set_ylabel("Speedup vs séquentiel")
    ax.set_title("Comparaison globale — meilleure config par phase")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    savefig(fig, f"{figdir}/fig10_global_comparison.png")

# ── main ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    rdir    = Path(args.results)
    figdir  = str(rdir / "figures")

    seq_rows  = load(rdir/"seq_results.csv")
    omp_rows  = load(rdir/"omp_results.csv")
    mpi_rows  = load(rdir/"mpi_results.csv")
    hyb_rows  = load(rdir/"hybrid_results.csv")
    cuda_rows = load(rdir/"cuda_results.csv")

    if not seq_rows:
        print(f"[ERR] {rdir}/seq_results.csv introuvable.")
        print("      Lancez : make benchmark  (ou make benchmark_quick)")
        sys.exit(1)

    ref = analyse_seq(seq_rows)
    omp_data, all_N, all_T_omp = {}, [], []
    if omp_rows:
        omp_data, all_N, all_T_omp = analyse_omp(omp_rows, ref)

    mpi_data, all_R, all_T_mpi = {}, [], []
    if mpi_rows:
        mpi_data, _, all_R, all_T_mpi = analyse_mpi(mpi_rows, ref)

    hyb_data = {}
    if hyb_rows:
        hyb_data, _, _, _ = analyse_hybrid(hyb_rows, ref)

    cuda_data, cuda_all_N = {}, []
    if cuda_rows:
        cuda_data, cuda_all_N = analyse_cuda(cuda_rows, ref)

    if HAS_MPL:
        print(); sep()
        print("  Génération des figures...")
        sep()
        make_figures(omp_data, mpi_data, hyb_data, cuda_data,
                     ref, all_N, all_T_omp, all_R, all_T_mpi,
                     cuda_all_N, figdir)
        print(f"\n  → {figdir}/")
    print()

if __name__ == "__main__":
    main()
