#!/usr/bin/env python3
"""
analyse.py — Analyse des résultats de benchmark Barnes-Hut parallèle
Lit les CSV produits par benchmark.sh et génère :
  - Tableaux de speedup en console
  - 6 figures PNG dans results/figures/

Usage :
    python3 scripts/analyse.py [--results results/]
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")          # pas d'écran nécessaire
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib non disponible — tableaux seulement", file=sys.stderr)

# ── Palette cohérente ────────────────────────────────────────
COLORS = {
    "v1":   "#4C72B0",
    "v2":   "#DD8452",
    "v3":   "#55A868",
    "v3s":  "#C44E52",
    "seq":  "#8C8C8C",
}
MARKERS = {"v1": "o", "v2": "s", "v3": "^", "v3s": "D", "seq": "x"}
VERSION_LABELS = {
    "v1":  "OMP v1 — static",
    "v2":  "OMP v2 — dynamic(32)",
    "v3":  "OMP v3 — guided(16)",
    "v3s": "OMP v3s — guided+sort",
}
MPI_LABELS = {
    "v1": "MPI v1 — bandes Y + Allgather 7f",
    "v2": "MPI v2 — blocs idx + Allgather 7f",
    "v3": "MPI v3 — bandes Y + Allgather 3f",
}

# ── Lecture CSV ───────────────────────────────────────────────

def load_csv(path):
    rows = []
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                row = {}
                for k, v in r.items():
                    try:    row[k] = int(v)
                    except (ValueError, TypeError):
                        try:    row[k] = float(v)
                        except (ValueError, TypeError):
                            row[k] = v
                rows.append(row)
    except FileNotFoundError:
        pass
    return rows

# ── Helpers console ───────────────────────────────────────────

def sep(n=70): print("─" * n)

def header(title):
    print()
    sep()
    print(f"  {title}")
    sep()

def fmt_us(v):
    if v >= 1_000_000: return f"{v/1e6:.2f} s"
    if v >= 1_000:     return f"{v/1e3:.1f} ms"
    return f"{v} µs"

# ── Phase 1 : analyse OMP ────────────────────────────────────

def analyse_omp(omp_rows, seq_rows):
    header("Phase 1 — OpenMP : speedup vs séquentiel")

    # Référence seq par N
    seq_ref = {r["n_particles"]: r["total_us"] for r in seq_rows}

    # Grouper par (version, N, T)
    data = defaultdict(dict)  # data[version][(N,T)] = total_us
    for r in omp_rows:
        data[r["version"]][(r["n_particles"], r["n_threads"])] = r["total_us"]

    # Trier les N et T
    all_N = sorted({r["n_particles"] for r in omp_rows})
    all_T = sorted({r["n_threads"]   for r in omp_rows})

    for N in all_N:
        ref = seq_ref.get(N, None)
        print(f"\n  N = {N:>7} particules  (séq = {fmt_us(ref) if ref else '?'})")
        print(f"  {'Version':<14}", end="")
        for T in all_T:
            print(f"  T={T:<3}", end="")
        print()
        for ver in sorted(data.keys()):
            print(f"  {VERSION_LABELS.get(ver,ver):<14}", end="")
            for T in all_T:
                t = data[ver].get((N,T))
                if t is None:
                    print(f"  {'—':>6}", end="")
                else:
                    spd = ref/t if ref else 0
                    print(f"  {spd:>5.2f}x", end="")
            print()

    # Efficacité
    header("Phase 1 — Efficacité parallèle OMP  (speedup / nthreads × 100%)")
    for N in all_N:
        ref = seq_ref.get(N)
        if not ref: continue
        print(f"\n  N = {N:>7}")
        print(f"  {'Version':<14}", end="")
        for T in all_T:
            print(f"  T={T:<3}", end="")
        print()
        for ver in sorted(data.keys()):
            print(f"  {VERSION_LABELS.get(ver,ver):<14}", end="")
            for T in all_T:
                t = data[ver].get((N,T))
                if t is None:
                    print(f"  {'—':>6}", end="")
                else:
                    eff = (ref/t)/T*100 if T>0 else 0
                    print(f"  {eff:>5.1f}%", end="")
            print()

    return data, seq_ref, all_N, all_T

# ── Phase 2 : analyse MPI ────────────────────────────────────

def analyse_mpi(mpi_rows, seq_ref):
    header("Phase 2 — MPI : speedup vs séquentiel")

    # Grouper par (version, N, R, T)
    data = defaultdict(dict)
    for r in mpi_rows:
        data[r["version"]][(r["n_particles"], r["n_ranks"], r["n_threads"])] = r["total_us"]

    all_N = sorted({r["n_particles"] for r in mpi_rows})
    all_R = sorted({r["n_ranks"]     for r in mpi_rows})
    all_T = sorted({r["n_threads"]   for r in mpi_rows})

    for N in all_N:
        ref = seq_ref.get(N)
        print(f"\n  N = {N:>7}  (séq = {fmt_us(ref) if ref else '?'})")
        for T in all_T:
            print(f"\n    OMP threads = {T}")
            print(f"    {'Version':<14}", end="")
            for R in all_R:
                print(f"  R={R:<3}", end="")
            print()
            for ver in sorted(data.keys()):
                print(f"    {MPI_LABELS.get(ver,ver):<14}", end="")
                for R in all_R:
                    t = data[ver].get((N,R,T))
                    if t is None:
                        print(f"  {'—':>6}", end="")
                    else:
                        spd = ref/t if ref else 0
                        print(f"  {spd:>5.2f}x", end="")
                print()

    return data, all_N, all_R, all_T

# ── Figures ───────────────────────────────────────────────────

def save_figures(omp_data, mpi_data, seq_ref, all_N, all_T, all_R, figdir):
    if not HAS_MPL:
        print("[INFO] matplotlib absent — figures non générées")
        return
    os.makedirs(figdir, exist_ok=True)

    def savefig(fig, name):
        p = os.path.join(figdir, name)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [fig] {p}")

    # ── Fig 1 : OMP speedup vs N (T fixé au max) ──────────────
    T_max = max(all_T) if all_T else 1
    fig, ax = plt.subplots(figsize=(8,5))
    for ver in sorted(omp_data.keys()):
        xs, ys = [], []
        for N in all_N:
            t = omp_data[ver].get((N, T_max))
            ref = seq_ref.get(N)
            if t and ref:
                xs.append(N); ys.append(ref/t)
        if xs:
            ax.plot(xs, ys, marker=MARKERS.get(ver,"o"),
                    color=COLORS.get(ver,"k"),
                    label=VERSION_LABELS.get(ver, ver), linewidth=2)
    ax.axhline(T_max, color="grey", linestyle="--", linewidth=1, label=f"Idéal ({T_max}x)")
    ax.set_xscale("log"); ax.set_xlabel("N particules"); ax.set_ylabel("Speedup")
    ax.set_title(f"OpenMP — Speedup vs N  (T = {T_max} threads)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    savefig(fig, "fig1_omp_speedup_vs_N.png")

    # ── Fig 2 : OMP speedup vs T (N fixé au max) ──────────────
    N_max = max(all_N) if all_N else 1
    fig, ax = plt.subplots(figsize=(7,5))
    for ver in sorted(omp_data.keys()):
        xs, ys = [], []
        for T in all_T:
            t = omp_data[ver].get((N_max, T))
            ref = seq_ref.get(N_max)
            if t and ref:
                xs.append(T); ys.append(ref/t)
        if xs:
            ax.plot(xs, ys, marker=MARKERS.get(ver,"o"),
                    color=COLORS.get(ver,"k"),
                    label=VERSION_LABELS.get(ver, ver), linewidth=2)
    ax.plot(all_T, all_T, "k--", linewidth=1, label="Idéal linéaire")
    ax.set_xlabel("Nb threads OMP"); ax.set_ylabel("Speedup")
    ax.set_title(f"OpenMP — Speedup vs nb threads  (N = {N_max})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    savefig(fig, "fig2_omp_speedup_vs_T.png")

    # ── Fig 3 : OMP efficacité vs T ───────────────────────────
    fig, ax = plt.subplots(figsize=(7,5))
    for ver in sorted(omp_data.keys()):
        xs, ys = [], []
        for T in all_T:
            t = omp_data[ver].get((N_max, T))
            ref = seq_ref.get(N_max)
            if t and ref and T>0:
                xs.append(T); ys.append((ref/t)/T*100)
        if xs:
            ax.plot(xs, ys, marker=MARKERS.get(ver,"o"),
                    color=COLORS.get(ver,"k"),
                    label=VERSION_LABELS.get(ver, ver), linewidth=2)
    ax.axhline(100, color="grey", linestyle="--", linewidth=1, label="Idéal 100%")
    ax.set_ylim(0, 110)
    ax.set_xlabel("Nb threads OMP"); ax.set_ylabel("Efficacité (%)")
    ax.set_title(f"OpenMP — Efficacité parallèle  (N = {N_max})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    savefig(fig, "fig3_omp_efficiency.png")

    # ── Fig 4 : OMP décomposition du temps (bar chart) ────────
    # Comparer tree / force / integrate pour chaque version à T_max, N_max
    # On doit relire les CSV pour avoir les sous-timings
    # (déjà dans omp_data... non, on n'a gardé que total_us)
    # → on skip cette fig si les données ne sont pas disponibles
    # (les sous-timings sont dans le CSV mais pas dans omp_data ici)
    # On relit depuis le CSV directement.

    # ── Fig 5 : MPI speedup vs N rangs (N fixé, T=1) ──────────
    T_mpi = 1
    fig, ax = plt.subplots(figsize=(8,5))
    for ver in sorted(mpi_data.keys()):
        xs, ys = [], []
        for R in all_R:
            t = mpi_data[ver].get((N_max, R, T_mpi))
            ref = seq_ref.get(N_max)
            if t and ref:
                xs.append(R); ys.append(ref/t)
        if xs:
            ax.plot(xs, ys, marker=MARKERS.get(ver,"o"),
                    color=COLORS.get(ver,"k"),
                    label=MPI_LABELS.get(ver, ver), linewidth=2)
    if all_R:
        ax.plot(all_R, all_R, "k--", linewidth=1, label="Idéal linéaire")
    ax.set_xlabel("Nb rangs MPI"); ax.set_ylabel("Speedup vs séquentiel")
    ax.set_title(f"MPI — Speedup vs nb rangs  (N = {N_max}, OMP T={T_mpi})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    savefig(fig, "fig4_mpi_speedup_vs_ranks.png")

    # ── Fig 6 : MPI+OMP (hybride) speedup vs procs totaux ─────
    fig, ax = plt.subplots(figsize=(8,5))
    for ver in sorted(mpi_data.keys()):
        xs, ys = [], []
        for R in all_R:
            for T in all_T:
                t = mpi_data[ver].get((N_max, R, T))
                ref = seq_ref.get(N_max)
                if t and ref:
                    xs.append(R*T); ys.append(ref/t)
        if xs:
            # Trier par procs total
            pts = sorted(zip(xs,ys))
            xs2, ys2 = zip(*pts)
            ax.plot(xs2, ys2, marker=MARKERS.get(ver,"o"),
                    color=COLORS.get(ver,"k"),
                    label=MPI_LABELS.get(ver, ver), linewidth=2)
    procs = sorted({R*T for R in all_R for T in all_T})
    if procs:
        ax.plot(procs, procs, "k--", linewidth=1, label="Idéal linéaire")
    ax.set_xlabel("Nb procs totaux (rangs × threads)")
    ax.set_ylabel("Speedup vs séquentiel")
    ax.set_title(f"Hybride MPI+OMP — Speedup  (N = {N_max})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    savefig(fig, "fig5_hybrid_speedup.png")

    # ── Fig 6 : comparaison MPI comm overhead (tree_us %) ─────
    # Relit le CSV MPI pour décomposer tree/force/integrate
    pass  # rempli plus bas après rechargement des données complètes

# ── Fig décomposition du temps OMP ────────────────────────────

def fig_omp_breakdown(omp_rows, seq_rows, N_target, T_target, figdir):
    if not HAS_MPL: return
    versions = sorted({r["version"] for r in omp_rows})
    data = {}
    for r in omp_rows:
        if r["n_particles"]==N_target and r["n_threads"]==T_target:
            data[r["version"]] = {
                "tree": r.get("tree_us",0),
                "force": r.get("force_us",0),
                "integrate": r.get("integrate_us",0),
            }
    # Ajouter séq
    for r in seq_rows:
        if r["n_particles"]==N_target:
            data["seq"] = {"tree": r.get("tree_us",0),
                           "force": r.get("force_us",0),
                           "integrate": r.get("integrate_us",0)}
    if len(data) < 2: return

    labels = list(data.keys())
    tree_v    = [data[v]["tree"]      for v in labels]
    force_v   = [data[v]["force"]     for v in labels]
    integ_v   = [data[v]["integrate"] for v in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9,5))
    w = 0.5
    b1 = ax.bar(x, tree_v,  w, label="Construction arbre", color="#4C72B0")
    b2 = ax.bar(x, force_v, w, bottom=tree_v, label="Calcul de forces", color="#DD8452")
    b3 = ax.bar(x, integ_v, w,
                bottom=[t+f for t,f in zip(tree_v, force_v)],
                label="Intégration", color="#55A868")

    nice_labels = [VERSION_LABELS.get(v, v) for v in labels]
    ax.set_xticks(x); ax.set_xticklabels(nice_labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Temps cumulé (µs)")
    ax.set_title(f"Décomposition du temps — N={N_target}, T={T_target} threads")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    p = os.path.join(figdir, "fig6_omp_time_breakdown.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {p}")

def fig_mpi_breakdown(mpi_rows, N_target, R_target, T_target, figdir):
    if not HAS_MPL: return
    versions = sorted({r["version"] for r in mpi_rows})
    data = {}
    for r in mpi_rows:
        if (r["n_particles"]==N_target and
            r["n_ranks"]==R_target and
            r["n_threads"]==T_target):
            data[r["version"]] = {
                "tree":      r.get("tree_us",0),
                "force":     r.get("force_us",0),
                "integrate": r.get("integrate_us",0),
            }
    if len(data) < 1: return

    labels = list(data.keys())
    tree_v  = [data[v]["tree"]      for v in labels]
    force_v = [data[v]["force"]     for v in labels]
    integ_v = [data[v]["integrate"] for v in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8,5))
    w = 0.5
    ax.bar(x, tree_v,  w, label="Construction arbre", color="#4C72B0")
    ax.bar(x, force_v, w, bottom=tree_v, label="Calcul de forces", color="#DD8452")
    ax.bar(x, integ_v, w,
           bottom=[t+f for t,f in zip(tree_v,force_v)],
           label="Intégration", color="#55A868")
    nice = [MPI_LABELS.get(v,v) for v in labels]
    ax.set_xticks(x); ax.set_xticklabels(nice, rotation=10, ha="right", fontsize=8)
    ax.set_ylabel("Temps cumulé (µs)")
    ax.set_title(f"Décomposition MPI — N={N_target}, R={R_target}, T={T_target}")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    p = os.path.join(figdir, "fig7_mpi_time_breakdown.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {p}")

# ── main ─────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    rdir = Path(args.results)
    figdir = str(rdir / "figures")

    seq_rows = load_csv(rdir / "seq_results.csv")
    omp_rows = load_csv(rdir / "omp_results.csv")
    mpi_rows = load_csv(rdir / "mpi_results.csv")

    if not seq_rows:
        print(f"[ERR] Aucun résultat séquentiel dans {rdir}/seq_results.csv")
        print("      Lancez d'abord : make benchmark  (ou make benchmark_quick)")
        sys.exit(1)

    # Console
    omp_data, seq_ref, all_N, all_T = analyse_omp(omp_rows, seq_rows)

    all_R = []
    mpi_data = {}
    if mpi_rows:
        mpi_data, all_N_mpi, all_R, all_T_mpi = analyse_mpi(mpi_rows, seq_ref)

    # Figures
    if HAS_MPL:
        print()
        sep()
        print("  Génération des figures...")
        sep()
        save_figures(omp_data, mpi_data, seq_ref, all_N, all_T, all_R, figdir)

        N_target = max(all_N) if all_N else 10000
        T_target = max(all_T) if all_T else 2
        fig_omp_breakdown(omp_rows, seq_rows, N_target, T_target, figdir)

        if mpi_rows:
            R_target = max(all_R) if all_R else 2
            fig_mpi_breakdown(mpi_rows, N_target, R_target, T_target, figdir)

        print(f"\n  Figures sauvegardées dans : {figdir}/")
    print()

if __name__ == "__main__":
    main()
