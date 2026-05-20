#!/usr/bin/env python3
"""
analyse_results.py — Lit results/summary.csv et affiche un tableau
de speedup + courbe d'efficacité parallèle.

Usage :
    python3 scripts/analyse_results.py
    python3 scripts/analyse_results.py --csv results/summary.csv
"""

import argparse
import csv
import sys
from collections import defaultdict

# ── Lecture ──────────────────────────────────────────────────

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "variant":  r["variant"],
                "n":        int(r["n_particles"]),
                "ranks":    int(r["n_ranks"]),
                "threads":  int(r["n_threads"]),
                "total_us": int(r["total_us"]),
                "speedup":  float(r["speedup"]),
            })
    return rows

# ── Affichage ────────────────────────────────────────────────

def header(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_table(rows, cols, fmt):
    col_w = [max(len(c), 10) for c in cols]
    hdr = "  ".join(c.ljust(w) for c, w in zip(cols, col_w))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = "  ".join(str(fmt[c](r)).ljust(w) for c, w in zip(cols, col_w))
        print(line)

# ── Analyse speedup OpenMP ───────────────────────────────────

def omp_analysis(rows):
    header("OpenMP — Speedup vs séquentiel (1 rang / 1 thread)")
    omp = [r for r in rows if r["variant"] == "omp"]
    if not omp:
        print("  Aucune donnée OMP."); return
    omp_sorted = sorted(omp, key=lambda r: (r["n"], r["threads"]))
    print_table(omp_sorted,
        ["n", "threads", "total_us", "speedup", "efficiency"],
        {
            "n":         lambda r: r["n"],
            "threads":   lambda r: r["threads"],
            "total_us":  lambda r: r["total_us"],
            "speedup":   lambda r: f"{r['speedup']:.3f}x",
            "efficiency":lambda r: f"{r['speedup']/r['threads']*100:.1f}%",
        }
    )

# ── Analyse speedup MPI ─────────────────────────────────────

def mpi_analysis(rows):
    header("MPI — Speedup vs séquentiel (1 rang / 1 thread)")
    mpi = [r for r in rows if r["variant"] == "mpi"]
    if not mpi:
        print("  Aucune donnée MPI."); return
    mpi_sorted = sorted(mpi, key=lambda r: (r["n"], r["ranks"]))
    print_table(mpi_sorted,
        ["n", "ranks", "total_us", "speedup", "efficiency"],
        {
            "n":         lambda r: r["n"],
            "ranks":     lambda r: r["ranks"],
            "total_us":  lambda r: r["total_us"],
            "speedup":   lambda r: f"{r['speedup']:.3f}x",
            "efficiency":lambda r: f"{r['speedup']/r['ranks']*100:.1f}%",
        }
    )

# ── Analyse hybride ─────────────────────────────────────────

def hybrid_analysis(rows):
    header("Hybride (OpenMP + MPI) — Speedup vs séquentiel")
    hyb = [r for r in rows if r["variant"] == "hybrid"]
    if not hyb:
        print("  Aucune donnée hybride."); return
    hyb_sorted = sorted(hyb, key=lambda r: (r["n"], r["ranks"], r["threads"]))
    print_table(hyb_sorted,
        ["n", "ranks", "threads", "procs_total", "total_us", "speedup", "efficiency"],
        {
            "n":          lambda r: r["n"],
            "ranks":      lambda r: r["ranks"],
            "threads":    lambda r: r["threads"],
            "procs_total":lambda r: r["ranks"] * r["threads"],
            "total_us":   lambda r: r["total_us"],
            "speedup":    lambda r: f"{r['speedup']:.3f}x",
            "efficiency": lambda r: f"{r['speedup']/(r['ranks']*r['threads'])*100:.1f}%",
        }
    )

# ── Best config ─────────────────────────────────────────────

def best_config(rows):
    header("Meilleure configuration par taille de problème")
    by_n = defaultdict(list)
    for r in rows:
        if r["variant"] != "seq":
            by_n[r["n"]].append(r)
    for n in sorted(by_n):
        best = max(by_n[n], key=lambda r: r["speedup"])
        ptot = best["ranks"] * best["threads"]
        print(f"  N={n:>7}  →  {best['variant']:8}  "
              f"ranks={best['ranks']}  threads={best['threads']}  "
              f"procs={ptot}  speedup={best['speedup']:.3f}x  "
              f"({best['total_us']} us)")

# ── main ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/summary.csv")
    args = ap.parse_args()
    try:
        rows = load(args.csv)
    except FileNotFoundError:
        print(f"[erreur] Fichier non trouvé : {args.csv}")
        sys.exit(1)

    print(f"Fichier analysé : {args.csv}  ({len(rows)} lignes)")
    omp_analysis(rows)
    mpi_analysis(rows)
    hybrid_analysis(rows)
    best_config(rows)
    print()

if __name__ == "__main__":
    main()
