"""
analyse_qicas_16.py
===================
Read completed result JSONs from qicas_casscf_16systems.py and print
a summary table comparing QICAS vs autoCAS active spaces, CASCI energies,
and CASSCF convergence metrics.

Usage:
    python analyse_qicas_16.py --results_dir results/
"""

import os
import json
import glob
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()

    files = sorted(glob.glob(
        os.path.join(args.results_dir, 'qicas_casscf_*.json')))

    if not files:
        print(f"No result files found in {args.results_dir}/")
        return

    results = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            if os.path.getsize(f) > 0:
                results.append(d)
        except Exception as e:
            print(f"  Cannot read {f}: {e}")

    print(f"\nLoaded {len(results)} result files.\n")

    # ── Goal 1 table: CASCI comparison ────────────────────────────────
    print("=" * 110)
    print("GOAL 1 — Active space proposal: QICAS → CASCI")
    print("=" * 110)
    print(f"{'System':<40} {'Row':>4} {'2S':>3} "
          f"{'QICAS CAS':>10} {'autoCAS':>10} {'ΔNO':>5} "
          f"{'ΔCASCI(mHa)':>13} {'QICAS↓?':>8}")
    print("-" * 110)

    for d in results:
        if d['status'] != 'OK':
            print(f"  {d['name']:<38} ERROR: {d.get('error','?')[:50]}")
            continue
        name  = d['name']
        row   = d['metal_row']
        s2    = d['spin_2s']
        q     = d['qicas']
        g1    = d['goal1_casci']
        cmp   = d['comparison_with_autocas']
        cas_q = f"({q['n_active_e']},{q['n_active']})"
        cas_a = f"({cmp['autocas_ne']},{cmp['autocas_no']})"
        dno   = cmp['delta_no']
        delta = g1['delta_mha']
        better = "YES" if g1['qicas_better'] else "no"
        print(f"  {name:<38} {row:>4} {s2:>3} "
              f"{cas_q:>10} {cas_a:>10} {dno:>+5} "
              f"{delta:>+13.3f} {better:>8}")

    # ── Goal 2 table: CASSCF convergence ──────────────────────────────
    print()
    print("=" * 110)
    print("GOAL 2 — Warm-start efficiency: QICAS orbitals → CASSCF")
    print("=" * 110)
    print(f"{'System':<40} {'Row':>4} {'2S':>3} "
          f"{'CAS':>8} {'HF→itr':>7} {'QI→itr':>7} {'Δitr':>7} "
          f"{'Gap→CASSCF(mHa)':>17} {'<1.6mHa?':>9}")
    print("-" * 110)

    n_faster = 0; n_chem = 0; n_valid = 0

    for d in results:
        if d['status'] != 'OK':
            continue
        n_valid += 1
        name    = d['name']
        row     = d['metal_row']
        s2      = d['spin_2s']
        g2      = d['goal2_casscf']
        q       = d['qicas']
        cas_str = f"({q['n_active_e']},{q['n_active']})"
        hf_it   = g2['from_hf']['n_macro_iter']
        qi_it   = g2['from_qicas']['n_macro_iter']
        sp      = g2['iter_speedup']
        gap     = g2.get('gap_casci_qicas_to_casscf_mha', float('nan'))
        acc     = g2.get('within_chemical_accuracy', False)
        if sp > 0: n_faster += 1
        if acc:    n_chem   += 1
        acc_str = "YES ✓" if acc else "no"
        print(f"  {name:<38} {row:>4} {s2:>3} "
              f"{cas_str:>8} {hf_it:>7} {qi_it:>7} {sp:>+7} "
              f"{gap:>+17.3f} {acc_str:>9}")

    print()
    print(f"  Summary ({n_valid} completed systems):")
    print(f"    QICAS orbitals faster: {n_faster}/{n_valid}")
    print(f"    Chemical accuracy:     {n_chem}/{n_valid}")

    # ── FQI reduction summary ──────────────────────────────────────────
    print()
    print("=" * 80)
    print("F_QI reduction (QICAS orbital rotation effectiveness)")
    print("=" * 80)
    print(f"{'System':<40} {'FQI_init':>9} {'FQI_final':>10} {'Reduction':>10}")
    print("-" * 80)
    for d in results:
        if d['status'] != 'OK':
            continue
        q = d['qicas']
        print(f"  {d['name']:<38} {q['fqi_initial']:>9.4f} "
              f"{q['fqi_final']:>10.4f} {q['fqi_reduction']:>10.4f}")

    print()
    print("Done.")


if __name__ == '__main__':
    main()
