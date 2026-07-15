"""
check_ecp.py  (fixed)
=====================
Verifies that def2-svp ECP loads correctly for 4d metals in PySCF.

Two fixes vs previous version:
  1. ECP specified per-atom as a dict — only applied to the metal, not Cl.
  2. Spin states chosen to be consistent with ECP electron counts.

Run:
    python check_ecp.py
"""

from pyscf import gto

# All-electron Z values
Z_AE = {'Mo': 42, 'Rh': 45, 'Ru': 44, 'Pd': 46}
ECP_CORE = 28   # electrons removed by def2-svp ECP for 4d metals

# Test geometry: metal + 1 Cl at 2.3 Ang, charge=-1
# Spin chosen so (n_e_with_ecp - spin_2s) % 2 == 0
# Mo: 32 e → spin 0 or 2; Rh: 35 e → spin 1; Ru: 34 e → spin 0; Pd: 36 e → spin 0
TEST = {
    'Mo': {'spin_2s': 2},
    'Rh': {'spin_2s': 1},
    'Ru': {'spin_2s': 0},
    'Pd': {'spin_2s': 0},
}

print("=" * 70)
print("ECP verification for def2-svp on 4d metals")
print("ECP applied per-atom (metal only) — Cl left all-electron")
print("=" * 70)
print()

all_pass = True

for metal, cfg in TEST.items():
    s2   = cfg['spin_2s']
    z_ae = Z_AE[metal]
    chg  = -1

    expected_ae  = z_ae + 17 - abs(chg)          # all-electron MeCl^-1
    expected_ecp = expected_ae - ECP_CORE         # after removing core

    # ── WITH ECP (per-atom dict — only metal gets ECP) ─────────────
    mol_ecp = gto.Mole()
    mol_ecp.atom    = f'{metal} 0 0 0; Cl 2.3 0 0'
    mol_ecp.charge  = chg
    mol_ecp.spin    = s2
    mol_ecp.basis   = 'def2-svp'
    mol_ecp.ecp     = {metal: 'def2-svp'}   # <-- metal ONLY
    mol_ecp.unit    = 'angstrom'
    mol_ecp.verbose = 0
    mol_ecp.build()

    # ── WITHOUT ECP (all-electron reference) ───────────────────────
    # Use spin_2s=0 to avoid parity issues with all-electron count
    s2_ae = 0 if expected_ae % 2 == 0 else 1
    mol_ae = gto.Mole()
    mol_ae.atom    = f'{metal} 0 0 0; Cl 2.3 0 0'
    mol_ae.charge  = chg
    mol_ae.spin    = s2_ae
    mol_ae.basis   = 'def2-svp'
    mol_ae.unit    = 'angstrom'
    mol_ae.verbose = 0
    mol_ae.build()

    n_e_ecp = mol_ecp.nelectron
    n_e_ae  = mol_ae.nelectron
    n_removed = n_e_ae - n_e_ecp

    ecp_loaded  = bool(mol_ecp._ecp)
    count_ok    = (n_e_ecp == expected_ecp)
    removed_ok  = (n_removed == ECP_CORE)
    pass_       = ecp_loaded and count_ok and removed_ok

    if not pass_:
        all_pass = False

    status = "PASS" if pass_ else "FAIL"
    print(f"  {metal}:  {status}")
    print(f"    ECP dict loaded:          {ecp_loaded}  "
          f"({'OK' if ecp_loaded else 'FAIL — ECP not found'})")
    print(f"    All-electron count:       {n_e_ae}  (expected {expected_ae})")
    print(f"    With-ECP count:           {n_e_ecp}  (expected {expected_ecp})")
    print(f"    Core electrons removed:   {n_removed}  "
          f"(expected {ECP_CORE}{'  OK' if removed_ok else '  WRONG'})")
    print(f"    Basis functions (w/ ECP): {mol_ecp.nao_nr()}")
    print()

print("=" * 70)
if all_pass:
    print("All 4d metals: PASS — ECP loading confirmed.")
    print("Safe to run qicas_casscf_16systems.py.")
else:
    print("FAILURES detected — do NOT submit until resolved.")
    print()
    print("If ECP still fails, try the string without hyphen:")
    print("  mol.ecp = {metal: 'def2svp'}")
print("=" * 70)
