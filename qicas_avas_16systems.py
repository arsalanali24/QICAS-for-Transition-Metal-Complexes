"""
qicas_avas_16systems.py
=======================
AVAS-QICAS pipeline for 16 transition-metal benchmark systems.

Key difference from qicas_casscf_16systems.py:
    CANONICAL HF SCRIPT:  UHF → frontier window → DMRG → QICAS
    THIS SCRIPT:          RHF/UHF → AVAS orbitals → DMRG → QICAS

Why AVAS:
    Canonical HF orbitals are delocalized over the whole molecule.
    For low-spin systems this gives a diffuse entropy profile with no
    clear plateau → QICAS selects the wrong (too large) active space.
    AVAS pre-rotates HF orbitals to maximise metal d-character, giving
    a sharp entropy plateau even for singlets → QICAS selects correctly.

Reference choices:
    spin_2s = 0 (singlet): RHF → AVAS → QICAS
    spin_2s > 0 (open-shell): UHF → AVAS → QICAS

Results go to results_avas/ (never mixed with results/ from canonical run).

Usage:
    python qicas_avas_16systems.py --system CSD_NiCl4_2m_sqpl_spin0
    python qicas_avas_16systems.py --system_index $SLURM_ARRAY_TASK_ID
    python qicas_avas_16systems.py --system_index 0 --M 100 --sweeps 30
"""

import os
import sys
import json
import time
import argparse
import traceback
import numpy as np

from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas as pyscf_avas
from pyscf import dmrgscf
from pyscf.dmrgscf import dmrgci
from scipy.linalg import expm as scipy_expm
from scipy.optimize import minimize as scipy_minimize

# ── Geometry builders ─────────────────────────────────────────────────────

import math

def _tet(d):
    c = d / math.sqrt(3)
    return [(c,c,c),(-c,-c,c),(-c,c,-c),(c,-c,-c)]

def _oct(d):
    return [(d,0,0),(-d,0,0),(0,d,0),(0,-d,0),(0,0,d),(0,0,-d)]

def _sqpl(d):
    return [(d,0,0),(-d,0,0),(0,d,0),(0,-d,0)]

GEOMETRY_BUILDERS = {'tet':_tet,'oct':_oct,'sqpl':_sqpl,'sq_pl':_sqpl}

DIST_TABLE = {
    ('Cr','Cl','tet'):2.24,('Cr','Cl','oct'):2.34,
    ('Mn','Cl','tet'):2.35,('Mn','Cl','oct'):2.48,
    ('Mn','Br','tet'):2.50,('Mn','F','oct'): 1.98,
    ('Fe','Cl','tet'):2.19,('Fe','Cl','oct'):2.38,
    ('Co','Cl','oct'):2.44,
    ('Ni','Cl','sqpl'):2.20,('Ni','Cl','oct'):2.40,
}

# d-orbital AO labels for AVAS — metal-specific
METAL_D_LABELS = {
    'Cr':'Cr 3d','Mn':'Mn 3d','Fe':'Fe 3d',
    'Co':'Co 3d','Ni':'Ni 3d',
    'Mo':'Mo 4d','Rh':'Rh 4d','Ru':'Ru 4d','Pd':'Pd 4d',
}

# ── System definitions ────────────────────────────────────────────────────

SYSTEMS = {
    "CSD_CrCl4_2m_tet_spin4": dict(
        metal='Cr',ligand='Cl',n_ligands=4,charge=-2,spin_2s=4,
        geometry='tet',dist_ang=None,metal_row='3d',
        autocas_ne=12,autocas_no=8),
    "CSD_MnCl4_2m_tet_spin5": dict(
        metal='Mn',ligand='Cl',n_ligands=4,charge=-2,spin_2s=5,
        geometry='tet',dist_ang=None,metal_row='3d',
        autocas_ne=17,autocas_no=10),
    "CSD_MnBr4_2m_tet_spin5": dict(
        metal='Mn',ligand='Br',n_ligands=4,charge=-2,spin_2s=5,
        geometry='tet',dist_ang=None,metal_row='3d',
        autocas_ne=17,autocas_no=10),
    "CSD_MnF6_4m_oct_spin5": dict(
        metal='Mn',ligand='F', n_ligands=6,charge=-4,spin_2s=5,
        geometry='oct',dist_ang=None,metal_row='3d',
        autocas_ne=15,autocas_no=8),
    "CSD_CrCl4_2m_tet_spin2": dict(
        metal='Cr',ligand='Cl',n_ligands=4,charge=-2,spin_2s=2,
        geometry='tet',dist_ang=None,metal_row='3d',
        autocas_ne=6,autocas_no=7),
    "CSD_FeCl4_2m_tet_spin0": dict(
        metal='Fe',ligand='Cl',n_ligands=4,charge=-2,spin_2s=0,
        geometry='tet',dist_ang=None,metal_row='3d',
        autocas_ne=4,autocas_no=4),
    "CSD_NiCl4_2m_sqpl_spin0": dict(
        metal='Ni',ligand='Cl',n_ligands=4,charge=-2,spin_2s=0,
        geometry='sqpl',dist_ang=None,metal_row='3d',
        autocas_ne=6,autocas_no=6),
    "CSD_NiCl6_4m_oct_spin2": dict(
        metal='Ni',ligand='Cl',n_ligands=6,charge=-4,spin_2s=2,
        geometry='oct',dist_ang=None,metal_row='3d',
        autocas_ne=12,autocas_no=6),
    "CSD_CoCl6_4m_oct_spin1": dict(
        metal='Co',ligand='Cl',n_ligands=6,charge=-4,spin_2s=1,
        geometry='oct',dist_ang=None,metal_row='3d',
        autocas_ne=3,autocas_no=2),
    "Mo_Cl6_chg-3_spin3_oct_d2p299": dict(
        metal='Mo',ligand='Cl',n_ligands=6,charge=-3,spin_2s=3,
        geometry='oct',dist_ang=2.299,metal_row='4d',
        autocas_ne=25,autocas_no=20),
    "Mo_Cl6_chg-3_spin1_oct_d2p299": dict(
        metal='Mo',ligand='Cl',n_ligands=6,charge=-3,spin_2s=1,
        geometry='oct',dist_ang=2.299,metal_row='4d',
        autocas_ne=7,autocas_no=7),
    "Rh_Cl6_chg-3_spin0_oct_d2p32": dict(
        metal='Rh',ligand='Cl',n_ligands=6,charge=-3,spin_2s=0,
        geometry='oct',dist_ang=2.320,metal_row='4d',
        autocas_ne=2,autocas_no=4),
    "Ru_Cl6_chg-3_spin1_oct_d2p232": dict(
        metal='Ru',ligand='Cl',n_ligands=6,charge=-3,spin_2s=1,
        geometry='oct',dist_ang=2.232,metal_row='4d',
        autocas_ne=11,autocas_no=12),
    "Pd_Cl4_chg-2_spin0_sq_pl_d2p3": dict(
        metal='Pd',ligand='Cl',n_ligands=4,charge=-2,spin_2s=0,
        geometry='sq_pl',dist_ang=2.300,metal_row='4d',
        autocas_ne=22,autocas_no=16),
    "Rh_Cl6_chg-3_spin2_oct_d2p32": dict(
        metal='Rh',ligand='Cl',n_ligands=6,charge=-3,spin_2s=2,
        geometry='oct',dist_ang=2.320,metal_row='4d',
        autocas_ne=8,autocas_no=10),
    "Pd_Cl6_chg-2_spin0_oct_d2p3": dict(
        metal='Pd',ligand='Cl',n_ligands=6,charge=-2,spin_2s=0,
        geometry='oct',dist_ang=2.300,metal_row='4d',
        autocas_ne=4,autocas_no=5),
}

ALL_SYSTEM_NAMES = list(SYSTEMS.keys())


# ── Molecule builder ──────────────────────────────────────────────────────

def build_mol(name, s):
    dist = s['dist_ang'] or DIST_TABLE.get(
        (s['metal'],s['ligand'],s['geometry']),2.30)
    builder = GEOMETRY_BUILDERS[s['geometry']]
    lig_coords = builder(dist)[:s['n_ligands']]
    atom_list  = [(s['metal'],(0.,0.,0.))]
    for xyz in lig_coords:
        atom_list.append((s['ligand'],xyz))
    mol = gto.Mole()
    mol.atom     = atom_list
    mol.charge   = s['charge']
    mol.spin     = s['spin_2s']
    mol.basis    = 'def2-svp'
    mol.unit     = 'angstrom'
    mol.symmetry = False
    if s['metal_row'] == '4d':
        mol.ecp = {s['metal']: 'def2-svp'}
    mol.verbose = 4
    mol.build()
    return mol


# ── HF reference ─────────────────────────────────────────────────────────

def run_hf(mol, s):
    """
    RHF for singlets (spin_2s=0), UHF for open-shell.
    RHF gives cleaner canonical orbitals for AVAS on singlets.
    """
    if s['spin_2s'] == 0:
        mf = scf.RHF(mol)
        print("  [HF] Using RHF (singlet system)")
    else:
        mf = scf.UHF(mol)
        print("  [HF] Using UHF (open-shell system)")

    mf.max_cycle  = 500
    mf.conv_tol   = 1e-10
    mf.init_guess = 'atom'
    mf.level_shift = 0.2
    mf.kernel()

    if not mf.converged:
        print("  HF retry 1: stronger level shift...")
        mf.level_shift = 0.5
        mf.kernel()
    if not mf.converged:
        print("  HF retry 2: adding damping...")
        mf.level_shift = 0.3
        mf.damp = 0.5
        mf.kernel()
    if not mf.converged:
        raise RuntimeError("HF did not converge after 3 attempts")
    return mf


# ── AVAS orbital construction ─────────────────────────────────────────────

def run_avas(mol, mf, s, threshold=0.1):
    """
    Build AVAS orbitals targeting the metal d-shell.

    AVAS rotates canonical HF MOs to maximise metal d-character,
    giving a compact set of d-like orbitals with clear entropy separation
    from the ligand-dominated inactive orbitals.

    Parameters
    ----------
    threshold : float
        Minimum AO projection weight to include an MO in the active space.
        0.1 = moderate (recommended), 0.2 = strict, 0.05 = loose.

    Returns
    -------
    mo_avas   : np.array, full MO coefficient matrix reordered by AVAS
    ncas_avas : number of AVAS active orbitals (the d-shell set)
    nelec_avas: number of active electrons from AVAS
    """
    ao_labels = [METAL_D_LABELS[s['metal']]]
    print(f"  [AVAS] target AO labels: {ao_labels}")
    print(f"  [AVAS] projection threshold: {threshold}")

    # For UHF, AVAS works on the alpha MO set
    if isinstance(mf, scf.uhf.UHF):
        mo_coeff = mf.mo_coeff[0]   # alpha MOs
        mo_occ   = mf.mo_occ[0]
        # Create a temporary RHF-like object for AVAS
        mf_rhf = mf.to_rhf()
    else:
        mo_coeff = mf.mo_coeff
        mo_occ   = mf.mo_occ
        mf_rhf   = mf

    try:
        ncas_avas, nelec_avas, mo_avas = pyscf_avas.avas(
            mf_rhf, ao_labels,
            threshold=threshold,
            minao='minao',
            with_iao=False,
        )
        print(f"  [AVAS] active space from projection: "
              f"CAS({nelec_avas},{ncas_avas})")
        return mo_avas, ncas_avas, nelec_avas

    except Exception as e:
        print(f"  [AVAS] failed ({e}), falling back to canonical MOs")
        # Fallback: use canonical HF orbitals with frontier window
        if isinstance(mf, scf.uhf.UHF):
            return mf.mo_coeff[0], None, None
        else:
            return mf.mo_coeff, None, None


# ── DMRG on AVAS window ───────────────────────────────────────────────────

def run_dmrg_avas(mol, mf, mo_avas, ncas_avas, nelec_avas,
                   M=100, n_sweeps=30, scratch_dir=None, name='system'):
    """
    Run low-m DMRGCI on the AVAS active space.

    Unlike the canonical script which uses a frontier window,
    here the active space IS the AVAS set — no additional window selection.
    The DMRG captures correlation within the d-shell + bonding partners.
    """
    os.makedirs(scratch_dir or f'/tmp/qicas_avas_{name}', exist_ok=True)
    sd = scratch_dir or f'/tmp/qicas_avas_{name}'

    # Parity check
    if (nelec_avas - mol.spin) % 2 != 0:
        nelec_avas += 1
    if (nelec_avas - mol.spin) % 2 != 0:
        nelec_avas -= 2
    nelec_avas = min(nelec_avas, 2 * ncas_avas)
    nelec_avas = max(nelec_avas, mol.spin)

    print(f"  [DMRG] CAS({nelec_avas},{ncas_avas}), M={M}, sweeps={n_sweeps}")

    mc = mcscf.CASSCF(mf.to_rhf() if isinstance(mf,scf.uhf.UHF) else mf,
                      ncas_avas, nelec_avas)
    mc.fcisolver = dmrgci.DMRGCI(mol, maxM=M, tol=1e-8)
    mc.fcisolver.scratchDirectory = sd
    mc.fcisolver.runtimeDir       = sd
    mc.fcisolver.maxIter          = n_sweeps
    mc.fcisolver.block_extra_keyword = ['num_thrds 8']
    mc.max_cycle_macro = 1   # CASCI only — no orbital optimisation
    e_dmrg = mc.kernel(mo_avas)[0]

    return mc, e_dmrg


# ── Entropy functions (same as canonical script) ──────────────────────────

def _orbital_entropy_from_lam(n_i, G_ii):
    lam = np.array([1.0-2*n_i+G_ii, n_i-G_ii, n_i-G_ii, G_ii])
    lam = np.clip(lam, 1e-14, 1.0)
    lam /= lam.sum()
    return -float(np.sum(lam * np.log(lam)))


def get_rdms(mc):
    try:
        dm1, dm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
        print("  [RDM] via make_rdm12()")
        return dm1, dm2
    except Exception as e:
        print(f"  [RDM] make_rdm12 failed ({e}), using MF approximation")
        dm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
        n = mc.ncas
        dm2 = np.zeros((n,n,n,n))
        for i in range(n):
            dm2[i,i,i,i] = (dm1[i,i]/2.0)**2 * 4.0
        return dm1, dm2


def entropies_from_rdms(gamma, Gamma, n):
    ent = np.zeros(n)
    for i in range(n):
        ent[i] = _orbital_entropy_from_lam(gamma[i,i]/2.0, Gamma[i,i,i,i]/4.0)
    return ent


def entropy_plateau_cas_size(entropies, spin_2s,
                              entropy_floor=0.05, min_active=2):
    """
    Largest absolute entropy gap (Stein-Reiher plateau method).
    With AVAS orbitals the plateau is sharper — this method works better.
    """
    sig_idx   = np.where(entropies > entropy_floor)[0]
    n_sig     = len(sig_idx)
    if n_sig <= max(spin_2s, min_active):
        return max(spin_2s, min_active)
    sig_sorted = np.sort(entropies[sig_idx])[::-1]
    gaps       = sig_sorted[:-1] - sig_sorted[1:]
    d_cas      = int(np.argmax(gaps)) + 1
    return int(np.clip(max(d_cas, spin_2s, min_active), 2, n_sig))


# ── QICAS orbital rotation (same algorithm as canonical script) ───────────

def qicas_orbital_rotation(gamma, Gamma, n_win, d_cas,
                            max_iter=300, tol=1e-7):
    """
    Minimize F_QI = sum_{i in N} S(rho_i) by orbital rotation.
    Non-active set N = bottom (n_win - d_cas) by initial entropy (fixed).
    """
    n = n_win
    ent_0      = entropies_from_rdms(gamma, Gamma, n)
    sorted_idx = np.argsort(ent_0)[::-1]
    nonact_idx = sorted_idx[d_cas:].tolist()
    fqi_0      = float(ent_0[nonact_idx].sum())
    print(f"  [QICAS rotation] D_CAS={d_cas}, |N|={len(nonact_idx)}, "
          f"F_QI(i)={fqi_0:.6f}")

    def _fqi(x_flat):
        X = x_flat.reshape(n,n); X = (X-X.T)/2
        U = scipy_expm(X)
        g = U @ gamma @ U.T
        val = 0.0
        for i in nonact_idx:
            u_i  = U[:,i]
            G_ii = np.einsum('p,q,r,s,pqrs->', u_i,u_i,u_i,u_i,Gamma)/4.0
            val += _orbital_entropy_from_lam(g[i,i]/2.0, G_ii)
        return val

    res = scipy_minimize(_fqi, np.zeros(n*n), method='L-BFGS-B',
                         options={'maxiter':max_iter,'ftol':tol,
                                  'gtol':tol*0.1,'maxfun':max_iter*20})

    X_opt = res.x.reshape(n,n); X_opt = (X_opt-X_opt.T)/2
    U_opt = scipy_expm(X_opt)
    g_opt = U_opt @ gamma @ U_opt.T
    ent_opt = np.zeros(n)
    for i in range(n):
        u_i     = U_opt[:,i]
        G_ii    = np.einsum('p,q,r,s,pqrs->', u_i,u_i,u_i,u_i,Gamma)/4.0
        ent_opt[i] = _orbital_entropy_from_lam(g_opt[i,i]/2.0, G_ii)

    fqi_f = float(ent_opt[nonact_idx].sum())
    print(f"  [QICAS rotation] F_QI(f)={fqi_f:.6f}, "
          f"reduction={fqi_0-fqi_f:.6f}, ok={res.success}, nit={res.nit}")
    return U_opt, ent_opt, g_opt, fqi_0, fqi_f


def determine_active_space_from_qicas(ent_qicas, gamma_qicas,
                                       spin_2s, d_cas, mf, ncas_avas):
    """
    Classify AVAS+QICAS orbitals by entropy plateau.
    Active electrons counted from HF occupations of AVAS orbitals.
    d_cas is fixed from before the rotation — no second plateau evaluation.
    """
    n_win      = len(ent_qicas)
    sorted_idx = np.argsort(ent_qicas)[::-1]
    active_rel = sorted(sorted_idx[:d_cas].tolist())

    # Electron count: use the AVAS nelec as reference (more reliable)
    # since AVAS electron count is based on chemical projection
    n_active_e = ncas_avas   # from AVAS projection

    # Parity check
    if (n_active_e - spin_2s) % 2 != 0:
        n_active_e += 1
    if (n_active_e - spin_2s) % 2 != 0:
        n_active_e -= 2

    # Sanity: cannot exceed 2 electrons per orbital
    n_active_e = min(n_active_e, 2 * d_cas)
    n_active_e = max(n_active_e, spin_2s)
    if (n_active_e - spin_2s) % 2 != 0:
        n_active_e -= 1

    return active_rel, d_cas, n_active_e


# ── CASCI and CASSCF ─────────────────────────────────────────────────────

def run_casci_comparison(mol, mf, mo_hf, mo_qicas, n_active_e, n_active):
    """CASCI at both HF-AVAS and QICAS-rotated orbitals."""
    def _e(mo):
        mc = mcscf.CASCI(mf.to_rhf() if isinstance(mf,scf.uhf.UHF) else mf,
                         n_active, n_active_e)
        mc.verbose = 3
        return float(mc.kernel(mo)[0])
    e_hf    = _e(mo_hf)
    e_qicas = _e(mo_qicas)
    delta   = (e_qicas - e_hf) * 1000.0
    return e_hf, e_qicas, delta


def run_casscf_comparison(mol, mf, mo_hf, mo_qicas,
                           n_active_e, n_active, max_macro=100):
    """CASSCF from both AVAS-HF and AVAS-QICAS starting orbitals."""
    results = {}
    mf_ref = mf.to_rhf() if isinstance(mf, scf.uhf.UHF) else mf

    for label, mo_start in [('from_hf_avas', mo_hf),
                             ('from_qicas_avas', mo_qicas)]:
        t0 = time.time()
        mc = mcscf.CASSCF(mf_ref, n_active, n_active_e)
        mc.max_cycle_macro = max_macro
        mc.conv_tol        = 1e-9
        mc.conv_tol_grad   = 1e-4
        mc.verbose         = 4

        count = [0]
        def _cb(envs): count[0] += 1
        mc.callback = _cb

        e,_,_,_,_ = mc.kernel(mo_start)
        results[label] = {
            'e_casscf':     float(e),
            'converged':    bool(mc.converged),
            'n_macro_iter': count[0],
            't_s':          time.time() - t0,
        }
    return results


# ── Main per-system runner ────────────────────────────────────────────────

def run_one_system(name, s, M_dmrg=100, n_sweeps=30,
                   avas_threshold=0.1, out_dir='results_avas'):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'qicas_avas_{name}.json')

    result = {
        'name':            name,
        'metal':           s['metal'],
        'ligand':          s['ligand'],
        'charge':          s['charge'],
        'spin_2s':         s['spin_2s'],
        'metal_row':       s['metal_row'],
        'M_dmrg':          M_dmrg,
        'avas_threshold':  avas_threshold,
        'pipeline':        'AVAS-QICAS',
        'autocas_reference': {
            'cas_ne': s['autocas_ne'],
            'cas_no': s['autocas_no'],
            'note':   'autoCAS M=150, stored for comparison only'
        },
        'status': 'RUNNING',
    }

    t_total = time.time()

    try:
        print(f"\n{'='*65}")
        print(f"  {name}")
        print(f"  PIPELINE: AVAS-QICAS")
        print(f"  {s['metal']}/{s['ligand']}  charge={s['charge']}  "
              f"2S={s['spin_2s']}  row={s['metal_row']}")
        print(f"{'='*65}")

        # Step 1: Build molecule
        mol = build_mol(name, s)
        result['n_electrons'] = int(mol.nelectron)
        result['n_basis']     = mol.nao_nr()

        # Step 2: HF reference (RHF for singlets, UHF for open-shell)
        print("\n[Step 2] HF reference...")
        t0 = time.time()
        mf = run_hf(mol, s)
        result['hf_type'] = 'RHF' if s['spin_2s'] == 0 else 'UHF'
        result['e_hf']    = float(mf.e_tot)
        result['t_hf_s']  = time.time() - t0
        print(f"  E({result['hf_type']}) = {mf.e_tot:.10f} Ha  "
              f"({result['t_hf_s']:.1f} s)")

        # Step 3: AVAS orbital construction
        print(f"\n[Step 3] AVAS (threshold={avas_threshold})...")
        t0 = time.time()
        mo_avas, ncas_avas, nelec_avas = run_avas(mol, mf, s, avas_threshold)
        t_avas = time.time() - t0

        if ncas_avas is None:
            raise RuntimeError("AVAS failed and fallback not available")

        result['avas'] = {
            'ncas':       ncas_avas,
            'nelec':      nelec_avas,
            'ao_labels':  [METAL_D_LABELS[s['metal']]],
            'threshold':  avas_threshold,
            't_s':        t_avas,
        }
        print(f"  AVAS active space: CAS({nelec_avas},{ncas_avas})")
        print(f"  AVAS wall time: {t_avas:.1f} s")

        # Step 4: DMRG on AVAS orbitals
        print(f"\n[Step 4] DMRG (M={M_dmrg}, {n_sweeps} sweeps) "
              f"on AVAS orbitals...")
        t0 = time.time()
        scratch = f'/tmp/qicas_avas_{name}'
        mc_dmrg, e_dmrg = run_dmrg_avas(
            mol, mf, mo_avas, ncas_avas, nelec_avas,
            M=M_dmrg, n_sweeps=n_sweeps,
            scratch_dir=scratch, name=name)
        t_dmrg = time.time() - t0
        result['e_dmrg']   = float(e_dmrg)
        result['t_dmrg_s'] = t_dmrg
        print(f"  E(DMRG) = {e_dmrg:.10f} Ha  ({t_dmrg:.1f} s)")

        # Step 5: Get 1-RDM and 2-RDM
        print("\n[Step 5] Getting 1-RDM and 2-RDM from DMRG...")
        gamma, Gamma = get_rdms(mc_dmrg)
        entropies_init = entropies_from_rdms(gamma, Gamma, ncas_avas)
        d_cas_init = entropy_plateau_cas_size(entropies_init, s['spin_2s'])
        result['entropies_initial'] = {
            'values':             entropies_init.tolist(),
            'd_cas_from_plateau': d_cas_init,
            'n_avas_orbitals':    ncas_avas,
        }
        print(f"  Entropy range (AVAS basis): "
              f"[{entropies_init.min():.4f}, {entropies_init.max():.4f}]")
        print(f"  D_CAS from plateau: {d_cas_init}  "
              f"(autoCAS reference: {s['autocas_no']})")

        # Step 6: QICAS orbital rotation within AVAS space
        print(f"\n[Step 6] QICAS rotation within AVAS space...")
        t0 = time.time()
        U_qicas, ent_qicas, gamma_qicas, fqi_i, fqi_f = \
            qicas_orbital_rotation(gamma, Gamma, ncas_avas, d_cas_init)
        t_rot = time.time() - t0

        active_rel, n_active, n_active_e = determine_active_space_from_qicas(
            ent_qicas, gamma_qicas, s['spin_2s'], d_cas_init, mf, nelec_avas)

        result['qicas'] = {
            'n_active':        n_active,
            'n_active_e':      n_active_e,
            'fqi_initial':     fqi_i,
            'fqi_final':       fqi_f,
            'fqi_reduction':   fqi_i - fqi_f,
            't_rotation_s':    t_rot,
            'entropies_qicas': ent_qicas.tolist(),
        }
        print(f"  QICAS active space: CAS({n_active_e},{n_active})")
        print(f"  F_QI: {fqi_i:.4f} → {fqi_f:.4f} "
              f"(reduction: {fqi_i-fqi_f:.4f})")
        print(f"  Rotation time: {t_rot:.1f} s")

        # Step 7: Build MO arrays
        # mo_hf_avas: AVAS orbitals as-is (HF-AVAS starting point)
        # mo_qicas_avas: AVAS orbitals rotated by U_qicas
        print("\n[Step 7] Building MO arrays...")
        mo_hf_avas    = mo_avas.copy()
        mo_qicas_avas = mo_avas.copy()

        # Apply QICAS rotation to the active block of AVAS orbitals
        # AVAS orders: [core | active | virtual]
        # Need to find where the active block starts
        n_mo = mo_avas.shape[1]
        # n_core in AVAS = (n_total_elec - nelec_avas) / 2
        n_core_avas = (mol.nelectron - nelec_avas) // 2
        act_start = n_core_avas
        act_end   = n_core_avas + ncas_avas

        # Rotate active block: mo_active_new = mo_active @ U.T
        mo_qicas_avas[:, act_start:act_end] = (
            mo_avas[:, act_start:act_end] @ U_qicas.T
        )

        # Step 8 (Goal 1): CASCI comparison
        print(f"\n[Step 8 / Goal 1] CASCI at CAS({n_active_e},{n_active})...")
        t0 = time.time()
        e_hf_ci, e_qi_ci, delta_ci = run_casci_comparison(
            mol, mf, mo_hf_avas, mo_qicas_avas, n_active_e, n_active)
        t_ci = time.time() - t0

        result['goal1_casci'] = {
            'cas_ne':          n_active_e,
            'cas_no':          n_active,
            'e_casci_hf_avas': e_hf_ci,
            'e_casci_qicas':   e_qi_ci,
            'delta_mha':       delta_ci,
            'qicas_better':    delta_ci < 0,
            't_s':             t_ci,
        }
        print(f"  E(CASCI|HF-AVAS)  = {e_hf_ci:.10f} Ha")
        print(f"  E(CASCI|QICAS)    = {e_qi_ci:.10f} Ha")
        print(f"  Δ = {delta_ci:+.3f} mHa "
              f"({'QICAS better' if delta_ci < 0 else 'HF-AVAS better'})")

        # Step 9 (Goal 2): CASSCF from both starting points
        print(f"\n[Step 9 / Goal 2] CASSCF from HF-AVAS and QICAS-AVAS "
              f"at CAS({n_active_e},{n_active})...")
        casscf_res = run_casscf_comparison(
            mol, mf, mo_hf_avas, mo_qicas_avas, n_active_e, n_active)

        e_hf_scf  = casscf_res['from_hf_avas']['e_casscf']
        e_qi_scf  = casscf_res['from_qicas_avas']['e_casscf']
        n_hf_iter = casscf_res['from_hf_avas']['n_macro_iter']
        n_qi_iter = casscf_res['from_qicas_avas']['n_macro_iter']
        speedup   = n_hf_iter - n_qi_iter

        result['goal2_casscf'] = {
            'cas_ne':            n_active_e,
            'cas_no':            n_active,
            'from_hf_avas':      casscf_res['from_hf_avas'],
            'from_qicas_avas':   casscf_res['from_qicas_avas'],
            'iter_speedup':      speedup,
            'qicas_faster':      speedup > 0,
            'energy_diff_mha':   (e_qi_scf - e_hf_scf) * 1000.0,
        }
        print(f"  E(CASSCF|HF-AVAS)  = {e_hf_scf:.10f}  "
              f"({n_hf_iter} iters, "
              f"{'conv' if casscf_res['from_hf_avas']['converged'] else 'not conv'})")
        print(f"  E(CASSCF|QICAS)    = {e_qi_scf:.10f}  "
              f"({n_qi_iter} iters, "
              f"{'conv' if casscf_res['from_qicas_avas']['converged'] else 'not conv'})")
        print(f"  Iter speedup: {speedup:+d} "
              f"({'QICAS faster' if speedup > 0 else 'HF-AVAS faster'})")

        # Post-hoc comparison with autoCAS and canonical QICAS
        result['comparison'] = {
            'avas_qicas_no':   n_active,
            'avas_qicas_ne':   n_active_e,
            'autocas_no':      s['autocas_no'],
            'autocas_ne':      s['autocas_ne'],
            'delta_no_vs_autocas': n_active - s['autocas_no'],
            'delta_ne_vs_autocas': n_active_e - s['autocas_ne'],
            'note': 'autoCAS and canonical QICAS stored for post-hoc comparison only',
        }

        result['status']      = 'OK'
        result['wall_time_s'] = time.time() - t_total

        print(f"\n  ── DONE in {result['wall_time_s']:.0f} s ──")
        print(f"  autoCAS reference:    CAS({s['autocas_ne']},{s['autocas_no']})")
        print(f"  AVAS initial:         CAS({nelec_avas},{ncas_avas})")
        print(f"  AVAS+QICAS final:     CAS({n_active_e},{n_active})")

    except Exception as exc:
        result['status']      = 'ERROR'
        result['error']       = str(exc)
        result['traceback']   = traceback.format_exc()
        result['wall_time_s'] = time.time() - t_total
        print(f"\n  ERROR: {exc}")
        print(traceback.format_exc())

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Result: {out_path}")
    return result


# ── CLI entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='AVAS-QICAS pipeline for 16 TM benchmark systems')
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--system',       type=str)
    grp.add_argument('--system_index', type=int)
    parser.add_argument('--M',         type=int,   default=100)
    parser.add_argument('--sweeps',    type=int,   default=30)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='AVAS projection threshold (default 0.1)')
    parser.add_argument('--out_dir',   type=str,   default='results_avas')
    args = parser.parse_args()

    if args.system_index is not None:
        if not (0 <= args.system_index < len(ALL_SYSTEM_NAMES)):
            print(f"ERROR: index must be 0-{len(ALL_SYSTEM_NAMES)-1}")
            sys.exit(1)
        name = ALL_SYSTEM_NAMES[args.system_index]
    else:
        name = args.system
        if name not in SYSTEMS:
            print(f"ERROR: unknown system '{name}'")
            print(f"Available: {ALL_SYSTEM_NAMES}")
            sys.exit(1)

    run_one_system(name, SYSTEMS[name],
                   M_dmrg=args.M, n_sweeps=args.sweeps,
                   avas_threshold=args.threshold,
                   out_dir=args.out_dir)


if __name__ == '__main__':
    main()
