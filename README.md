# QICAS for Transition Metal Complexes

Quantum Information-Assisted Complete Active Space (QICAS) pipeline
applied to 3d and 4d transition metal complexes for ML label generation.

## Project Context

Part of the **qHPC-GREEN** project at PC2 Paderborn (account: hpc-prf-qehpc).
Goal: generate reliable CASSCF active space labels for ~6100 transition metal
complexes to train an ML model for VQE/quantum computing applications.

Based on the QICAS method:
> Ding, Knecht, Schilling. *J. Phys. Chem. Lett.* 2023, 14, 11022–11029.

## Repository Structure

```
├── qicas_casscf_16systems.py   # Canonical QICAS pipeline (HF/UHF → window → DMRG → QICAS → CASSCF)
├── qicas_avas_16systems.py     # AVAS-QICAS pipeline (RHF/UHF → AVAS → DMRG → QICAS → CASSCF)
├── analyse_qicas_16.py         # Analysis script: reads result JSONs and prints summary tables
├── check_ecp.py                # ECP verification for 4d metals (Mo, Rh, Ru, Pd)
├── submit_qicas_16.slurm       # SLURM array script for canonical pipeline (16 systems)
├── submit_qicas_avas.slurm     # SLURM array script for AVAS pipeline (16 systems)
└── README.md
```

## Two Pipelines

### 1. Canonical QICAS (`qicas_casscf_16systems.py`)
```
UHF → frontier window (20-24 MOs) → DMRG(M=100) → γ,Γ → F_QI rotation → CASCI + CASSCF
```
- Faithful to original QICAS paper (Figure 2)
- UHF replaces HF for open-shell transition metals
- Active space determined by entropy plateau (Stein-Reiher method)

**Key finding:** Works well for high-spin systems (2S≥3, ΔCASCI = 169–566 mHa)
but gives diffuse entropy profiles for low-spin systems (2S≤1, ΔCASCI ≈ 0).

### 2. AVAS-QICAS (`qicas_avas_16systems.py`)
```
RHF/UHF → AVAS(metal d) → DMRG(M=100) → γ,Γ → F_QI rotation → CASCI + CASSCF
```
- RHF for singlets, UHF for open-shell
- AVAS pre-rotates orbitals to concentrate metal d-character
- Gives sharp entropy plateau even for low-spin systems
- Fixes the canonical pipeline failure for low-spin 4d complexes

## 16 Benchmark Systems

| System | Metal | Ligand | Geometry | 2S | Row |
|--------|-------|--------|----------|----|-----|
| CSD_CrCl4_2m_tet_spin4 | Cr | Cl | tet | 4 | 3d |
| CSD_MnCl4_2m_tet_spin5 | Mn | Cl | tet | 5 | 3d |
| CSD_MnBr4_2m_tet_spin5 | Mn | Br | tet | 5 | 3d |
| CSD_MnF6_4m_oct_spin5  | Mn | F  | oct | 5 | 3d |
| CSD_CrCl4_2m_tet_spin2 | Cr | Cl | tet | 2 | 3d |
| CSD_FeCl4_2m_tet_spin0 | Fe | Cl | tet | 0 | 3d |
| CSD_NiCl4_2m_sqpl_spin0| Ni | Cl | sqpl| 0 | 3d |
| CSD_NiCl6_4m_oct_spin2 | Ni | Cl | oct | 2 | 3d |
| CSD_CoCl6_4m_oct_spin1 | Co | Cl | oct | 1 | 3d |
| Mo_Cl6_chg-3_spin3     | Mo | Cl | oct | 3 | 4d |
| Mo_Cl6_chg-3_spin1     | Mo | Cl | oct | 1 | 4d |
| Rh_Cl6_chg-3_spin0     | Rh | Cl | oct | 0 | 4d |
| Ru_Cl6_chg-3_spin1     | Ru | Cl | oct | 1 | 4d |
| Pd_Cl4_chg-2_spin0     | Pd | Cl | sqpl| 0 | 4d |
| Rh_Cl6_chg-3_spin2     | Rh | Cl | oct | 2 | 4d |
| Pd_Cl6_chg-2_spin0     | Pd | Cl | oct | 0 | 4d |

## Usage

### Single system test
```bash
source ~/.block2_fix/block2_env.sh

# Canonical pipeline
python qicas_casscf_16systems.py --system Rh_Cl6_chg-3_spin0_oct_d2p32 --M 100 --sweeps 30

# AVAS pipeline
python qicas_avas_16systems.py --system Rh_Cl6_chg-3_spin0_oct_d2p32 --M 100 --sweeps 30 --threshold 0.1
```

### Full 16-system SLURM array
```bash
# Canonical
sbatch submit_qicas_16.slurm

# AVAS
sbatch submit_qicas_avas.slurm
```

### Analyse results
```bash
python analyse_qicas_16.py --results_dir results/
python analyse_qicas_16.py --results_dir results_avas/
```

## Key Findings

### Canonical QICAS benchmark results (M=100, 15/16 systems)

| Spin class | Systems | Avg ΔCASCI | CASSCF conv | Notes |
|------------|---------|------------|-------------|-------|
| High-spin (2S≥3) | 4 | −315 mHa | 0/4 | Active space too large for FCI |
| Mid-spin (2S=2) | 3 | −32 mHa | 1/3 | Mixed results |
| Low-spin (2S≤1) | 8 | −4 mHa | 4/8 | Near-zero QICAS benefit |

### AVAS-QICAS improvement for low-spin systems

| System | Canonical | AVAS-QICAS | autoCAS |
|--------|-----------|------------|---------|
| RhCl6 spin0 | CAS(24,12) Δ=0 mHa | CAS(6,6) Δ=−7 mHa | CAS(2,4) |
| PdCl6 spin0 | CAS(24,12) Δ=0 mHa | CAS(6,5) Δ=−X mHa | CAS(4,5) |

## Dependencies

- PySCF 2.13.1
- block2 (DMRG backend)
- numpy, scipy

## Cluster Setup (Noctua2 / Otus, PC2 Paderborn)

```bash
source ~/.block2_fix/block2_env.sh
```

## Reference

Ding, L.; Knecht, S.; Schilling, C. Quantum Information-Assisted Complete
Active Space Optimization (QICAS). *J. Phys. Chem. Lett.* **2023**, *14*,
11022–11029. https://doi.org/10.1021/acs.jpclett.3c02536
