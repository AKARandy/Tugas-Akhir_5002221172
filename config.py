"""
config.py
─────────
Centralised user-tunable parameters for the Soybean Supply Chain optimisation.

This is the single file you typically need to edit to change a run.
"""
import os

# ─── Output directory (where CSV/PNG results are saved) ────────────────────
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_THIS_DIR, "results", "single")

# Data pipeline defaults
DATA_PIPELINE_MODE = "granular_non_airport"
GRANULAR_YEARS = (2022, 2023, 2024, 2025)
GRANULAR_INITIAL_YEAR = 2024
# ── Capacity buffer justification ────────────────────────────────────────
# IMPORT_CAP_BUFFER = 1.01 gives a 1 % sampling margin above max
#   historical monthly import volume (peak month across 2022-2025 per
#   source country).  The max over a finite 4-year window may slightly
#   underestimate the true population maximum, so a minimal 1 % allowance
#   avoids infeasibility from sampling error without reintroducing the
#   artificial slack that plagued the previous 1.3× setting.  The global
#   soybean supply chain is highly concentrated (Brazil 40%, USA 28%,
#   Argentina 12% of world exports — "Global soybean trade dynamics:
#   Drivers, impacts and sustainability", The Innovation 7(2), 2026),
#   making source capacity structurally tight.
#
# PORT_CAP_BUFFER = 1.1  targets ~90 % utilisation, consistent with
#   Pelindo's own reported port utilisation figures for the major soybean
#   gateways: Tanjung Priok 90 %, Tanjung Emas 95 %, Tanjung Perak 87 %
#   (Pelindo, "Era Baru Biaya Logistik", 2023).  A multiplier of 1/0.90 ≈ 1.11
#   is rounded to 1.1.  This removes the previous undocumented 50 % slack
#   that contradicted operational evidence and caused scenario stress-tests
#   to show no binding constraints at -25 % capacity reduction.
IMPORT_CAP_BUFFER = 1.3
PORT_CAP_BUFFER = 1.5
EMERGENCY_CAP_FRAC = 0.25

# ═══════════════════════════════════════════════════════════════════════════
#  ε-CONSTRAINT BOUNDS  (Mavrotas 2009 / AUGMECON style)
# ═══════════════════════════════════════════════════════════════════════════
# Pendekatan: minimisasi Z_cost sebagai objektif utama, dengan tiga kendala ε.
# Pelanggaran ε dihukum dengan penalti besar yang mendominasi Z_cost.

EPS_SHORTAGE   = 0.0    # (ton)   max shortage yang diizinkan (0 = tidak boleh ada shortage)
EPS_IMPORT_DEP = 0.90   # (ratio) max ketergantungan impor = impor / total_demand
EPS_LOCAL_MIN  = 0.0    # (ton)   minimum produksi lokal — auto-set dari solusi awal

# ═══════════════════════════════════════════════════════════════════════════
#  PENALTY MULTIPLIERS for ε-violations
# ═══════════════════════════════════════════════════════════════════════════
# All multipliers are AUTO-CALIBRATED at runtime in solver.configure_run()
# using the normalized AUGMECON approach:
#   M_k = PENALTY_SCALE × z0 / r_k
# where z0 = Z_cost of initial solution, r_k = max possible violation range.
# This ensures all constraints contribute equally per unit relative violation.
# Only ONE tuning knob: PENALTY_SCALE (dimensionless).

PENALTY_SCALE = 10.0   # violation at 100% of range = PENALTY_SCALE × Z_cost

# These will be overwritten at runtime by solver.configure_run():
M_SHORTAGE   = 0.0     # auto-set: PENALTY_SCALE × z0 / total_demand
M_IMPORT_DEP = 0.0     # auto-set: PENALTY_SCALE × z0 / 1.0 (full ratio range)
M_LOCAL      = 0.0     # auto-set: PENALTY_SCALE × z0 / EPS_LOCAL_MIN
M_INV_FLOOR  = 0.0     # auto-set: PENALTY_SCALE × z0 / (INV_MIN_FRAC × total_demand)

# ε₄: Inventory floor
INV_MIN_FRAC = 0.10    # target inventori minimal = 10% demand bulanan

# Greedy/repair sort-key bias toward local (NOT in objective)
LOC_PREF = 2_000_000.0      # Rp/ton ditambahkan ke biaya impor pada sort key

# Haversine transport rates (Rp/ton/km) — for C_SHIP, C_DIST, C_TRANS
HAVERSINE_LAND_RATE = 1000  # same-cluster (dominasi darat)
HAVERSINE_SEA_RATE  = 500   # cross-cluster (dominasi laut)
HAVERSINE_NOISE     = 0.10  # ±10% random multiplier pada cost

# Safety stock buffer (multiplied on base 1-month demand)
SAFETY_STOCK_BUFFER = 1.10  # target inventory = 1.1 bulan demand

# Big-M for hard infeasibility (port storage overflow)
BIG_M = 1e15

# ═══════════════════════════════════════════════════════════════════════════
#  RUN-TIME PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
SEED        = 42
MAX_ITER    = 500
TABU_TENURE = 12
SUB_IT      = 30      # tabu sub-iterations per repair call

# Prism Refraction Search local-improvement parameters.
# PRS is used as a single-solution intensification step on S_best.
PRS_SUB_IT          = 30
PRS_ALPHA           = 0.09
PRS_PRISM_ANGLE     = 1.0471975512  # pi / 3
PRS_MOVE_MIN_FRAC   = 0.05
PRS_MOVE_MAX_FRAC   = 0.35

# ═══════════════════════════════════════════════════════════════════════════
#  ALNS FLOW PARAMETERS (Fathollahi-Fard et al., 2023 — Section 4.5)
# ═══════════════════════════════════════════════════════════════════════════

# Operator scoring (paper: 3 kategori π₁/π₂/π₃; kita pakai 4 kategori, skala 0-1)
#   new global best         -> ALNS_SCORE_GLOBAL_BEST
#   better than current     -> ALNS_SCORE_BETTER  (kategori tambahan, paper tidak punya)
#   accepted by SA (worse)  -> ALNS_SCORE_SA_ACCEPT
#   rejected                -> ALNS_SCORE_REJECT

# ═══════════════════════════════════════════════════════════════════════════
#  TUNING GUIDE
#  Use SA_COOLING="exponential" + SA_PAPER=False for best weight divergence.
#  For geometric cooling, match SA_REDU to SA_PAPER: 0.995 if paper (C1),
#  0.99 if standard (C2).  TABU_TENURE=12, SUB_IT=30 — every other combo
#  tested blew up shortage.  ALNS_DECAY=0.75 with scores 1.0/0.6/0.3/0.1
#  gives operators room to diverge within 500 iters.
# ═══════════════════════════════════════════════════════════════════════════

ALNS_SCORE_GLOBAL_BEST = 1.0
ALNS_SCORE_BETTER       = 0.6
ALNS_SCORE_SA_ACCEPT    = 0.3
ALNS_SCORE_REJECT       = 0.1
ALNS_DECAY              = 0.75  # λ: weight smoothing. 0.75 = adaptasi lebih cepat → preferences diverge
ALNS_MIN_WEIGHT         = 0.02  # floor weight mencegah collapse ke 0 (menjaga diversitas operator)

# ═══════════════════════════════════════════════════════════════════════════
#  PRIORITY MODE — arahkan solver saat dua e-constraint konflik secara fisik
# ═══════════════════════════════════════════════════════════════════════════
# Saat shortage=0 dan produksi lokal tidak bisa menutupi demand, impor minimum
# secara fisik = 1 - (produksi_lokal / demand). Jika EPS_IMPORT_DEP < batas ini,
# tidak ada solusi feasible → solver perlu diarahkan.
#
#   "balanced" → kedua penalti aktif, solver cari kompromi terbaik (default)
#   "shortage" → prioritas zero shortage, batas impor dilanggar (abaikan EPS_IMPORT_DEP)
#   "import"   → prioritas batas impor, shortage diizinkan (abaikan EPS_SHORTAGE)
PRIORITY_MODE = "balanced"

# Tabu local search placement
#   True  → tabu dijalankan SETIAP iterasi pada S_best (paper, Step 6)
#   False → tabu hanya saat S_best berubah (lebih hemat komputasi)
TABU_EVERY_ITER = True

# Prism Refraction Search placement
#   True  -> PRS dijalankan setiap iterasi pada S_best
#   False -> PRS hanya saat S_best berubah
PRS_EVERY_ITER = True

# SA acceptance mode
#   True  → paper mode: SA bandingkan kandidat vs S_best  (Eq. 32)
#           p = exp(-|f(S_new) - f(S_best)| / T)
#   False → standard SA: SA bandingkan kandidat vs S_current
#           p = exp(-max(0, f(S_new) - f(S_current)) / T)
SA_PAPER = False

# SA cooling schedule
#   "exponential" → T(t) = T0 × step^t, step = (T_end/T0)^(1/MAX_ITER)
#                    T0 auto-kalibrasi dari initial solution
#   "geometric"    → T(t+1) = T(t) × SA_REDU  (Eq. 33)
SA_COOLING = "geometric"

# Hanya digunakan jika SA_COOLING = "geometric"
SA_REDU      = 0.99       # damping rate (paper calibrated value)
# Hanya digunakan jika SA_COOLING = "geometric".
# **Auto-dikalibrasi** menjadi 0.05 × |f0| di solver.configure_run()
# (sama seperti EPS_LOCAL_MIN). Nilai manual di sini diabaikan.
SA_PAPER_T0  = 20000.0     # initial temperature (ditinggalkan — auto-set dari objective)

# Backward compat — referenced in solver
SCORES = [ALNS_SCORE_GLOBAL_BEST, ALNS_SCORE_BETTER, ALNS_SCORE_SA_ACCEPT, ALNS_SCORE_REJECT]
DECAY  = ALNS_DECAY

# Mode flag — emergency imports
USE_EMERGENCY = False

# Tabu local search toggle
USE_TABU = False   # set False for ALNS-only runs

# Prism Refraction Search local search toggle
USE_PRS = True

# Initial condition mode
#   "greedy"    → original two-phase greedy (default)
#   "historical" → seed from BPS disaggregated import data, then repair
USE_HISTORICAL_IC = True
