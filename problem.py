"""
problem.py
──────────
The single source of truth for problem parameters: the immutable `Problem` dataclass.

The class wraps every constant the solver needs (dimensions, names, demand,
capacities, costs). It is built once via `Problem.load(data_dir)` and then passed
by reference to `SoybeanState`, the operators, and the reporting layer.

`SoybeanState` keeps a *reference* to the same Problem instance — it is never
copied. The `frozen=True` dataclass guarantees no one can mutate it after load.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

import config
import data_loader as dl


# ═══════════════════════════════════════════════════════════════════════════
#  Static problem dimensions and registries
# ═══════════════════════════════════════════════════════════════════════════

N_PROV   = 38   # |I| – 38 provinsi (termasuk 4 provinsi Papua baru sejak 2022)
N_IMP    = 7    # |S| - strategic multi-year import origins
N_PORT   = 11   # |H| – 11 pelabuhan transit (dari data BPS)
N_PERIOD = 12   # |T| – horizon perencanaan 12 bulan
EXCH_RATE = 16_000   # Rp/USD, rata-rata 2024 (approximation)

PROV_NAMES = [
    "Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Jambi",
    "Sumatera Selatan", "Bengkulu", "Lampung", "Kep. Bangka Belitung",
    "Kepulauan Riau", "DKI Jakarta", "Jawa Barat", "Jawa Tengah",
    "DI Yogyakarta", "Jawa Timur", "Banten", "Bali", "Nusa Tenggara Barat",
    "Nusa Tenggara Timur", "Kalimantan Barat", "Kalimantan Tengah",
    "Kalimantan Selatan", "Kalimantan Timur", "Kalimantan Utara",
    "Sulawesi Utara", "Sulawesi Tengah", "Sulawesi Selatan",
    "Sulawesi Tenggara", "Gorontalo", "Sulawesi Barat", "Maluku",
    "Maluku Utara", "Papua Barat", "Papua Barat Daya", "Papua",
    "Papua Selatan", "Papua Tengah", "Papua Pegunungan",
]

IMP_NAMES = ["USA", "Kanada", "Argentina", "Brazil", "Malaysia", "Bolivia", "Uruguay"]
N_IMP = len(IMP_NAMES)

PORT_NAMES = [
    "Belawan", "Panjang", "Batu Ampar", "Tanjung Priok", "Tanjung Emas",
    "Gresik", "Tanjung Perak", "Cigading", "Cilacap", "Banyuwangi",
    "Pontianak",
]
N_PORT = len(PORT_NAMES)

# PORT_SERV[h] = list of provinces served by port h (geographic mapping)
PORT_SERV: Dict[int, List[int]] = {
    0:  [0, 1, 2, 3],                                          # Belawan
    1:  [5, 6, 7, 8],                                          # Panjang
    2:  [9],                                                    # Batu Ampar
    3:  [10, 11],                                               # Tanjung Priok
    4:  [12, 13],                                               # Tanjung Emas
    5:  [14, 21, 22, 23],                                       # Gresik
    6:  [14, 16, 17, 18, 24, 25, 26, 27, 28, 29,
         30, 31, 32, 33, 34, 35, 36, 37],                      # Tanjung Perak
    7:  [11, 15],                                               # Cigading
    8:  [11, 12, 13],                                           # Cilacap
    9:  [14, 16, 17, 18],                                       # Banyuwangi
    10: [19, 20, 21],                                           # Pontianak
}

# CLUSTER[i] = geographic cluster of province i (for D4 destroy operator)
CLUSTER: Dict[int, int] = {
    **{i: 0 for i in range(0, 10)},   # Sumatera
    **{i: 1 for i in range(10, 16)},  # Jawa
    **{i: 2 for i in range(16, 19)},  # Bali & Nusa Tenggara
    **{i: 3 for i in range(19, 24)},  # Kalimantan
    **{i: 4 for i in range(24, 30)},  # Sulawesi
    **{i: 5 for i in range(30, 38)},  # Maluku & Papua
}


# ═══════════════════════════════════════════════════════════════════════════
#  Static data tables (BPS Survei 2022 fallback)
# ═══════════════════════════════════════════════════════════════════════════

# Annual production per province (BPS Tabel 4, 2022) — the only source for production
ANNUAL_PROD_TON_ALL = np.array([
    1_501.89, 8_026.28, 17.23, 495.19, 4_631.28, 39.31, 3.21, 2_109.76,
    4.53, 0.32, 0.00, 36_011.52, 62_031.66, 7_282.08, 69_656.57, 1_849.25,
    3_249.74, 9_726.35, 1_013.66, 78.47, 14.83, 6_114.88, 62.10, 0.00,
    4.64, 10_213.75, 3_852.04, 11_052.87, 1_549.34, 511.22, 63.17, 103.56,
    10.09, 6.73, 58.66, 29.33, 29.33, 29.33,
], dtype=float)

# Production seasonality (panen MT1 puncak April, MT2 puncak Agustus)
PROD_SEASON = np.array([0.25, 0.35, 0.90, 1.80, 1.95, 1.15,
                         0.75, 1.40, 1.55, 0.90, 0.50, 0.50])

# Fallback annual demand (BPS Survei 2022, Tabel 4) — used only if Neraca xlsx missing
ANNUAL_DEMAND_TON_FALLBACK = np.array([
    10_158.38, 59_643.58, 4_388.64, 24_628.81, 20_680.38, 19_351.56,
    12_932.28, 46_562.32, 2_976.50, 5_490.40, 61_313.90, 239_382.72,
    226_917.52, 31_006.03, 330_078.51, 36_041.77, 22_695.01, 52_394.07,
    11_236.66, 11_856.86, 4_061.26, 12_462.03, 13_580.00, 236.77,
    3_335.30, 5_957.28, 13_506.10, 13_802.54, 1_050.54, 131.84, 267.07,
    120.61, 750.04, 500.03, 1_638.98, 819.49, 819.49, 819.49,
], dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
#  The Problem dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Problem:
    """
    Immutable bundle of all problem-instance parameters.

    Built once via `Problem.load(data_dir)`. SoybeanState carries a reference
    to a Problem; operators read parameters via `state.problem.X`.
    """
    # ─── Dimensions ───────────────────────────────────────────────────────
    N_PROV: int
    N_IMP: int
    N_PORT: int
    N_PERIOD: int
    N_PROD: int
    EXCH_RATE: int

    # ─── Names & maps ─────────────────────────────────────────────────────
    PROV_NAMES: List[str]
    IMP_NAMES: List[str]
    PORT_NAMES: List[str]
    PROD_IDX: List[int]
    PORT_SERV: Dict[int, List[int]]
    CLUSTER: Dict[int, int]

    # ─── Demand & capacities (all numpy arrays) ───────────────────────────
    DEMAND: np.ndarray            # (N_PROV, N_PERIOD)
    PROD_CAP: np.ndarray          # (N_PROD, N_PERIOD)
    IMP_CAP_NORMAL: np.ndarray    # (N_IMP, N_PERIOD)
    IMP_CAP_EMERG: np.ndarray     # (N_IMP, N_PERIOD)
    PORT_THRU_CAP: np.ndarray     # (N_PORT, N_PERIOD)
    PROV_STOR_CAP: np.ndarray     # (N_PROV,)
    SAFETY_STOCK: np.ndarray      # (N_PROV,)
    X_THRESHOLD: float

    # ─── Cost parameters ──────────────────────────────────────────────────
    C_PROD: np.ndarray            # (N_PROD,)
    C_SHIP: np.ndarray            # (N_PROD, N_PROV)
    C_PURCH: np.ndarray           # (N_IMP,)   CIF Rp/ton
    C_EMG: np.ndarray             # (N_IMP,)   = C_PURCH × 1.3
    C_DIST: np.ndarray            # (N_PORT, N_PROV)
    C_TRANS: np.ndarray           # (N_PROV, N_PROV)
    H_PROV: np.ndarray            # (N_PROV,)  Rp/ton/bulan
    F_ACT: np.ndarray             # (N_IMP,)   Rp fixed
    F_EMG: float                  # Rp fixed per emergency activation

    # ─── Historical import data (for initial condition) ──────────────────────
    HIST_IMPORT: np.ndarray       # (N_IMP, N_PORT, N_PERIOD) — tons, from disaggregated XLSX

    # ─── Policy sets ──────────────────────────────────────────────────────
    CRITICAL_PROV: List[int]

    # ──────────────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, data_dir: str | None = None, seed: int = 42) -> "Problem":
        """
        Build a Problem instance by reading CSV/XLSX files from `data_dir`.

        Pipeline:
          1. Read country-level imports CSV → IMP_CAP_NORMAL, C_PURCH, demand season
          2. Read port-level soybean imports CSV → PORT_THRU_CAP
          3. Read port annual throughput CSV (reference only)
          4. Try Neraca xlsx for DEMAND; fall back to BPS 2022 if missing
          5. Generate randomised cost arrays via seeded RNG (reproducible)
        """
        if data_dir is None:
            data_dir = os.path.dirname(os.path.abspath(__file__))

        rng = np.random.default_rng(seed)

        # ── 1. Read CSVs ───────────────────────────────────────────────────
        country_data = dl.read_bps_trade_csv("Impor Kedelai Berdasarkan Negara Asal.csv",
                                             data_dir)

        # ── 2. DEMAND — try Neraca xlsx first ─────────────────────────────
        demand, annual_demand = _build_demand(data_dir, country_data)

        # ── 3. Production: PROD_IDX = provinces with > 100 ton/year ────────
        prod_idx = [i for i in range(N_PROV) if ANNUAL_PROD_TON_ALL[i] > 100.0]
        n_prod   = len(prod_idx)
        assert abs(PROD_SEASON.sum() - 12.0) < 1e-9
        prod_cap = np.array([
            ANNUAL_PROD_TON_ALL[prov] * PROD_SEASON / 12.0
            for prov in prod_idx
        ])

        # ── 4. Import capacities (volume × 1.3 buffer, with country-floor) ─
        if config.DATA_PIPELINE_MODE != "granular_non_airport":
            raise ValueError(
                f"Unsupported DATA_PIPELINE_MODE={config.DATA_PIPELINE_MODE!r}; "
                "current implementation expects 'granular_non_airport'."
            )

        granular_import = dl.read_granular_import_pipeline(
            data_dir,
            n_imp=N_IMP,
            n_port=N_PORT,
            n_period=N_PERIOD,
            years=tuple(config.GRANULAR_YEARS),
            initial_year=config.GRANULAR_INITIAL_YEAR,
            exch_rate=EXCH_RATE,
            import_cap_buffer=config.IMPORT_CAP_BUFFER,
            port_cap_buffer=config.PORT_CAP_BUFFER,
        )

        imp_cap_normal = granular_import["imp_cap_normal"]
        imp_cap_emerg = imp_cap_normal * config.EMERGENCY_CAP_FRAC

        # ── 5. Port soybean-specific throughput capacity ───────────────────
        port_thru_cap = granular_import["port_thru_cap"]

        # ── 6. Province-level storage / safety stock / emergency threshold ─
        prov_stor_cap = (annual_demand / 12.0) * 4.0
        safety_stock  = (annual_demand / 12.0) * config.SAFETY_STOCK_BUFFER
        x_threshold   = float(demand.sum(axis=0).mean() * 0.05)

        # ── 7. Cost arrays ────────────────────────────────────────────────
        c_prod  = rng.integers(8_000_000, 12_000_000, n_prod).astype(float)
        c_ship, c_dist, c_trans = _build_transport_costs(rng, n_prod, prod_idx)
        c_purch = granular_import["c_purch"]
        c_emg   = c_purch * 1.3
        h_prov  = rng.integers(50_000, 150_000, N_PROV).astype(float)
        f_act   = rng.integers(500_000_000, 5_000_000_000, N_IMP).astype(float)
        f_emg   = 5_000_000_000.0

        # ── 8. Historical import data (for initial condition) ──────────────
        hist_import = granular_import["hist_import"]
        # read_province_import() membaca Ekspor Impor Kedelai.xlsx (perdagangan
        # antar-provinsi, BUKAN impor internasional). Tidak dipakai di initial
        # solution — disimpan untuk referensi analisis saja.
        dl.read_province_import(data_dir, N_PROV, N_PERIOD)

        # Granular port capacity is already based on all-country non-airport evidence.
        # PORT_THRU_CAP is now direct granular multi-year evidence, with no floor.

        # ── 9. Critical provinces (non-producers, eligible for early transfers) ─
        critical_prov = [i for i in range(N_PROV) if i not in prod_idx]

        # ── 10. ε-penalty multipliers tuned to cost magnitudes ──────────────
        return cls(
            N_PROV=N_PROV, N_IMP=N_IMP, N_PORT=N_PORT,
            N_PERIOD=N_PERIOD, N_PROD=n_prod, EXCH_RATE=EXCH_RATE,
            PROV_NAMES=list(PROV_NAMES), IMP_NAMES=list(IMP_NAMES),
            PORT_NAMES=list(PORT_NAMES), PROD_IDX=list(prod_idx),
            PORT_SERV=dict(PORT_SERV), CLUSTER=dict(CLUSTER),
            DEMAND=demand, PROD_CAP=prod_cap,
            IMP_CAP_NORMAL=imp_cap_normal, IMP_CAP_EMERG=imp_cap_emerg,
            PORT_THRU_CAP=port_thru_cap,
            PROV_STOR_CAP=prov_stor_cap, SAFETY_STOCK=safety_stock,
            X_THRESHOLD=x_threshold,
            C_PROD=c_prod, C_SHIP=c_ship, C_PURCH=c_purch, C_EMG=c_emg,
            C_DIST=c_dist, C_TRANS=c_trans, H_PROV=h_prov,
            F_ACT=f_act, F_EMG=f_emg,
            CRITICAL_PROV=critical_prov,
            HIST_IMPORT=hist_import,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Private builders (called by Problem.load)
# ═══════════════════════════════════════════════════════════════════════════

def _build_demand(data_dir: str, country_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (DEMAND[N_PROV, N_PERIOD], annual_demand[N_PROV])."""
    neraca = dl.load_neraca_demand(data_dir, N_PROV, N_PERIOD, year=2024)

    if neraca is not None and np.nansum(neraca) > 0:
        # Fill 4 provinces missing from Neraca with proportional estimates
        for t in range(N_PERIOD):
            if np.isnan(neraca[6, t]):    # Bengkulu = avg(Sumsel, Lampung)
                ss = neraca[5, t] if not np.isnan(neraca[5, t]) else 0.0
                lp = neraca[7, t] if not np.isnan(neraca[7, t]) else 0.0
                neraca[6, t] = (ss + lp) / 2.0
            if np.isnan(neraca[33, t]):   # Papua Barat Daya = 40% Papua Barat
                pb = neraca[32, t] if not np.isnan(neraca[32, t]) else 0.0
                neraca[33, t] = pb * 0.4
            if np.isnan(neraca[35, t]):   # Papua Selatan = 20% Papua
                pp = neraca[34, t] if not np.isnan(neraca[34, t]) else 0.0
                neraca[35, t] = pp * 0.2
            if np.isnan(neraca[36, t]):   # Papua Tengah = 20% Papua
                pp = neraca[34, t] if not np.isnan(neraca[34, t]) else 0.0
                neraca[36, t] = pp * 0.2
        demand = np.nan_to_num(neraca, nan=0.0)
        annual = demand.sum(axis=1)
        print(f"[DATA] DEMAND dari Proyeksi Neraca Kedelai.xlsx (2024): "
              f"total = {demand.sum():,.0f} ton/tahun")
    else:
        # Fallback: BPS 2022 + scaling to 2.66M tons + import-based seasonality
        target = 2_660_000.0
        annual = ANNUAL_DEMAND_TON_FALLBACK * (target / ANNUAL_DEMAND_TON_FALLBACK.sum())
        total_monthly_kg = country_data.get('TOTAL', (np.zeros(12), np.zeros(12)))[0]
        if total_monthly_kg.sum() > 0:
            season = total_monthly_kg / total_monthly_kg.sum() * 12.0
        else:
            season = np.ones(12)
        demand = np.outer(annual / 12.0, season)
        print(f"[DATA] DEMAND dari BPS Survei 2022 + scaling: "
              f"total = {demand.sum():,.0f} ton/tahun")
    return demand, annual


def _build_import_capacity(
    country_data: dict,
    country_keys: list[str] | None = None,
    buffer: float = 1.3,
    floor_frac: float = 0.5,
) -> np.ndarray:
    """Legacy CSV capacity builder kept dynamic for audit paths."""
    if country_keys is None:
        country_keys = [
            'UNITED STATES', 'CANADA', 'ARGENTINA', 'BRAZIL',
            'MALAYSIA', 'BOLIVIA', 'URUGUAY',
        ]

    cap = np.zeros((len(country_keys), N_PERIOD), dtype=float)
    for s, key in enumerate(country_keys):
        monthly_kt = country_data.get(key, (np.zeros(N_PERIOD), np.zeros(N_PERIOD)))[0] / 1e6
        if monthly_kt.sum() > 0:
            cap[s] = np.maximum(monthly_kt * buffer, monthly_kt.mean() * floor_frac) * 1_000
    return cap


def _build_port_capacity(port_soy: dict) -> np.ndarray:
    """Build PORT_THRU_CAP[N_PORT, N_PERIOD]."""
    obs_kt = np.zeros((N_PORT, N_PERIOD))
    for h, csv_name in dl.PORT_SOY_MAP.items():
        vols_kg, _ = port_soy.get(csv_name, (np.zeros(12), np.zeros(12)))
        obs_kt[h, :] = vols_kg / 1e6   # ribu ton

    obs_ton = obs_kt * 1_000   # ribu ton → ton
    for h in range(N_PORT):
        annual_avg = obs_ton[h].mean()
        floor = max(annual_avg * 0.5, 500.0)
        obs_ton[h] = np.maximum(obs_ton[h], floor)

    thru_cap = obs_ton * 1.5                  # 1.5× buffer for soybean handling
    return thru_cap


def _compute_cif_prices(
    country_data: dict,
    country_keys: list[str] | None = None,
) -> np.ndarray:
    """C_PURCH (Rp/ton) = (USD/Kg) x 1000 x EXCH_RATE from CSV totals."""
    if country_keys is None:
        country_keys = [
            'UNITED STATES', 'CANADA', 'ARGENTINA', 'BRAZIL',
            'MALAYSIA', 'BOLIVIA', 'URUGUAY',
        ]

    def _cif(country_key: str) -> float:
        vk, vl = country_data.get(country_key, (np.zeros(12), np.zeros(12)))
        tk, tl = float(vk.sum()), float(vl.sum())
        if tk > 0:
            return (tl / tk) * 1000.0 * EXCH_RATE
        return 7_000_000.0    # fallback

    return np.array([_cif(key) for key in country_keys], dtype=float)



# ═══════════════════════════════════════════════════════════════════════════
#  Geographic coordinates (lat, lon) — from koordinat_38_provinsi_indonesia.csv
#  and koordinat_pelabuhan_bandara_import.csv
# ═══════════════════════════════════════════════════════════════════════════

PROV_COORDS = [
    ( 5.5483,  95.3238),  # 0  Aceh
    ( 3.5952,  98.6722),  # 1  Sumatera Utara
    (-0.9471, 100.4172),  # 2  Sumatera Barat
    ( 0.5071, 101.4478),  # 3  Riau
    (-1.6101, 103.6131),  # 4  Jambi
    (-2.9761, 104.7754),  # 5  Sumatera Selatan
    (-3.7928, 102.2608),  # 6  Bengkulu
    (-5.4290, 105.2611),  # 7  Lampung
    (-2.1293, 106.1098),  # 8  Kep. Bangka Belitung
    ( 0.9186, 104.4665),  # 9  Kepulauan Riau
    (-6.2088, 106.8456),  # 10 DKI Jakarta
    (-6.9175, 107.6191),  # 11 Jawa Barat
    (-6.9932, 110.4203),  # 12 Jawa Tengah
    (-7.7956, 110.3695),  # 13 DI Yogyakarta
    (-7.2575, 112.7521),  # 14 Jawa Timur
    (-6.1200, 106.1503),  # 15 Banten
    (-8.6705, 115.2126),  # 16 Bali
    (-8.5833, 116.1167),  # 17 Nusa Tenggara Barat
    (-10.1772, 123.6070), # 18 Nusa Tenggara Timur
    (-0.0319, 109.3250),  # 19 Kalimantan Barat
    (-2.2161, 113.9135),  # 20 Kalimantan Tengah
    (-3.4389, 114.8309),  # 21 Kalimantan Selatan
    (-0.5018, 117.1536),  # 22 Kalimantan Timur
    ( 2.8375, 117.3653),  # 23 Kalimantan Utara
    ( 1.4748, 124.8421),  # 24 Sulawesi Utara
    (-0.7893, 119.8592),  # 25 Sulawesi Tengah
    (-5.1477, 119.4327),  # 26 Sulawesi Selatan
    (-3.9985, 122.5129),  # 27 Sulawesi Tenggara
    ( 0.5371, 123.0596),  # 28 Gorontalo
    (-2.6806, 118.8861),  # 29 Sulawesi Barat
    (-3.6387, 128.1689),  # 30 Maluku
    ( 0.7244, 127.5806),  # 31 Maluku Utara
    (-0.8629, 134.0640),  # 32 Papua Barat
    (-0.8796, 131.2610),  # 33 Papua Barat Daya
    (-2.5337, 140.7181),  # 34 Papua
    (-8.4996, 140.4061),  # 35 Papua Selatan
    (-3.3599, 135.5007),  # 36 Papua Tengah
    (-4.0958, 138.9481),  # 37 Papua Pegunungan
]

PORT_COORDS = [
    ( 3.7876,  98.6957),  # 0  Belawan
    (-5.4759, 105.3198),  # 1  Panjang
    ( 1.1662, 104.0044),  # 2  Batu Ampar
    (-6.2088, 106.8456),  # 3  Tanjung Priok
    (-6.9932, 110.4203),  # 4  Tanjung Emas
    (-7.1539, 112.6561),  # 5  Gresik
    (-7.2575, 112.7521),  # 6  Tanjung Perak
    (-6.0123, 105.9570),  # 7  Cigading
    (-7.73844, 108.987578),  # 8  Cilacap / Tanjung Intan
    (-8.1166, 114.4000),  # 9  Banyuwangi / Tanjung Wangi
    (-0.01304626, 109.3354),  # 10 Pontianak
]

# Port → Island Cluster (same numbering as CLUSTER dict)
PORT_CLUSTER = {
    0: 0,  # Belawan = Sumatera
    1: 0,  # Panjang = Sumatera
    2: 0,  # Batu Ampar = Sumatera/Kep. Riau
    3: 1,  # Tanjung Priok = Jawa
    4: 1,  # Tanjung Emas = Jawa
    5: 1,  # Gresik = Jawa
    6: 1,  # Tanjung Perak = Jawa
    7: 1,  # Cigading = Jawa/Banten
    8: 1,  # Cilacap = Jawa
    9: 1,  # Banyuwangi = Jawa
    10: 3, # Pontianak = Kalimantan
}


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (km) between two (lat,lon) points in degrees."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2.0) ** 2)
    return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _build_transport_costs(rng: np.random.Generator, n_prod: int, prod_idx: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build transport costs using Haversine great-circle distances.

    Uses actual geographic coordinates (lat, lon) for all provinces and ports.
    IPI (Indeks Pembangunan Infrastruktur) multiplier still applied to
    capture road quality, port infrastructure, and logistics maturity.
    """
    c_ship = np.zeros((n_prod, N_PROV))
    c_dist = np.zeros((N_PORT, N_PROV))
    c_trans = np.zeros((N_PROV, N_PROV))

    # ── IPI Scores 2014 (Indeks Pembangunan Infrastruktur) ────────────
    # Source: Table 5, Faradis & Afifah (2020), JEPI Vol. 20 No. 1
    IPI_SCORES = [
        -0.249, -0.201, -0.224, -0.254, -0.258,   #  0–4:  Aceh → Jambi
        -0.254, -0.231, -0.201, -0.249, -0.146,   #  5–9:  Sumsel → Kepri
         5.553,  0.164,  0.077,  0.405, -0.015,   # 10–14: Jakarta → Jatim
         0.082,  0.369, -0.207, -0.260,            # 15–18: Banten → NTT
        -0.275, -0.278, -0.244, -0.273, -0.281,   # 19–23: Kalbar → Kalut
        -0.194, -0.266, -0.226, -0.255, -0.244,   # 24–28: Sulut → Gorontalo
        -0.262,                                      # 29:    Sulbar
        -0.270, -0.269,                              # 30–31: Maluku, Malut
        -0.281, -0.281, -0.282, -0.282, -0.282, -0.282,  # 32–37: Papua
    ]
    IPI_CAPPED = [min(0.5, max(-0.3, s)) for s in IPI_SCORES]
    min_ipi, max_ipi = min(IPI_CAPPED), max(IPI_CAPPED)
    IPI_MULT = [1.15 - 0.30 * ((s - min_ipi) / (max_ipi - min_ipi))
                for s in IPI_CAPPED]

    LAND_RATE = config.HAVERSINE_LAND_RATE
    SEA_RATE  = config.HAVERSINE_SEA_RATE
    NOISE     = config.HAVERSINE_NOISE

    # ── C_SHIP (Producer → Province) ─────────────────────────────────
    for k_arr, i_prov in enumerate(prod_idx):
        src_coord = PROV_COORDS[i_prov]
        for j_prov in range(N_PROV):
            if i_prov == j_prov:
                cost = 50_000 * IPI_MULT[j_prov]               # intra-provinsi flat
            else:
                dist = _haversine(*src_coord, *PROV_COORDS[j_prov])
                if CLUSTER[i_prov] == CLUSTER[j_prov]:
                    cost = dist * LAND_RATE * IPI_MULT[j_prov]
                else:
                    cost = dist * SEA_RATE * IPI_MULT[j_prov]
            noise = 1.0 + (rng.random() - 0.5) * 2.0 * NOISE
            c_ship[k_arr, j_prov] = cost * noise

    # ── C_DIST (Port → Province) ─────────────────────────────────────
    for h in range(N_PORT):
        port_coord = PORT_COORDS[h]
        port_cluster = PORT_CLUSTER.get(h)
        for i_prov in range(N_PROV):
            dist = _haversine(*port_coord, *PROV_COORDS[i_prov])
            if port_cluster == CLUSTER[i_prov]:
                cost = dist * LAND_RATE * IPI_MULT[i_prov]
            else:
                cost = dist * SEA_RATE * IPI_MULT[i_prov]
            noise = 1.0 + (rng.random() - 0.5) * 2.0 * NOISE
            c_dist[h, i_prov] = cost * noise

    # ── C_TRANS (Province → Province Transfer) ───────────────────────
    for i_prov in range(N_PROV):
        src_coord = PROV_COORDS[i_prov]
        for j_prov in range(N_PROV):
            if i_prov == j_prov:
                cost = 0.0
            else:
                dist = _haversine(*src_coord, *PROV_COORDS[j_prov])
                mult = (IPI_MULT[i_prov] + IPI_MULT[j_prov]) / 2.0
                if CLUSTER[i_prov] == CLUSTER[j_prov]:
                    cost = dist * LAND_RATE * mult
                else:
                    cost = dist * SEA_RATE * mult
            noise = 1.0 + (rng.random() - 0.5) * 2.0 * NOISE
            c_trans[i_prov, j_prov] = cost * noise

    return c_ship, c_dist, c_trans
