import json
import os
from pathlib import Path


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


notebook["cells"].append(md(r"""
# Optimasi Rantai Pasok Kedelai Indonesia dengan ALNS

Notebook ini adalah versi eksekutabel dari model rantai pasok kedelai pada kode saat ini. Alurnya:

1. memuat instance `Problem` dari data lokal,
2. menjelaskan formulasi matematika yang digunakan solver,
3. menjalankan ALNS dengan jumlah iterasi yang bisa dikendalikan dari environment,
4. mengevaluasi hasil dan artefak JSON,
5. membaca keluaran eksperimen multi-run bila tersedia.

Jumlah iterasi notebook dikendalikan oleh environment variable `NOTEBOOK_MAX_ITER`; defaultnya `200`. Untuk smoke test cepat:

```powershell
$env:NOTEBOOK_MAX_ITER=5
jupyter nbconvert --to notebook --execute Optimasi_Rantai_Pasok_Kedelai.ipynb --output C:\tmp\soybean_smoke.ipynb
```
"""))


notebook["cells"].append(md(r"""
## 1. Environment dan Data

Himpunan indeks yang dipakai:

| Simbol | Arti |
|---|---|
| $I$ | provinsi tujuan permintaan dan asal/tujuan transfer, indeks $i,j$ |
| $K$ | provinsi produsen lokal yang lolos ambang produksi, indeks $k$ |
| $H$ | pelabuhan masuk impor, indeks $h$ |
| $S$ | negara asal impor, indeks $s$ |
| $T$ | periode bulanan, indeks $t$ |

Data utama tersimpan di `Problem`: permintaan, kapasitas produksi, kapasitas throughput pelabuhan, kapasitas gudang provinsi, biaya, dan matriks kelayakan transfer. Pelabuhan diperlakukan sebagai pipa throughput, bukan titik penyimpanan.
"""))


notebook["cells"].append(code(r"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from problem import Problem
from solver import SoybeanALNSSolver
from result_artifacts import export_optimization_result, load_optimization_result

plt.style.use("ggplot")

NOTEBOOK_MAX_ITER = int(os.environ.get("NOTEBOOK_MAX_ITER", "200"))
DATA_DIR = Path(".")
OUTPUT_DIR = Path(config.OUTPUT_DIR)
ARTIFACT_PATH = OUTPUT_DIR / "optimization_result.json"

problem = Problem.load(data_dir=str(DATA_DIR), seed=config.SEED)
print(f"NOTEBOOK_MAX_ITER = {NOTEBOOK_MAX_ITER}")
print(f"Problem: {problem.N_PROV} provinsi, {problem.N_PORT} pelabuhan, {problem.N_PERIOD} bulan")
print(f"Demand tahunan nasional: {problem.DEMAND.sum():,.0f} ton")
print(f"Kapasitas produksi lokal tahunan: {problem.PROD_CAP.sum():,.0f} ton")
print(f"Rasio kapasitas lokal/demand: {problem.PROD_CAP.sum()/problem.DEMAND.sum():.2%}")
"""))


notebook["cells"].append(md(r"""
## 2. Eksplorasi Data

Permintaan nasional sekitar jutaan ton per tahun, sementara kapasitas produksi lokal hanya sekitar 10% dari kebutuhan. Ini membuat batas import dependency `EPS_IMPORT_DEP = 0.90` sangat ketat secara fisik: bila produksi lokal dipakai penuh, minimum rasio impor kira-kira

$$
\frac{\text{impor}}{\text{impor}+\text{lokal}} \approx 90\%.
$$

Karena penyebutnya adalah total suplai eksternal yang tersedia dari impor dan lokal, bukan total demand, shortage tidak boleh membuat rasio impor terlihat lebih baik secara palsu.
"""))


notebook["cells"].append(code(r"""
months = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

fig, ax = plt.subplots(figsize=(10, 4))
national_monthly = problem.DEMAND.sum(axis=0)
ax.bar(months, national_monthly, color="#4C78A8")
ax.set_title("Permintaan Kedelai Nasional per Bulan")
ax.set_xlabel("Bulan")
ax.set_ylabel("Ton")
for i, value in enumerate(national_monthly):
    ax.text(i, value * 1.01, f"{value:,.0f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.show()
"""))


notebook["cells"].append(code(r"""
annual_demand = problem.DEMAND.sum(axis=1)
annual_prod = np.zeros(problem.N_PROV)
for k, prov_idx in enumerate(problem.PROD_IDX):
    annual_prod[prov_idx] = problem.PROD_CAP[k].sum()

df_prov = pd.DataFrame({
    "Provinsi": problem.PROV_NAMES,
    "Demand": annual_demand,
    "Kapasitas Lokal": annual_prod,
    "Defisit Sebelum Impor/Transfer": annual_demand - annual_prod,
}).sort_values("Demand", ascending=False)

display(df_prov.head(15))

fig, ax = plt.subplots(figsize=(11, 6))
top = df_prov.head(15).set_index("Provinsi")[["Demand", "Kapasitas Lokal"]]
top.plot(kind="barh", ax=ax, color=["#4C78A8", "#54A24B"])
ax.invert_yaxis()
ax.set_title("15 Provinsi dengan Permintaan Tahunan Terbesar")
ax.set_xlabel("Ton")
plt.tight_layout()
plt.show()
"""))


notebook["cells"].append(code(r"""
port_df = pd.DataFrame({
    "Pelabuhan": problem.PORT_NAMES,
    "Throughput Tahunan": problem.PORT_THRU_CAP.sum(axis=1),
    "Melayani Provinsi": [len(problem.PORT_SERV[h]) for h in range(problem.N_PORT)],
}).sort_values("Throughput Tahunan", ascending=False)

display(port_df)
"""))


notebook["cells"].append(md(r"""
## 3. Formulasi Matematika

### Parameter

| Simbol | Definisi |
|---|---|
| $D_{it}$ | demand kedelai provinsi $i$ bulan $t$ |
| $P_{kt}$ | kapasitas produksi lokal produsen $k$ bulan $t$ |
| $U^N_{st}, U^E_{st}$ | kapasitas impor normal dan emergency dari negara $s$ bulan $t$ |
| $Q_{ht}$ | kapasitas throughput pelabuhan $h$ bulan $t$ |
| $G_{it}$ | kapasitas gudang provinsi $i$ bulan $t$ |
| $B_{it}$ | safety stock minimum provinsi $i$ bulan $t$ |
| $a_{hi}$ | 1 bila pelabuhan $h$ boleh melayani provinsi $i$ |
| $w_{ij}$ | 1 bila transfer dari $i$ ke $j$ layak secara jaringan |
| $c^\cdot$ | biaya produksi, pembelian impor, distribusi, transfer, inventory, dan fixed activation |

### Variabel keputusan

| Variabel | Definisi |
|---|---|
| $x^{loc}_{kit}\ge 0$ | pengiriman produksi lokal dari produsen $k$ ke provinsi $i$ bulan $t$ |
| $x^{imp}_{sht}\ge 0$ | impor normal dari negara $s$ melalui pelabuhan $h$ bulan $t$ |
| $x^{emg}_{sht}\ge 0$ | impor emergency dari negara $s$ melalui pelabuhan $h$ bulan $t$ |
| $x^{dist}_{hit}\ge 0$ | distribusi dari pelabuhan $h$ ke provinsi $i$ bulan $t$ |
| $x^{trns}_{ijt}\ge 0$ | transfer antar-provinsi dari $i$ ke $j$ bulan $t$ |
| $I_{it}\ge 0$ | inventory akhir provinsi $i$ bulan $t$ |
| $Sh_{it}\ge 0$ | shortage provinsi $i$ bulan $t$ |
| $u_{it}\in\{0,1\}$ | indikator safety stock terpenuhi |
| $y_{it}\in\{0,1\}$ | indikator aktivasi provinsi |
| $z_t\in\{0,1\}$ | indikator emergency import bulan $t$ |
"""))


notebook["cells"].append(md(r"""
### Objective decomposition

Solver memisahkan biaya murni $Z_{cost}$ dan penalti e-constraint:

$$
\min \; F(x)=Z_{cost}(x)+M_1v_1+M_2v_2+M_3v_3+M_4v_4.
$$

Biaya murni:

$$
\begin{aligned}
Z_{cost} =
&\sum_{k,i,t}(c^{prod}_{k}+c^{ship}_{ki})x^{loc}_{kit}
+\sum_{s,h,t}c^{imp}_{s}x^{imp}_{sht}
+\sum_{s,h,t}c^{emg}_{s}x^{emg}_{sht}\\
&+\sum_{h,i,t}c^{dist}_{hi}x^{dist}_{hit}
+\sum_{i,j,t}c^{trns}_{ij}x^{trns}_{ijt}
+\sum_{i,t}h_i I_{it}
+\sum_{i,t}f^{act}_{i}y_{it}
+\sum_t f^{emg}z_t.
\end{aligned}
$$

Empat violation term:

$$
v_1=\max(0,\sum_{i,t}Sh_{it}-\epsilon_1)
$$

$$
v_2=\max\left(0,\frac{\sum_{s,h,t}(x^{imp}_{sht}+x^{emg}_{sht})}
{\sum_{s,h,t}(x^{imp}_{sht}+x^{emg}_{sht})+\sum_{k,i,t}x^{loc}_{kit}}-\epsilon_2\right)
$$

$$
v_3=\max(0,\epsilon_3-\sum_{k,i,t}x^{loc}_{kit})
$$

$$
v_4=\max(0,\epsilon_4-\sum_{i,t}I_{it})
$$

`EPS_IMPORT_DEP = 0.90` tidak boleh diturunkan tanpa mengubah data produksi, karena kapasitas lokal nasional berada tepat di sekitar 10% demand tahunan.
"""))


notebook["cells"].append(md(r"""
### Constraints utama

Flow balance provinsi dengan inventory carry-over:

$$
I_{i,t-1}+\sum_k x^{loc}_{kit}+\sum_h x^{dist}_{hit}+\sum_j x^{trns}_{jit}
-\sum_j x^{trns}_{ijt}+Sh_{it}=D_{it}+I_{it}.
$$

Kapasitas produksi lokal:

$$
\sum_i x^{loc}_{kit}\le P_{kt}\quad\forall k,t.
$$

Keseimbangan throughput pelabuhan, karena pelabuhan bukan storage:

$$
\sum_s (x^{imp}_{sht}+x^{emg}_{sht})=\sum_i x^{dist}_{hit}\quad\forall h,t.
$$

Kapasitas throughput pelabuhan:

$$
\sum_s (x^{imp}_{sht}+x^{emg}_{sht})\le Q_{ht}\quad\forall h,t.
$$

Kapasitas gudang provinsi dan inventory floor:

$$
I_{it}\le G_{it}, \qquad I_{it}\ge B_{it}u_{it}.
$$

Kelayakan pelayanan pelabuhan dan transfer:

$$
x^{dist}_{hit}=0 \text{ bila } a_{hi}=0,\qquad
x^{trns}_{ijt}=0 \text{ bila } w_{ij}=0.
$$

Aktivasi biner menggunakan ambang besar/kecil yang disimpan di `Problem.X_THRESHOLD`:

$$
\sum_k x^{loc}_{kit}+\sum_h x^{dist}_{hit}+\sum_j x^{trns}_{jit}\le M_i y_{it}.
$$

Emergency import hanya aktif jika bulan tersebut diaktifkan:

$$
\sum_{s,h}x^{emg}_{sht}\le M^E_t z_t.
$$
"""))


notebook["cells"].append(md(r"""
## 4. Normalized AUGMECON Penalty Calibration

Penalty multiplier tidak di-hardcode. Setelah solusi awal dibuat, solver menghitung:

$$
M_k=\text{PENALTY\_SCALE}\times \frac{z_0}{r_k}.
$$

Dengan:

| Multiplier | Range $r_k$ | Makna |
|---|---|---|
| `M_SHORTAGE` | total demand | penalti shortage per ton |
| `M_IMPORT_DEP` | 1.0 | penalti rasio import dependency |
| `M_LOCAL` | `EPS_LOCAL_MIN` | penalti produksi lokal di bawah floor |
| `M_INV_FLOOR` | `INV_MIN_FRAC * total_demand` | penalti inventory floor |

Skala ini membuat setiap violation dibaca sebagai pelanggaran relatif, bukan angka mentah yang kebetulan besar karena satuannya ton atau rupiah.
"""))


notebook["cells"].append(code(r"""
config_snapshot = {
    "EPS_SHORTAGE": config.EPS_SHORTAGE,
    "EPS_IMPORT_DEP": config.EPS_IMPORT_DEP,
    "INV_MIN_FRAC": config.INV_MIN_FRAC,
    "PENALTY_SCALE": config.PENALTY_SCALE,
    "SA_COOLING": config.SA_COOLING,
    "SA_PAPER": config.SA_PAPER,
    "SA_REDU": config.SA_REDU,
    "USE_TABU": config.USE_TABU,
    "TABU_TENURE": config.TABU_TENURE,
    "SUB_IT": config.SUB_IT,
}
pd.Series(config_snapshot)
"""))


notebook["cells"].append(md(r"""
## 5. ALNS, SA, dan Tabu Search

Setiap iterasi ALNS:

1. memilih destroy operator berdasarkan bobot adaptif,
2. memilih repair operator berdasarkan bobot adaptif,
3. semua operator memanggil `feasibility_repair()` agar flow balance, throughput pelabuhan, storage provinsi, emergency import, dan transfer eligibility kembali konsisten,
4. menerima/menolak kandidat dengan simulated annealing,
5. memperbarui skor operator,
6. menjalankan tabu local search pada solusi terbaik jika `USE_TABU=True`.

Mode SA:

| Mode | Pembanding | Catatan |
|---|---|---|
| paper SA | kandidat dibandingkan terhadap $S_{best}$ | mengikuti Eq. 32 paper acuan, tetapi lebih sensitif |
| standard SA | kandidat dibandingkan terhadap $S_{current}$ | konfigurasi C2 multi-run paling robust |

Tabu search saat ini hanya melakukan satu jenis move: mengganti sebagian impor dengan produksi lokal untuk pasangan provinsi-bulan acak. Ia tidak menambah impor dan tidak merutekan ulang flow secara umum; perannya sebagai local intensification yang konservatif.
"""))


notebook["cells"].append(md(r"""
## 6. Jalankan Optimasi

Cell berikut menjalankan solver dengan `NOTEBOOK_MAX_ITER`. Notebook memakai `silent=True` agar smoke test tidak bergantung pada rendering semua PNG, lalu menulis `hasil/optimization_result.json` secara eksplisit. Run normal melalui `python run_optimization.py` tetap memakai `silent=False` dan menulis CSV/PNG serta JSON.
"""))


notebook["cells"].append(code(r"""
solver = SoybeanALNSSolver(
    problem,
    max_iter=NOTEBOOK_MAX_ITER,
    seed=config.SEED,
    use_tabu=config.USE_TABU,
)
result = solver.run(silent=True)
best_state = result["best"]
initial_state = result["initial"]
artifact_path = export_optimization_result(result, out_dir=config.OUTPUT_DIR)

print("Artifact exported:", artifact_path)
"""))


notebook["cells"].append(code(r"""
bd_initial = result["breakdown_initial"]
bd_best = result["breakdown_best"]

def summarize_state(label, state, breakdown):
    total_imp = float(state.x_imp.sum() + state.x_emg.sum())
    total_loc = float(state.x_loc.sum())
    return {
        "Skenario": label,
        "Objective": breakdown["objective"],
        "Z_cost": breakdown["z_cost"],
        "Penalty": breakdown["total_penalty"],
        "Shortage": float(state.sh.sum()),
        "Import Dependency": total_imp / max(total_imp + total_loc, 1.0),
        "Local": total_loc,
    }

summary = pd.DataFrame([
    summarize_state("Initial", initial_state, bd_initial),
    summarize_state("Best", best_state, bd_best),
])
display(summary)
"""))


notebook["cells"].append(code(r"""
hist = pd.DataFrame({
    "iteration": np.arange(len(result["history_best"])),
    "best_objective": result["history_best"],
    "z_cost": result["history_z_cost"],
    "penalty": result["history_penalty"],
})

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hist["iteration"], hist["best_objective"], label="Best objective", color="#4C78A8")
ax.set_title("Konvergensi ALNS")
ax.set_xlabel("Iterasi")
ax.set_ylabel("Objective")
ax.legend()
plt.tight_layout()
plt.show()
"""))


notebook["cells"].append(code(r"""
total_imp = float(best_state.x_imp.sum() + best_state.x_emg.sum())
total_loc = float(best_state.x_loc.sum())
import_dependency = total_imp / max(total_imp + total_loc, 1.0)

eps_status = pd.DataFrame([
    {"Constraint": "Shortage", "Value": float(best_state.sh.sum()), "Limit": config.EPS_SHORTAGE,
     "Status": "OK" if best_state.sh.sum() <= config.EPS_SHORTAGE + 0.5 else "VIOLATED"},
    {"Constraint": "Import dependency", "Value": import_dependency, "Limit": config.EPS_IMPORT_DEP,
     "Status": "OK" if import_dependency <= config.EPS_IMPORT_DEP + 1e-6 else "VIOLATED"},
    {"Constraint": "Local production floor", "Value": total_loc, "Limit": config.EPS_LOCAL_MIN,
     "Status": "OK" if total_loc >= config.EPS_LOCAL_MIN - 0.5 else "VIOLATED"},
    {"Constraint": "Inventory floor", "Value": float(best_state.inv.sum()),
     "Limit": config.INV_MIN_FRAC * float(problem.DEMAND.sum()), "Status": "diagnostic"},
])
display(eps_status)
"""))


notebook["cells"].append(md(r"""
## 7. JSON Artifact dan Dashboard

`SoybeanALNSSolver.run(silent=False)` menghasilkan `hasil/optimization_result.json` berisi:

- metadata schema, waktu pembuatan, seed, iterasi, dan snapshot konfigurasi,
- metadata problem: dimensi, nama provinsi/pelabuhan/importir, `PROD_IDX`, `PORT_SERV`, `CLUSTER`, `DEMAND`, `PORT_THRU_CAP`,
- metrik hasil: history objective, biaya, penalti, operator counts, bobot akhir, shortage diagnostic, elapsed time,
- array lengkap state awal dan terbaik: `x_loc`, `x_imp`, `x_emg`, `x_dist`, `x_trns`, `inv`, `sh`, `safe`, `w`, `y`, `z`.

Dashboard dibuat terpisah:

```powershell
python build_dashboard.py
```

Script tersebut hanya membaca JSON dan menulis `dashboard/data.js`.
"""))


notebook["cells"].append(code(r"""
if ARTIFACT_PATH.exists():
    artifact = load_optimization_result(ARTIFACT_PATH)
    print("Schema:", artifact["schema_version"])
    print("Created:", artifact["created_at"])
    print("Run:", artifact["run"])
    print("History length:", len(artifact["metrics"]["history_best"]))
    print("Best arrays:", sorted(artifact["states"]["best"].keys()))
else:
    print("Artifact belum tersedia. Jalankan cell optimasi atau python run_optimization.py.")
"""))


notebook["cells"].append(md(r"""
## 8. Hasil Multi-Run

Eksperimen multi-run saat ini memakai `run_experiments_multi.py` dan menyimpan hasil agregat di:

- `multi_hasil/summary.csv`
- `multi_hasil/all_runs.csv`
- `multi_hasil/operator_history.csv`
- `multi_hasil/summary.json`

Multi-run sengaja tidak menyimpan dump state per seed karena ukurannya besar dan tidak ramah version control.
"""))


notebook["cells"].append(code(r"""
summary_path = Path("multi_hasil/summary.csv")
all_runs_path = Path("multi_hasil/all_runs.csv")
operator_history_path = Path("multi_hasil/operator_history.csv")

if summary_path.exists():
    multi_summary = pd.read_csv(summary_path)
    display(multi_summary)
else:
    print("Belum ada multi_hasil/summary.csv. Jalankan python run_experiments_multi.py bila diperlukan.")

if all_runs_path.exists():
    all_runs = pd.read_csv(all_runs_path)
    cols = ["config", "seed", "z_cost", "shortage", "import_dep", "service_rate", "elapsed_seconds"]
    display(all_runs[[c for c in cols if c in all_runs.columns]].head())

if operator_history_path.exists():
    op_hist = pd.read_csv(operator_history_path)
    display(op_hist.groupby(["config", "outcome"]).size().unstack(fill_value=0))
"""))


notebook["cells"].append(md(r"""
## 9. Catatan Interpretasi

- Port throughput adalah batas arus masuk/keluar per bulan; tidak ada inventory pelabuhan.
- Import dependency dihitung sebagai `imports / (imports + local)`.
- Emergency import mengikuti variabel tiga indeks melalui pelabuhan, yaitu $x^{emg}_{sht}$.
- `EPS_IMPORT_DEP = 0.90` berada di batas fisik karena produksi lokal hanya sekitar 10% kebutuhan nasional.
- Penalti AUGMECON dinormalisasi otomatis; bila hasil tidak stabil, diagnosis utama adalah temperatur SA, desain operator, atau local search, bukan hardcode multiplier.
- Konfigurasi multi-run yang paling stabil dalam catatan repo adalah C2: standard SA, geometric cooling, tabu aktif.
"""))


out_path = Path(__file__).resolve().parent / "Optimasi_Rantai_Pasok_Kedelai.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
    f.write("\n")

print(f"Notebook created: {out_path}")
