# Optimasi Rantai Pasok Kedelai Indonesia dengan ALNS

Tugas Akhir — Alief Randiansyah Pradanaputra (5002221172)

Repositori ini berisi model optimasi rantai pasok kedelai Indonesia menggunakan
Adaptive Large Neighborhood Search (ALNS)

## Struktur

```
├── *.py + alns/          # Python source code & ALNS shim
├── docs/                 # Dashboard interaktif (deploy ke GitHub Pages)
├── thesis/               # LaTeX (Buku TA, Paper POMITS, Bibliography)
├── results/
│   ├── single/           # Single-run optimization output
│   ├── baseline/         # Baseline experiment (3 konfigurasi × 50 seed)
│   └── scenarios/        # Scenario stress-test (16 skenario × 10 seed)
├── *.xlsx / *.csv        # Data input (impor, demand, koordinat, dll.)
├── requirements.txt
└── README.md
```

## Instalasi

```powershell
pip install -r requirements.txt
```

## Penggunaan

```powershell
# Single run (500 iterasi, silent=False → generate CSV/PNG)
python run_optimization.py

# Build dashboard data & buka docs/index.html
python build_dashboard.py

# Baseline experiment (3 konfigurasi × 50 seed)
python run_experiments_multi.py

# Scenario stress-test (16 skenario × 10 seed)
python run_scenarios_best_model.py
```


## Konfigurasi

`config.py` adalah satu-satunya file yang perlu diedit untuk mengubah parameter
run (MAX_ITER, SA_COOLING, PENALTY_SCALE, operator toggles, dll.).
