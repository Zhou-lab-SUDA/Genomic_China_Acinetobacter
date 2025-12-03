# City Network Pathogen Transmission Simulation

Modular simulation framework for pathogen genotype dynamics across heterogeneous city networks. Models transmission of bacterial clades (e.g., *Acinetobacter baumannii*) with distinct transmission modes: outpatient colonization vs inpatient infection. Supports basic simulation, invasion scenario, and COVID-19 three-phase intervention.

## Recent Updates
### 2025-12-02
- **Major refactor**: Replaced M/C parameters with τ/β transmission parameters
  - `M_VALUES` → `TAU_VALUES` (τ: outpatient colonization capacity)
  - `C_VALUES` → `BETA_VALUES` (β: inpatient infection capacity)
  - Updated all visualizations to use τ/β notation
  - Aligned code terminology with manuscript biological concepts
- **Terminology update**: "Hospital Network" → "City Network" in visualization.html

### 2025-11-29
- **Update sensitivity analysis**: python sensitivity_oneway.py --output-dir sensitivity_results

### 2025-11-18
- **Reorganized outputs**
- **Bug fixes**: Fixed import errors after code cleanup
- **Parameter validation**: Added fail-fast validation for COVID and invasion scenarios (prevents runtime crashes from invalid time parameters)

### 2025-11-12
- **Random genotype generation**: Each seed generates unique τ/β genotype values by default
- **Fixed G0 background**: Genotype 0 (G0) always neutral (τ=β=0.5) as susceptible background
- **COVID three-phase plot**: Added Growth Rate and Resilience Score visualization
- **Flexible genotype mode**: Use `--fixed-genotypes` to revert to config.py fixed values

## Quick Start

Basic simulation:
```bash
python model.py
```

Invasion scenario (Clade 2.5 invades Clade 2.4):
```bash
python model.py invasion --generations 2000 --introduction-time 500
```

COVID-19 intervention (3-phase differential control):
```bash
python model.py covid --generations 1500 --intervention-start 300 --recovery-start 800
```

## Project Structure

```
.
├── model.py              # CLI entry point
├── config.py             # Defaults and parameters
├── core/                 # Simulation engine (fitness, migration, reproduction)
├── scenarios/            # Invasion and COVID scenarios
├── visualization/        # Plotting utilities
├── utils/                # Export and statistics
└── outputs/              # Auto-created results (CSV, JSON, PDF, HTML)
```

## Key Parameters

**Common:**
- `--hospitals N`, `--metros M` — Network size (default: 100 cities, 35 metro)
- `--generations G` — Simulation length (default: 1000)
- `--seed S` — Random seed (default: 42)
- `--metro-migration`, `--peripheral-migration` — Patient transfer rates (reflects O/I ratio differences)
- `--output-dir PATH` — Output directory (default: `outputs/`)

**Invasion specific:**
- `--introduction-time T` — When to introduce Clade 2.5 (must be < generations)
- `--introduction-dose D` — Initial frequency (0.0-1.0, default: 0.05)
- `--introduction-sites N` — Number of metro cities for introduction (default: 3)

**COVID specific:**
- `--intervention-start T1` — Start of lockdown phase (must be < generations)
- `--recovery-start T2` — Start of recovery phase (must be > T1 and < generations)
- `--metro-closure`, `--periph-closure` — Migration reduction ratios
- `--baseline-decay`, `--intervention-decay`, `--recovery-decay` — Distance decay per phase (km)

## Outputs

**Data files** (`outputs/run_*/data/`):
- `timeline.csv` — Time series of clade frequencies
- `network_state.csv` — Final spatial distribution
- `transfer_matrix*.csv` — Patient transfer networks between cities
- `network_analysis.json` — Network metrics, degree statistics, and genotype parameters (TAU_VALUES, BETA_VALUES)
- `run_config.json` — All parameters for reproducibility

**Figures** (`outputs/run_*/figures/`):
- `*.pdf` — Static plots (trajectories, networks, spatial spread)
- `*.html` — Interactive Plotly visualizations

Filenames include timestamp and run parameters: `run_YYYYMMDD_HHMMSS_{scenario}_h{hospitals}_m{metros}_g{generations}_s{seed}`

## Model Overview

Wright-Fisher population genetics + spatial network migration:

1. **Genotypes**: 8 genotypes with τ-values (outpatient colonization capacity) and β-values (inpatient infection capacity)
   - τ > 0.6: Metro-oriented (Clade 2.5) — high dispersal via mobile outpatients
   - τ < 0.4: Peripheral-oriented (Clade 2.4) — low dispersal, immobilized inpatients
   - G0 (τ=0.5): Neutral background (susceptible population)

2. **Selection**: Genotypes have differential fitness based on city healthcare structure
   - Metro cities (high O/I ratio): favor high-τ genotypes (outpatient colonizers remain mobile)
   - Peripheral cities (low O/I ratio): favor high-β genotypes (inpatient infections immobilize patients)
   - Fitness reflects transmission advantage in local healthcare environment

3. **Migration**: Patient transfers between cities with spatial decay and hierarchical bias
   - Transfer probability ∝ exp(-distance / decay_length)
   - Hierarchical flow: peripheral → metro cities (upward referral bias)
   - Migration rates reflect O/I ratio differences (metro: 0.07, peripheral: 0.02)

4. **Scenarios**:
   - **Basic**: Both clades compete from start
   - **Invasion**: Clade 2.4 establishes first, then Clade 2.5 introduced
   - **COVID**: Three-phase intervention tests clade-specific resilience

## Requirements

```bash
pip install numpy pandas matplotlib seaborn plotly networkx
```

Or use `requirements.txt` if available.

## Notes

- All defaults in `config.py`
- Parameter validation prevents runtime crashes (e.g., `intervention_start < generations`)
- Use `--fixed-genotypes` for reproducible τ/β values across runs (otherwise randomized by seed)
- COVID scenario requires 3 valid time points: 0 < intervention_start < recovery_start < generations
- Invasion scenario requires: 0 < introduction_time < generations
