# City Network Pathogen Transmission Simulation

Modular simulation framework for pathogen genotype dynamics across heterogeneous city networks. Models transmission of bacterial clades (e.g., *Acinetobacter baumannii*) with distinct transmission modes: outpatient colonization vs inpatient infection. Supports basic simulation, invasion scenario, and COVID-19 three-phase intervention.

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
- `--hospitals N`, `--metros M`: network size (default: 100 cities, 35 metro)
- `--generations G`: simulation length (default: 1000)
- `--seed S`: random seed (default: 42)
- `--metro-migration`, `--peripheral-migration`: patient transfer rates reflecting O/I ratio differences
- `--output-dir PATH`: output directory (default: `outputs/`)

**Invasion specific:**
- `--introduction-time T`: generation at which Clade 2.5 is introduced
- `--introduction-dose D`: initial frequency (0.0-1.0, default: 0.05)
- `--introduction-sites N`: number of metro cities used for introduction (default: 3)

**COVID specific:**
- `--intervention-start T1`: start of the mobility-intervention phase
- `--recovery-start T2`: start of the recovery phase
- `--metro-closure`, `--periph-closure`: migration reduction ratios
- `--baseline-decay`, `--intervention-decay`, `--recovery-decay`: distance decay per phase (km)

## Outputs

**Data files** (`outputs/run_*/data/`):
- `timeline.csv`: time series of clade frequencies
- `network_state.csv`: final spatial distribution
- `transfer_matrix*.csv`: patient transfer networks between cities
- `network_analysis.json`: network metrics, degree statistics and genotype parameters (`TAU_VALUES`, `BETA_VALUES`)
- `run_config.json`: complete parameter record for the run

**Figures** (`outputs/run_*/figures/`):
- `*.pdf`: static plots of trajectories, networks and spatial spread
- `*.html`: interactive Plotly visualizations

Filenames include timestamp and run parameters: `run_YYYYMMDD_HHMMSS_{scenario}_h{hospitals}_m{metros}_g{generations}_s{seed}`

## Model Overview

### Network and parameterization

The model operates on a synthetic city network. Nodes are assigned metropolitan
or peripheral healthcare structure, and directed mobility links follow a
distance-decay function with an upward-referral bias from peripheral to
metropolitan nodes. The metropolitan/peripheral contrast is motivated by the
observed outpatient-to-inpatient ratios. Genotype-specific colonisation and
infection parameters then determine selection within nodes and dispersal among
nodes. This design tests how hierarchical mobility and opposing transmission
strategies influence the maintenance of spatially structured ecotypes.

Wright-Fisher population genetics + spatial network migration:

1. **Genotypes**: 8 genotypes with τ-values (outpatient colonization capacity) and β-values (inpatient infection capacity)
   - τ > 0.6: metro-oriented (Clade 2.5), with high dispersal via mobile outpatients
   - τ < 0.4: peripheral-oriented (Clade 2.4), with low dispersal through immobilized inpatients
   - G0 (τ=0.5): Neutral background (susceptible population)

2. **Selection**: Genotypes have differential fitness based on city healthcare structure
   - Metro cities (high O/I ratio): favor high-τ genotypes (outpatient colonizers remain mobile)
   - Peripheral cities (low O/I ratio): favor high-β genotypes (inpatient infections immobilize patients)
   - Fitness reflects transmission advantage in local healthcare environment

3. **Migration**: Patient transfers between cities with spatial decay and hierarchical bias
   - Transfer probability ∝ exp(-distance / decay_length)
   - Hierarchical flow: peripheral → metro cities (upward referral bias)
   - Migration rates reflect O/I ratio differences

4. **Scenarios**:
   - **Basic**: Both clades compete from start
   - **Invasion**: Clade 2.4 establishes first, then Clade 2.5 introduced
   - **COVID**: Three-phase intervention tests clade-specific resilience

> **Figure: model overview and scenarios**
> Add `model_summary.png` here, combining the synthetic city network, lineage-frequency trajectories and the basic, invasion and three-phase mobility-intervention scenarios.

Example sensitivity outputs are provided as
[`sensitivity_analysis.pdf`](example_res/sensitivity_results/sensitivity_analysis.pdf),
[`fst_diversity_analysis.pdf`](example_res/sensitivity_results/fst_diversity_analysis.pdf)
and [`coexistence_stability.pdf`](example_res/sensitivity_results/coexistence_stability.pdf).

## Requirements

```bash
pip install numpy pandas matplotlib seaborn plotly networkx
```

The same package list is provided in `requirements.txt`.

## Notes

- `config.py` contains all default parameters.
- `--fixed-genotypes` reuses the same τ/β values across runs; otherwise the seed controls their random generation.
- The COVID scenario requires `0 < intervention_start < recovery_start < generations`.
- The invasion scenario requires `0 < introduction_time < generations`.
