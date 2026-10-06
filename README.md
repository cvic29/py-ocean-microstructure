# py-ocean-microstructure

Set of Python routines to process microstructure shear and temperature data from
oceanic platforms (e.g. Rockland VMP profilers), and compute turbulent
dissipation rates following the ATOMIX group recommendations (Lueck et al., 2024).

## What it does

The package processes microstructure data through successive quality levels,
from raw L1 netCDF files to final dissipation estimates (L4):

- **L1 → L2**: conversion and initial conditioning of shear and temperature data
  (`export_L1_mat_to_nc.py`, `process_L1_to_L2.py`)
- **L2 → L3**: despiking, spectral analysis, and turbulence variable estimation
  (`despike.py`, `spectral_analysis.py`, `process_L2_to_L3.py`)
- **L3 → L4**: final dissipation rate (ε) and temperature variance dissipation (χ)
  estimates (`process_L3_to_L4.py`)
- Ancillary utilities: seawater viscosity (`viscosity.py`), parameter file handling
  (`yaml_functions.py`), plotting (`plot_figure_a_la_Fer.py`)

Processing is controlled by plain-text YAML parameter files
(e.g. `parameters_kaseaope3.yaml`, `parameters_swotalis3.yaml`), which also carry
the recommended global attributes / metadata for the output netCDF files
(Table 7 in Fer et al., 2024).

## Requirements

- Python 3
- numpy
- netCDF4
- PyYAML
- matplotlib (for plotting)

## Usage

1. Copy/edit a parameter file (YAML) for your campaign and station.
2. Edit the file paths and parameter file name in `process_microstructure_data.py`.
3. Run:

   ```bash
   python process_microstructure_data.py
