# py-ocean-microstructure

Set of Python routines to process microstructure shear and temperature data from
oceanic platforms (e.g. Rockland VMP profilers), and compute turbulent
dissipation rates following the ATOMIX group recommendations (Lueck et al., 2024).
The computation of temperature variance dissipation mostly follows Piccolroaz et al. (2021). 

## What it does

The package processes microstructure data through successive quality levels,
from raw L1 netCDF files to final dissipation estimates (L4):

- It considers that the data extraction from device and conversion into physical 
  units is done by proprietary software (e.g., Zissou Essential for Rockland 
  instruments). Use `export_L1_mat_to_nc.py` to convert L1 variables from a mat
  file to a netCDF file with the standard variables (Fer et al., 2024).  
- **L1 → L2**: initial conditioning of shear and temperature data, despiking and  
  filtering (`process_L1_to_L2.py`)
- **L2 → L3**: spectral analysis, and correction for platform acceleration contamination
  (`process_L2_to_L3.py`)
- **L3 → L4**: estimate dissipation rate (ε) and temperature variance dissipation (χ)
  and quality flags (`process_L3_to_L4.py`)
- Estimate of final is less straightforward than as there is not yet standard processing, 
  we propose a quality flag and algorithm to define it (`estimate_chi_final_and_flags.py`)
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

## References 

Fer, I., Dengler, M., Holtermann, P., Le Boyer, A., & Lueck, R. (2024). ATOMIX benchmark datasets for dissipation rate measurements using shear probes. Scientific Data, 11(1), 518.

Lueck, R., Fer, I., Bluteau, C., Dengler, M., Holtermann, P., Inoue, R., ... & Stevens, C. (2024). Best practices recommendations for estimating dissipation rates from shear probes. Frontiers in Marine Science, 11, 1334327.

Piccolroaz, S., Fernández-Castro, B., Toffolon, M., & Dijkstra, H. A. (2021). A multi-site, year-round turbulence microstructure atlas for the deep perialpine Lake Garda. Scientific Data, 8(1), 188.
