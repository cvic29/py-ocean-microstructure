'''
CV 2025/10/05 : compute the rate of energy dissipation from microstructure shear measurements 
                following Lueck et al. (2024). This script calls successive functions to process 
                data at levels L2, L3 and L4. 
                --> edit parameters.yaml to use the parameters you want. 
                Note : when param.chatty (sometimes param.superchatty), 
                       'x.y.z' correspond to section x.y.z in Lueck et al. (2024) 
'''
import numpy as np
from netCDF4 import Dataset
from yaml_functions import load_param 
from process_L1_to_L2 import process_L1_to_L2
from process_L2_to_L3 import process_L2_to_L3
from process_L3_to_L4 import process_L3_to_L4
import warnings; warnings.filterwarnings('ignore')
import os 

# ---> Select files 

# - KASEAOPE-3 
#station         = 'STY3'
#path_files      = r'../DATA/VMP/'+station+'/'
#param_file      = 'parameters_kaseaope3.yaml' 

# - SWOTALIS-3 
station         = 'U2PO'  
path_files      = r'../../../2023_Swotalis/Data/VMP_by_stations/'+station+'/'
param_file      = 'parameters_swotalis3.yaml' 

files           = os.listdir(path_files) 
#list_file_nc    = [f for f in files if f.endswith('.nc')] # original files (no gradT) 
list_file_nc    = [f for f in files if f.endswith('gradT.nc')] # modified with gradT to get chi  
list_file_nc.sort()  # Sorts alphabetically

#list_file_nc = [list_file_nc[0]] # test w/ 1 file  
#list_file_nc = ['VMP_011_Profile2_gradT.nc'] # MUN0 (quiet)
#list_file_nc = ['VMP_010_Profile5_gradT.nc'] # MUN2 (energetic) 
#list_file_nc = ['VMP_040_Profile1_gradT.nc'] # MUN1 (energetic) 
#list_file_nc = ['VMP_160_Profile2_gradT.nc'] 

# ---> Processing parameters and levels  
processing_levels = ['L2','L3','L4'] 
#processing_levels = ['L4'] 

# ---> Load parameters 
param = load_param(param_file)

# ---> Loop on files 
for file_nc in list_file_nc: 
    print(' ###################################################################################### ')
    print(f' ################### Processing {file_nc} ')  
    print(' ###################################################################################### ')
    param.file_nc = path_files+file_nc 

    if 'L2' in processing_levels: 
        print(' ') 
        print(' ========================================================================= ') 
        print(f' === process {file_nc} from L1 (converted) to L2 (cleaned) ') 
        print(' ========================================================================= ') 
        process_L1_to_L2(param) 
 
    if 'L3' in processing_levels:
        print(' ') 
        print(' ========================================================================= ') 
        print(f' === process {file_nc} from L2 (cleaned) to L3 (spectra) ') 
        print(' ========================================================================= ') 
        process_L2_to_L3(param) 

    if 'L4' in processing_levels:
        print(' ') 
        print(' ========================================================================= ') 
        print(f' === process {file_nc} from L3 (spectra) to L4 (dissipation) ') 
        print(' ========================================================================= ') 
        process_L3_to_L4(param) 

