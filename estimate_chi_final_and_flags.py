'''
CV 2026/09/18 : This routine determines final chi from chi estimated with the two temperature (and shear) probes. 
                It uses an algorithm that uses epsilon_max (from thermistor_fit) 
                A single quality metrics (R) is defined to qualify chi_final
                variables CHI_FINAL and CHI_FLAGS are saved in netcdf file  

contact : clement.vic@ifremer.fr and B.Fernandez-Castro@soton.ac.uk   
''' 
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
import matplotlib.gridspec as gridspec
from netCDF4 import Dataset
import scipy.stats as stats 
import os
import re
from microstructure_processing_BFC_v5 import Tspec

# ---> Select files 

# - KASEAOPE-3 
#station         = 'STY3'
#path_files      = r'../DATA/VMP/'+station+'/'

# - SWOTALIS-3 
station         = '007'
path_files      = r'../../../2023_Swotalis/Data/VMP_by_stations/'+station+'/'

files           = os.listdir(path_files)
list_file_nc    = [f for f in files if f.endswith('gradT.nc')] # modified with gradT to get chi  
list_file_nc.sort()  # Sorts alphabetically
#list_file_nc    = ['VMP_011_Profile2_gradT.nc'] # MUN0 (quiet) 

# ---> Loop on files 
for file_nc in list_file_nc:
    print(' ###################################################################################### ')
    print(f' ################### Processing {file_nc} ')
    print(' ###################################################################################### ')
    
    print(' ------> Read dataset ')
    nc = Dataset(path_files+file_nc,'r+')
    pres        = nc.groups['L3_spectra'].variables['PRES'][:]
    mad_T       = nc.groups['L4_dissipation'].variables['MAD_T'][:]
    mad_ST      = nc.groups['L4_dissipation'].variables['MAD_ST'][:]
    mad_crit    = nc.groups['L4_dissipation'].variables['MAD_critical'][:]
    lkh_ratio   = nc.groups['L4_dissipation'].variables['LKH_RATIO'][:]
    kB_T        = nc.groups['L4_dissipation'].variables['KB_T'][:] # Batchelor wavenumber from MLE fit (temp only) 
    kB_S        = nc.groups['L4_dissipation'].variables['KB_S'][:] # Batchelor wavenumber from epsilon from shear  
    eps_S       = nc.groups['L4_dissipation'].variables['EPSI_FINAL'][:]
    chi_T       = nc.groups['L4_dissipation'].variables['CHI_T'][:]
    chi_ST      = nc.groups['L4_dissipation'].variables['CHI_ST'][:]
    mad_T       = nc.groups['L4_dissipation'].variables['MAD_T'][:]
    mad_ST      = nc.groups['L4_dissipation'].variables['MAD_ST'][:]
    mad_crit    = nc.groups['L4_dissipation'].variables['MAD_critical'][:]
    lr_T        = nc.groups['L4_dissipation'].variables['LKH_RATIO'][:]
    eps_T_max   = nc.groups['L4_dissipation'].variables['EPSI_T_MAX'][:]

    mad_crit = mad_crit[0,0] # actually constant with probes and depth 
    lr_crit  = 2  

    print(' ------> Process quality metrics') 
    # - determine which flags are relevant 
    mad = np.nan*np.copy(mad_ST) # MAD to consider to qualify chi 
    lr  = np.nan*np.copy(lr_T)   # LR is only relevant when using T method for chi 
    R   = np.nan*np.copy(chi_T)  # Quality flag that takes into account the whole algorithm 
    chi_best = np.nan*np.copy(chi_T) # for each probe, best chi between chi_T and chi_ST and nan  
    
    for i in range(eps_S.shape[0]): # loop on individual estimates 
        if eps_S[i] > eps_T_max[i]: # epsilon from shear is larger than max epsilon diagnosable by MLE so use the ST method 
            for t in range(2): # loop on temp sensors  
                mad[t,i] = mad_ST[t,i]
                if mad_ST[t,i] < mad_crit: 
                    R[t,i] = 0 # good 
                    chi_best[t,i] = chi_ST[t,i]  
                else: # quality checks do not pass and we cannot use the MLE method  
                    R[t,i] = 3 # bad 
        else: # case where eps_S=nan or eps_S < eps_T_max : we can use the MLE method  
            for t in range(2): # loop on temp sensors  
                mad[t,i] = mad_T[t,i]
                lr[t,i]  = lr_T[t,i]
                if (mad_T[t,i] < mad_crit) & (lr_T[t,i] > lr_crit): 
                    R[t,i] = 1 # good 
                    chi_best[t,i] = chi_T[t,i]  
                else: # give a chance to the ST method where shear data is used to correct for missing variance  
                    mad[:,i] = mad_ST[:,i] 
                    if mad_ST[t,i] < mad_crit:
                        R[t,i] = 2 # good  
                        chi_best[t,i] = chi_ST[t,i]  
                    else: # both methods fail  
                        R[t,i] = 4 # bad  
    
    R = R.astype(int) 
    # - select final chi 
    chi_final = np.nanmean(chi_best,axis=0) 

    if 0:
        plt.figure()
        plt.plot(chi_final,pres) 
        plt.xscale('log') 
        plt.savefig('tmp.pdf') 

    print(' ------> Save in netcdf file ')   
    group = nc.groups['L4_dissipation'] 
    
    # --- Define L4 variables and their attributes   
    if 'CHI_FINAL' not in group.variables:
        nc_chi_final = group.createVariable('CHI_FINAL', 'f8', ('TIME_SPECTRA'))
        nc_chi_final.standard_name = "temperature_variance_dissipation"
        nc_chi_final.units = "K2 s-1"
        nc_chi_final.long_name = "Best estimate of temperature variance dissipation"
    else: nc_chi_final = group.variables['CHI_FINAL']
    
    if 'CHI_FLAG' not in group.variables:
        nc_chi_flag = group.createVariable('CHI_FLAG','i4',('N_TEMP_SENSORS','TIME_SPECTRA'))
        nc_chi_flag.standard_name = 'temperature_variance_dissipation_qc_flag'
        nc_chi_flag.units = '1' 
    else: nc_chi_flag = group.variables['CHI_FLAG'] 
    nc_chi_flag.long_name = 'quality flag for temperature variance dissipation, defined following a specific algorithm that should be documented in a companion paper' # erase and replace previous text that would not make sense 
    
    # --- Copy variables  
    nc_chi_final[:] = chi_final  
    nc_chi_flag[:]  = R
 
    nc.close()
