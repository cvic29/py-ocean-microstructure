'''
CV 2026/10/05 : process microstructure dataset from L1 (physical units) to L2 (cleaned), 
                following nomenclature in Lueck et al. (2024). 
                Prints refer to sections in Lueck et al.
                contact : clement.vic@ifremer.fr  
'''
import numpy as np
from netCDF4 import Dataset
import gsw 
import scipy.signal as sig
from despike import despike
from yaml_functions import add_to_yaml 
import datetime 

def process_L1_to_L2(param):  
    # --- Parameters that are used here but not saved in the netcdf file 
    Nbut = 1  # order of Butterworth filters 

    # --- Read data 
    if param.chatty: print(' ------> Read L1 data ') 
    nc = Dataset(param.file_nc,'r')
    # --> global dimensions and attributes 
    lat_min    = nc.geospatial_lat_min
    lat_max    = nc.geospatial_lat_max
    fs_fast    = round(nc.fs_fast)
    n_shear    = nc.dimensions['N_SHEAR_SENSORS'].size
    n_acc      = nc.dimensions['N_ACC_SENSORS'].size
    nt         = nc.dimensions['TIME'].size
    # --> L1_converted variables 
    time       = nc.groups['L1_converted'].variables['TIME'][:] # this is strictly equal to time in L2  
    time_units = nc.groups['L1_converted'].variables['TIME'].units
    shear_L1   = nc.groups['L1_converted'].variables['SHEAR'][:]
    pres       = nc.groups['L1_converted'].variables['PRES'][:]
    acc_L1     = nc.groups['L1_converted'].variables['ACC'][:]
    nc.close() 
 
    if param.chatty: print(' ------> 3.2.1 Section selection ')
    lat_avg   = 0.5*(lat_min+lat_max)
    z         = gsw.z_from_p(pres,lat_avg)
    w         = abs(np.diff(z)*fs_fast)         # fall speed 
    Wn_lp     = param.despike_sh[1]*(2/fs_fast) # non-dimensional low-pass cutoff frequency 
    b_lp,a_lp = sig.butter(Nbut,Wn_lp,btype='lowpass',output='ba')
    w_lp      = sig.filtfilt(b_lp,a_lp,w,method='gust') # low-pass w before section selection    
    #time      = (time-time[0])*86400 # [second]
    # --- Find beginning of the section 
    t0 = 0
    while pres[t0]<param.profile_min_P: t0+=1
    while w_lp[t0]<param.profile_min_W: t0+=1
    # --- Find end of the section 
    t1 = w.shape[0]-1
    while w_lp[t1]<param.profile_min_W: t1-=1
    # --- Define section_number, sort of mask   
    section_number = np.ones(nt).astype(int)
    section_number[:t0] = 0
    section_number[t1:] = 0
   
    if param.chatty: print(' ------> 3.2.2 Cleaning shear-probe data ') 
    shear_L2              = np.copy(shear_L1)
    despike_pass_count_sh = np.zeros((n_shear)) # number of passes in the despiking routine for shear  
    acc_L2                = np.copy(acc_L1)
    despike_pass_count_aa = np.zeros((n_acc))   # number of passes in the despiking routine for acceleration 

    for i in range(n_shear):
        if param.superchatty: print('         ---> shear signal #%.i'%(i+1))
        [shear_L2[i,t0:t1],_,despike_pass_count_sh[i],despike_fraction] = despike(shear_L1[i,t0:t1],param,var='shear')
        if param.superchatty: print('              %.i passes in the despiking routine '%despike_pass_count_sh[i])
        if param.superchatty: print('              %.2f percent of data have been replaced '%(100*despike_fraction))
    for i in range(n_acc):
        if param.superchatty: print('         ---> acceleration signal #%.i'%(i+1))
        [acc_L2[i,t0:t1],_,despike_pass_count_aa[i],despike_fraction] = despike(acc_L1[i,t0:t1],param,var='acc')
        if param.superchatty: print('              %.i passes in the despiking routine '%despike_pass_count_aa[i])
        if param.superchatty: print('              %.2f percent of data have been replaced '%(100*despike_fraction))

    if param.chatty: print(' ------> [extra section] compute despike_fraction_sh and despike_fraction_aa ') 
    # Note : this section is detailed in L4 processing in Lueck et al. (2024) 
    #        but I think it actually belongs here as we need shear despiked but not high-passed 
    #        (section 3.2.3 down below) to estimate the fraction of despiked data 
    
    # n_estimates is the number of segments over which statistics based on spectra will be computed 
    n_good_data       = section_number[section_number==1].shape[0]
    n_estimates       = int(1 + np.floor((n_good_data - param.diss_length) / (param.diss_length-param.overlap)))
    param.n_good_data = n_good_data      # save it for L3 processing   
    param.n_estimates = n_estimates # save it for L3 processing   
    # also make a copy in the parameters' file 
    add_to_yaml(param.param_file_name, {
        "n_good_data" : n_good_data, 
        "n_estimates" : n_estimates
    }) 
    
    despike_fraction_sh = np.zeros((n_shear,n_estimates)) # percentage of modified data for shear 
    despike_fraction_aa = np.zeros((n_acc,n_estimates))   # percentage of modified data for acceleration / vibration 
    
    shear_L1_tmp = shear_L1[:,section_number==1] # extract data that will be used to compute dissipation 
    shear_L2_tmp = shear_L2[:,section_number==1]
    acc_L1_tmp   = acc_L1[:,section_number==1] # extract data that will be used to compute dissipation 
    acc_L2_tmp   = acc_L2[:,section_number==1]
 
    # --- Loop over segments 
    select = np.arange(param.diss_length).astype(int) # first bunch of indices over which performing statistics 
    counter = 0
    while select[-1] < n_good_data:
        for s in range(n_shear):
            diff = abs(shear_L2_tmp[s,select] - shear_L1_tmp[s,select])
            despike_fraction_sh[s,counter] = np.count_nonzero(diff) / param.diss_length
            diff = abs(acc_L2_tmp[s,select] - acc_L1_tmp[s,select])
            despike_fraction_aa[s,counter] = np.count_nonzero(diff) / param.diss_length
        select = select + param.diss_length - param.overlap
        counter += 1

    if param.chatty: print(' ------> 3.2.3 High-pass filter time series ')
    Wn_hp      = param.HP_cut*(2/fs_fast)  # non-dimensional high-pass cutoff frequency 
    b_hp,a_hp  = sig.butter(Nbut,Wn_hp,btype='highpass',output='ba')
    for i in range(n_shear):
        shear_L2[i,:] = sig.filtfilt(b_hp,a_hp,shear_L2[i,:],method='gust')
    for i in range(n_acc):
        acc_L2[i,:] = sig.filtfilt(b_hp,a_hp,acc_L2[i,:],method='gust')    
 
    if param.chatty: print(' ------> Save data in netcdf file ')
    # --- Open file 
    nc = Dataset(param.file_nc,'r+')
    
    # --- Copy global attributes
    param_dict = vars(param).copy()  # Convert namespace to dictionary
    # Remove keys that are not global attributes
    keys_to_remove = ['param_file_name', 'file_nc', 'chatty', 'superchatty']
    for key in keys_to_remove:
        param_dict.pop(key, None)
    # Copy attributes 
    for key, value in param_dict.items():
        try:
            nc.setncattr(key, str(value) if isinstance(value, datetime.date) else value)
        except TypeError:
            # Handle non-serializable types (e.g., inf, custom objects)
            ncfile.setncattr(key, str(value))

    # --- Create the L2_cleaned group 
    if 'L2_cleaned' not in nc.groups: 
        group = nc.createGroup('L2_cleaned')
    
    # --- Create the L4_dissipation group 
    if 'L4_dissipation' not in nc.groups: 
        group = nc.createGroup('L4_dissipation') 
        if 'TIME_SPECTRA' not in nc.dimensions: 
            nc.createDimension('TIME_SPECTRA',n_estimates)     

    # --- Define L2 variables and their attributes   
    group = nc.groups['L2_cleaned']

    if 'TIME' not in group.variables: 
        nc_time = group.createVariable('TIME', 'f8', ('TIME',))
        nc_time.standard_name = "time"
        nc_time.units = time_units
        nc_time.axis = "T"
        nc_time.long_name = "Decimal day"
        nc_time.comment = "1 January noon is day 0.5."    
    else: nc_time = group.variables['TIME']

    if 'SHEAR' not in group.variables: 
        nc_shear = group.createVariable('SHEAR', 'f8', ('N_SHEAR_SENSORS', 'TIME'))
        nc_shear.standard_name = "sea_water_velocity_shear"
        nc_shear.units = "s-1"
        nc_shear.long_name = "rate of change of cross axis sea water velocity along transect measured by shear probes"
        nc_shear.comment = "de-spiked and high-pass filtered"
    else: nc_shear = group.variables['SHEAR']

    if 'PSPD_REL' not in group.variables: 
        nc_pspd_rel = group.createVariable('PSPD_REL', 'f8', ('TIME',))
        nc_pspd_rel.standard_name = "platform_speed_wrt_sea_water"
        nc_pspd_rel.units = "m s-1"
        nc_pspd_rel.long_name = "Platform speed with respect to sea water"
    else: nc_pspd_rel = group.variables['PSPD_REL']

    if 'SECTION_NUMBER' not in group.variables: 
        nc_section_number = group.createVariable('SECTION_NUMBER', 'i4', ('TIME',))
        nc_section_number.standard_name = "unique_identifier_for_each_section_of_data_from_timeseries"
        nc_section_number.units = "1"
        nc_section_number.long_name = "A unique identifier counter defining sections of the time series from the fast channels extracted for dissipation estimates"
    else: nc_section_number = group.variables['SECTION_NUMBER']

    if 'ACC' not in group.variables: 
        nc_acc = group.createVariable('ACC', 'f8', ('N_ACC_SENSORS', 'TIME'))
        nc_acc.standard_name = "platform_acceleration"
        nc_acc.units = "m s-2"
        nc_acc.long_name = "platform acceleration detected by accelerometers"
        nc_acc.comment = "de-spiked and high-pass filtered"
    else: nc_acc = group.variables['ACC']
    
    # --- L4: Define variables and their attributes   
    group = nc.groups['L4_dissipation'] 

    if 'DESPIKE_FRACTION_SH' not in group.variables: 
        nc_despike_fraction_sh = group.createVariable('DESPIKE_FRACTION_SH','f8',('N_SHEAR_SENSORS','TIME_SPECTRA'))
        nc_despike_fraction_sh.standard_name = "fraction_of_shear_data_modified_by_despiking_algorithm"
        nc_despike_fraction_sh.units = "1" 
        nc_despike_fraction_sh.long_name = "fraction of data, within a segment, that was modified by the despiking algorithm for the shear probes"
    else: nc_despike_fraction_sh = group.variables['DESPIKE_FRACTION_SH']
    
    if 'DESPIKE_FRACTION_AA' not in group.variables: 
        nc_despike_fraction_aa = group.createVariable('DESPIKE_FRACTION_AA','f8',('N_ACC_SENSORS','TIME_SPECTRA'))
        nc_despike_fraction_aa.standard_name = "fraction_of_vibration_data_modified_by_despiking_algorithm"
        nc_despike_fraction_aa.units = "1" 
        nc_despike_fraction_aa.long_name = "fraction of vibration or acceleration data, within a segment, that was modified by the despiking algorithm"
    else: nc_despike_fraction_aa = group.variables['DESPIKE_FRACTION_AA']

    if 'DESPIKE_PASS_COUNT_SH' not in group.variables: 
        nc_despike_pass_count_sh = group.createVariable('DESPIKE_PASS_COUNT_SH','f8',('N_SHEAR_SENSORS', 'TIME_SPECTRA'))
        nc_despike_pass_count_sh.standard_name = "number_of_despike_passes_for_shear_probes"
        nc_despike_pass_count_sh.units = "1"
        nc_despike_pass_count_sh.long_name = "the number of passes ran with despiking routine to obtain a clean times series of shear probes. One value per section per probe"
    else: nc_despike_pass_count_sh = group.variables['DESPIKE_PASS_COUNT_SH']

    # --- Copy variables 
    nc_time[:]                  = time   
    nc_shear[:]                 = shear_L2
    nc_pspd_rel[:]              = np.concatenate(([0],w_lp)) 
    nc_section_number[:]        = section_number   
    nc_acc[:]                   = acc_L2
    nc_despike_fraction_sh[:]   = despike_fraction_sh  
    nc_despike_fraction_aa[:]   = despike_fraction_aa  
    nc_despike_pass_count_sh[:] = np.tile(despike_pass_count_sh[:,np.newaxis],(1,n_estimates))  
    
    nc.close()
    return 
