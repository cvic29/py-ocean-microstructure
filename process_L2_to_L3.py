'''
CV 2026/10/05 : process microstructure dataset from L2 (cleaned) to L3 (spectra), 
                following nomenclature in Lueck et al. (2024). 
                Prints refer to sections in Lueck et al.
                contact : clement.vic@ifremer.fr  
'''
import numpy as np
from netCDF4 import Dataset
import scipy.interpolate as itp 
import scipy.signal      as signal 
import scipy.optimize    as opt
from spectral_analysis import compute_spectrum,clean_shear_spec
import gsw as gsw  
import microstructure_processing_BFC_v3 as microp 
import matplotlib.pyplot as plt   

def process_L2_to_L3(param):
    # --- Read data 
    if param.chatty: print(' ------> Read L2 data ')
    nc = Dataset(param.file_nc,'r')
    # --> global dimensions and attributes
    n_shear   = nc.dimensions['N_SHEAR_SENSORS'].size
    n_temp    = nc.dimensions['N_TEMP_SENSORS'].size 
    n_acc     = nc.dimensions['N_ACC_SENSORS'].size
    if param.compute_chi: 
        lon = nc.longitude_profile 
        lat = nc.latitude_profile 
    # --> L1_converted and L2_cleaned variables 
    pres            = nc.groups['L1_converted'].variables['PRES'][:]
    temp_ctd        = np.squeeze(nc.groups['L1_converted'].variables['TEMP_CTD'][:])
    time            = nc.groups['L2_cleaned'].variables['TIME'][:]
    time_units      = nc.groups['L2_cleaned'].variables['TIME'].units 
    time_slow       = nc.groups['L1_converted'].variables['TIME_SLOW'][:]
    shear           = nc.groups['L2_cleaned'].variables['SHEAR'][:]
    acc             = nc.groups['L2_cleaned'].variables['ACC'][:]
    pspd_rel        = nc.groups['L2_cleaned'].variables['PSPD_REL'][:]
    section_number  = nc.groups['L2_cleaned'].variables['SECTION_NUMBER'][:]
    if param.compute_chi: 
        temp        = nc.groups['L1_converted'].variables['TEMP'][:] 
        grad_temp   = nc.groups['L1_converted'].variables['GRAD_TEMP'][:] 
        salt        = nc.groups['L1_converted'].variables['SALINITY'][:] 
    nc.close()

    # --- Extract good data for fast channels 
    time      = time[section_number==1]
    pres      = pres[section_number==1]
    shear     = shear[:,section_number==1]
    acc       = acc[:,section_number==1]
    pspd_rel  = pspd_rel[section_number==1] 
    if param.compute_chi: 
        temp      = temp[:,section_number==1] 
        grad_temp = grad_temp[:,section_number==1] 
        salt      = salt[section_number==1] 

    # --- Define variables 
    n_freq        = int(1 + np.floor(param.fft_length/2)) # size of frequency and wavenumber vectors 
    sh_spec       = np.zeros((n_shear,n_freq,param.n_estimates))
    acc_spec      = np.zeros((n_acc,n_freq,param.n_estimates))
    sh_spec_clean = np.zeros((n_shear,n_freq,param.n_estimates))
    sh_acc_spec   = np.zeros((n_shear*n_acc,n_freq,param.n_estimates))
    time_L3       = np.zeros((param.n_estimates)) 
    pres_L3       = np.zeros((param.n_estimates)) 
    pspd_rel_L3   = np.zeros((param.n_estimates)) 
    temp_L3       = np.zeros((param.n_estimates)) 
    kcyc          = np.zeros((n_freq,param.n_estimates)) 
    if param.compute_chi: 
        grad_temp_spec             = np.zeros((n_temp,n_freq,param.n_estimates)) # raw spectra  
        grad_temp_spec_clean       = np.zeros((n_temp,n_freq,param.n_estimates)) # spectra corrected for frequency response  
        grad_temp_spec_noise       = np.zeros((n_temp,n_freq,param.n_estimates)) 
        grad_temp_spec_noise_clean = np.zeros((n_temp,n_freq,param.n_estimates)) 

    # --- Loop over segments 
    if param.chatty: print(' ------> Loop over segments [3.3.1 to 3.3.5] ')
    select = np.arange(param.diss_length).astype(int) # first bunch of indices over which performing statistics 
    counter = 0
    while select[-1] < param.n_good_data:
        if param.chatty:      print('         ---> segment %i/%i'%(counter+1,param.n_estimates))
        
        # --- Compute spectra 
        if param.superchatty: print('              ---> 3.3.1 Spectral calculation ')
        sh_spec_tmp,F,_,_ = compute_spectrum(shear[:,select], n_fft=param.fft_length,\
                                             rate=param.fs_fast, detrend_method='linear')
        for s in range(n_shear): sh_spec[s,:,counter] = sh_spec_tmp[s,s,:] 
        
        # --- Convert frequency to wavenumber spectra 
        W = np.nanmean(pspd_rel[select])
        K = F/W
        kcyc[:,counter] = np.copy(K) # save it in L3 variables         
        sh_spec[:,:,counter] *= W        
 
        # --- Correction for the spatial response 
        if param.superchatty: print('              ---> 3.3.2 Correction for the spatial response ')  
        K0 = 50. # [cpm] Eq (14) in Lueck et al. (2024) [from Macoun and Lueck (2004)]; note that K0=48 in ODAS   
        H = np.copy(K)
        H[K>150]  = 1 # wavenumber correction stops beyond 150 cpm 
        H[K<=150] = 1./(1+(K[K<=150]/K0)**2)
        sh_spec[:,:,counter] /= H 
        
        # --- Correction for the high-pass filter  
        if param.superchatty: print('              ---> 3.3.3 Correction for high-pass filter ')  
        # Note : cannot find it in get_diss_odas (Ilker Fer email 13th Feb 2025) 
        H_HP = np.copy(F)
        H_HP[0] = 0
        H_HP[1:] = (1 + (param.HP_cut/F[1:])**2)**2
        sh_spec[:,:,counter] *= H_HP 

        # --- Vibration-coherent noise removal (Goodman)  
        if param.superchatty: print('              ---> 3.3.4 Vibration-coherent noise removal ')  
        sh_spec_clean_tmp,F,acc_spec_tmp,_,sh_acc_spec_tmp =\
            clean_shear_spec(shear[:,select], acc[:,select],n_fft=param.fft_length, rate=param.fs_fast)
        for s in range(n_shear): sh_spec_clean[s,:,counter] = sh_spec_clean_tmp[s,s,:]
        for s in range(n_acc):   acc_spec[s,:,counter]      = acc_spec_tmp[s,s,:]
        sh_acc_spec[:,:,counter] = sh_acc_spec_tmp.reshape(n_shear*n_acc,n_freq) 

        sh_spec_clean[:,:,counter] *= W # convert frequency spectrum to wavenumber spectrum         
        sh_spec_clean[:,:,counter] /= H
        sh_spec_clean[:,:,counter] *= H_HP
        
        # --- Compute uncertainty parameters (only once), [Eqs (17) and (18) in Lueck et al. (2024)]     
        if counter == 0: 
            if param.superchatty: print('              ---> 3.3.5 Uncertainty of a shear spectrum ')
            # Note: make the assumption that parameter 'overlap' in compute_spectrum is equal to fft_length//2
            #       (defaults, not really interesting to tweak)
            #       n_fft_segments  = Nf in Lueck et al. (2024)  
            #       n_vib_sensors   = Nv in Lueck et al. (2024), number of vibration sensors used in 3.3.4
            n_fft_segments = (param.diss_length-param.fft_length//2) // (param.fft_length-param.fft_length//2)  
            n_vib_sensors  = n_acc # if using all vibration sensors   
            spec_std       = np.sqrt((5./4.)*(n_fft_segments - n_vib_sensors)**(-7./9.))
   
        # --- Compute other variables' mean over segment 
        time_L3[counter]     = np.nanmean(time[select]) 
        pres_L3[counter]     = np.nanmean(pres[select]) 
        pspd_rel_L3[counter] = np.nanmean(pspd_rel[select])
        # - for temperature, first interpolate onto fast timeline so that we can use 'select' indices  
        itp_slow_to_fast     = itp.interp1d(time_slow,temp_ctd,
                                            kind='linear',fill_value='extrapolate') # nearest if out of range  
        temp_ctd_itp         = itp_slow_to_fast(time.data)  
        temp_L3[counter]     = np.nanmean(temp_ctd_itp[select]) 
 
        if param.compute_chi: 
            if param.superchatty: print('              ---> [...] Compute temperature gradient spectra ')
            # This contains 3 steps : 
            #      1. Compute temperature gradient spectra 
            #      2. Compute noise spectrum    
            #      3. Correct spectra for the frequency response of the probes 
            salt_ctd = np.nanmean(salt[select]) # this should be salinity measured by the CTD  
            SA_ctd = gsw.SA_from_SP(salt_ctd,pres_L3[counter],lon,lat)  
            CT_ctd = gsw.CT_from_t(SA_ctd,temp_L3[counter],pres_L3[counter]) 
            for tid in range(n_temp):
                # --- 1. Compute temperature gradient spectra 
                f,grad_temp_spec[tid,:,counter] = signal.welch(grad_temp[tid,select],fs=param.fs_fast,window='hann',
                                                               nperseg=param.fft_length,noverlap=param.fft_length//2, 
                                                               nfft=None, detrend='constant')   
                # --- 2. Compute noise spectrum 
                if param.fit_noise_model: # -> first option is noise fitting   
                    spec_tmp = np.copy(grad_temp_spec[tid,1:,counter])  
                    spec_tmp = spec_tmp[f[1:]<0.7*param.f_AA] 
                    imin     = np.nanargmin(spec_tmp)
                    fmin     = np.max([10,f[imin+1]])
                    ii       = (f>0) & (f<fmin)
                    #chi_grad_temp = 6.*1.4e-7*np.trapz(grad_temp_spec[tid,ii,counter], f[ii]) / pspd_rel_L3[counter]**2 
                
                    if fmin<0.7*param.f_AA:
                        ii = (f>fmin) & (f<param.f_AA) 
                        popt, pcov = opt.curve_fit(microp.FP07noise0, f[ii], grad_temp_spec[tid,ii,counter])    
                        np0[t,:] = popt[0]  
                        np1[t,:] = popt[1]
                        grad_temp_spec_noise[tid,:,counter] = microp.FP07noise(f, popt)
 
                else: # -> second option is simpler : use a pre-fitted noise model 
                    noise_model_p1 = -10.2 
                    noise_model_p2 = -0.8330 
                    grad_temp_spec_noise[tid,:,counter] = microp.FP07noise0(f, noise_model_p1, noise_model_p2)*(2*np.pi*f)**2
              
                # --- 3. Correct spectra for the frequency response of the probes
                # NB1 : for details, see Eq (3) in Piccolroaz et al. (2021)  
                # NB2 : this is duplicate from part of the function thermistor_fit used at L4 
                #       but we thought it'd be best including this step here at L3.  
                F0  = 25*np.sqrt(W)
                tau = (2*np.pi*F0/np.sqrt(np.sqrt(2)-1))**(-1)
                H = 1./(1 + (2*np.pi*tau*f)**2)**2 # double-pole correction (single or double could be an option in the future)
                grad_temp_spec_clean[tid,:,counter]       = grad_temp_spec[tid,:,counter] / H 
                grad_temp_spec_noise_clean[tid,:,counter] = grad_temp_spec_noise[tid,:,counter] / H 

                # - Convert dT/dt frequency spectra [K^2 s^-2 Hz^-1] to dT/dz frequency spectra [K^2 m^-2 Hz^-1]             
                grad_temp_spec[tid,:,counter]             /= W**2  
                grad_temp_spec_clean[tid,:,counter]       /= W**2  
                grad_temp_spec_noise[tid,:,counter]       /= W**2  
                grad_temp_spec_noise_clean[tid,:,counter] /= W**2  
 
                # - Finally, convert frequency spectra [K^2 m^-2 Hz^-1] to wavenumber spectra [K^2 m^-2 cpm^-1]
                grad_temp_spec[tid,:,counter]             *= W  
                grad_temp_spec_clean[tid,:,counter]       *= W  
                grad_temp_spec_noise[tid,:,counter]       *= W  
                grad_temp_spec_noise_clean[tid,:,counter] *= W  

        # --- Update counter and segment onto which processing is performed 
        # Note : overlap in compute_spectrum is different from param.overlap, 
        #        which is the overlap between dissipation estimates  
        select = select + param.diss_length - param.overlap
        counter += 1

    if 0: # quick check on gradT spectra  
        plt.figure()
        for counter in [10,100,200]: 
            plt.loglog(f,grad_temp_spec[0,:,counter],'b',lw=0.5) # /!\ units might not be ok for grad_temp_spec  
            plt.loglog(f,grad_temp_spec_noise[0,:,counter],'orange',lw=0.5)
        plt.axvline(param.f_AA, color = 'r') 
        plt.ylim(1e-11,1e-2) 
        plt.savefig('tmp_spec.pdf') 


    if param.chatty: print(' ------> Save data in netcdf file ')
    nc = Dataset(param.file_nc,'r+')
    # --- Create the L3_spectra group 
    if 'L3_spectra' not in nc.groups:
        group = nc.createGroup('L3_spectra')
    
    # --- Create missing dimensions 
    if 'N_GLOBAL_VALUES' not in nc.dimensions:
        nc.createDimension('N_GLOBAL_VALUES',1) 
    if 'N_WAVENUMBER' not in nc.dimensions:
        nc.createDimension('N_WAVENUMBER',n_freq) 
    if 'N_SH_ACC_SPEC' not in nc.dimensions: 
        nc.createDimension('N_SH_ACC_SPEC',n_shear*n_acc) 

    # --- Define L3 variables and their attributes   
    group = nc.groups['L3_spectra']    
    
    if 'N_FFT_SEGMENTS' not in group.variables: 
        nc_n_fft_segments = group.createVariable('N_FFT_SEGMENTS', 'i4', ('N_GLOBAL_VALUES')) 
        nc_n_fft_segments.standard_name = "number_of_fft_segments" 
        nc_n_fft_segments.units = "1"
        nc_n_fft_segments.long_name = "Number of FFT segments used in each spectrum estimate"
    else: nc_n_fft_segments = group.variables['N_FFT_SEGMENTS'] 
    
    if 'N_VIB_SENSORS' not in group.variables: 
        nc_n_vib_sensors = group.createVariable('N_VIB_SENSORS', 'i4', ('N_GLOBAL_VALUES')) 
        nc_n_vib_sensors.standard_name = "number_of_vibration_sensors_used_for_cleaning_spectra" 
        nc_n_vib_sensors.units = "1" 
        nc_n_vib_sensors.long_name = "Number of vibration or acceleration sensors used for cleaning of shear spectrum with the Goodman algoritm"
    else: nc_n_vib_sensors = group.variables['N_VIB_SENSORS']

    if 'SPEC_STD' not in group.variables: 
        nc_spec_std = group.createVariable('SPEC_STD', 'f8', ('N_GLOBAL_VALUES')) 
        nc_spec_std.standard_name = "standard_deviation_uncertainty_of_shear_spectrum"
        nc_spec_std.units = "1" 
        nc_spec_std.long_name = "statistical uncertainty (standard deviation) of the natural logarithm of spectrum of shear"
    else: nc_spec_std = group.variables['SPEC_STD']

    if 'TIME' not in group.variables: 
        nc_time = group.createVariable('TIME', 'f8', ('TIME_SPECTRA'))    
        nc_time.standard_name = "time" 
        nc_time.units = time_units
        nc_time.axis = "T"
        nc_time.long_name = "Decimal day"
        nc_time.comment = "1 January noon is day 0.5." 
    else: nc_time = group.variables['TIME']

    if 'PRES' not in group.variables: 
        nc_pres = group.createVariable('PRES', 'f8', ('TIME_SPECTRA')) 
        nc_pres.standard_name = "sea_water_pressure" 
        nc_pres.units = "decibar" 
        nc_pres.long_name = "Sea water pressure, equals 0 at sea-level" 
    else: nc_pres = group.variables['PRES']
   
    if 'SH_SPEC' not in group.variables: 
        nc_sh_spec = group.createVariable('SH_SPEC', 'f8', ('N_SHEAR_SENSORS', 'N_WAVENUMBER', 'TIME_SPECTRA'))
        nc_sh_spec.standard_name = "shear_probe_spectrum" 
        nc_sh_spec.units = "s-2 cpm-1" 
        nc_sh_spec.long_name = "cyclic wavenumber spectrum of sea water velocity shear" 
    else: nc_sh_spec = group.variables['SH_SPEC'] 
 
    if 'KCYC' not in group.variables: 
        nc_kcyc = group.createVariable('KCYC', 'f8', ('N_WAVENUMBER', 'TIME_SPECTRA')) 
        nc_kcyc.standard_name = "cyclic_wavenumber"
        nc_kcyc.units = "cpm"
        nc_kcyc.long_name = "wavenumber along the direction of instrument motion in cycles per meter"
    else: nc_kcyc = group.variables['KCYC']

    if 'PSPD_REL' not in group.variables: 
        nc_pspd_rel = group.createVariable('PSPD_REL', 'f8', ('TIME_SPECTRA')) 
        nc_pspd_rel.standard_name = "platform_speed_wrt_sea_water"
        nc_pspd_rel.units = "m s-1"
        nc_pspd_rel.long_name = "Platform speed with respect to sea water" 
    else: nc_pspd_rel = group.variables['PSPD_REL']
        
    if 'SECTION_NUMBER' not in group.variables: 
        nc_section_number = group.createVariable('SECTION_NUMBER', 'i4', ('TIME_SPECTRA')) 
        nc_section_number.standard_name = "unique_identifier_for_each_section_of_data_from_timeseries"
        nc_section_number.units = "1"
        nc_section_number.long_name = "A unique indentifier counter defining sections of the time series extracted for dissipation estimates" 
    else: nc_section_number = group.variables['SECTION_NUMBER'] 

    if 'TEMP' not in group.variables: 
        nc_temp = group.createVariable('TEMP', 'f8', ('TIME_SPECTRA')) 
        nc_temp.standard_name = "sea_water_temperature" 
        nc_temp.units = "degree_Celsius" 
        nc_temp.long_name = "sea water temperature in-situ ITS-90 scale"
    else: nc_temp = group.variables['TEMP']
 
    if 'SH_SPEC_CLEAN' not in group.variables: 
        nc_sh_spec_clean = group.createVariable('SH_SPEC_CLEAN', 'f8', ('N_SHEAR_SENSORS', 'N_WAVENUMBER', 'TIME_SPECTRA'))
        nc_sh_spec_clean.standard_name = "shear_probe_spectrum_clean"
        nc_sh_spec_clean.units = "s-2 cpm-1"
        nc_sh_spec_clean.long_name = "cleaned cyclic wavenumber spectrum of sea water velocity shear"
    else: nc_sh_spec_clean = group.variables['SH_SPEC_CLEAN'] 

    if 'ACC_SPEC' not in group.variables: 
        nc_acc_spec = group.createVariable('ACC_SPEC', 'f8', ('N_ACC_SENSORS', 'N_WAVENUMBER', 'TIME_SPECTRA')) 
        nc_acc_spec.standard_name = "acceleration_sensor_spectrum"
        nc_acc_spec.units = "m2 s-4 cpm-1"
        nc_acc_spec.long_name = "acceleration spectrum from accelerometers" 
    else: nc_acc_spec = group.variables['ACC_SPEC'] 

    if 'SH_ACC_SPEC' not in group.variables: 
        nc_sh_acc_spec = group.createVariable('SH_ACC_SPEC', 'f8', ('N_SH_ACC_SPEC', 'N_WAVENUMBER', 'TIME_SPECTRA'))
        nc_sh_acc_spec.standard_name = "shear_and_acceleration_cross-spectral_matrix"    
        nc_sh_acc_spec.units = "1"    
        nc_sh_acc_spec.long_name = "Complex cross-spectral matrix of shear and acceleration"     
    else: nc_sh_acc_spec = group.variables['SH_ACC_SPEC'] 

    if param.compute_chi: 
        if 'GRAD_TEMP_SPEC' not in group.variables: 
            nc_grad_temp_spec = group.createVariable('GRAD_TEMP_SPEC','f8',('N_TEMP_SENSORS', 'N_WAVENUMBER', 'TIME_SPECTRA')) 
            nc_grad_temp_spec.standard_name = "grad_temp_spectrum" 
            nc_grad_temp_spec.units = "degree_Celsius2 m-2 cpm-1" 
            nc_grad_temp_spec.long_name = "cyclic wavenumber spectrum of sea water temperature vertical gradient"  
        else: nc_grad_temp_spec = group.variables['GRAD_TEMP_SPEC']

        if 'GRAD_TEMP_SPEC_CLEAN' not in group.variables: 
            nc_grad_temp_spec_clean = group.createVariable('GRAD_TEMP_SPEC_CLEAN','f8',('N_TEMP_SENSORS', 'N_WAVENUMBER', 'TIME_SPECTRA')) 
            nc_grad_temp_spec_clean.standard_name = "grad_temp_spectrum_clean" 
            nc_grad_temp_spec_clean.units = "degree_Celsius2 m-2 cpm-1" 
            nc_grad_temp_spec_clean.long_name = "cleaned (corrected) cyclic wavenumber spectrum of sea water temperature vertical gradient"  
        else: nc_grad_temp_spec_clean = group.variables['GRAD_TEMP_SPEC_CLEAN']
    
        if 'GRAD_TEMP_SPEC_NOISE' not in group.variables: 
            nc_grad_temp_spec_noise = group.createVariable('GRAD_TEMP_SPEC_NOISE','f8',('N_TEMP_SENSORS', 'N_WAVENUMBER', 'TIME_SPECTRA')) 
            nc_grad_temp_spec_noise.standard_name = "grad_temp_spectrum_noise" 
            nc_grad_temp_spec_noise.units = "degree_Celsius2 m-2 cpm-1" 
            nc_grad_temp_spec_noise.long_name = "cyclic wavenumber noise spectrum of sea water temperature vertical gradient"  
        else: nc_grad_temp_spec_noise = group.variables['GRAD_TEMP_SPEC_NOISE']
        
        if 'GRAD_TEMP_SPEC_NOISE_CLEAN' not in group.variables: 
            nc_grad_temp_spec_noise_clean = group.createVariable('GRAD_TEMP_SPEC_NOISE_CLEAN','f8',('N_TEMP_SENSORS', 'N_WAVENUMBER', 'TIME_SPECTRA')) 
            nc_grad_temp_spec_noise_clean.standard_name = "grad_temp_spectrum_noise_clean" 
            nc_grad_temp_spec_noise_clean.units = "degree_Celsius2 m-2 cpm-1" 
            nc_grad_temp_spec_noise_clean.long_name = "cleaned (corrected) cyclic wavenumber noise spectrum of sea water temperature vertical gradient"  
        else: nc_grad_temp_spec_noise_clean = group.variables['GRAD_TEMP_SPEC_NOISE_CLEAN']
    
    # --- Copy variables 
    nc_n_fft_segments[:] = n_fft_segments  
    nc_n_vib_sensors[:]  = n_vib_sensors
    nc_spec_std[:]       = spec_std 
    nc_time[:]           = time_L3     
    nc_pres[:]           = pres_L3   
    nc_sh_spec[:]        = sh_spec
    nc_kcyc[:]           = kcyc  
    nc_pspd_rel[:]       = pspd_rel_L3 
    nc_section_number[:] = np.ones((param.n_estimates)) # not sure about the relevance 
    nc_temp[:]           = temp_L3
    nc_sh_spec_clean[:]  = sh_spec_clean
    nc_acc_spec[:]       = acc_spec 
    nc_sh_acc_spec[:]    = sh_acc_spec 
    if param.compute_chi: 
        nc_grad_temp_spec[:]             = grad_temp_spec 
        nc_grad_temp_spec_clean[:]       = grad_temp_spec_clean  
        nc_grad_temp_spec_noise[:]       = grad_temp_spec_noise
        nc_grad_temp_spec_noise_clean[:] = grad_temp_spec_noise_clean   
 
    nc.close()
    return 
