'''
CV 2026/10/05 : process microstructure data from L3 (spectra) to L4 (dissipation and temperature variance dissipation)
                following nomenclature in Lueck et al. (2024). 

contact : clement.vic@ifremer.fr 
''' 
import numpy as np
from netCDF4 import Dataset
from viscosity import visc35
from spectral_analysis import spec_integral,spectral_model
import microstructure_processing_BFC_v5 as microp

def process_L3_to_L4(param):
    # --- Read data 
    if param.chatty: print(' ------> Read L3 data ')
    nc = Dataset(param.file_nc,'r')
    # --> global dimensions and attributes
    n_shear   = nc.dimensions['N_SHEAR_SENSORS'].size
    
    # --> L3_spectra variables 
    section_number  = nc.groups['L3_spectra'].variables['SECTION_NUMBER'][:]
    time            = nc.groups['L3_spectra'].variables['TIME'][:]
    time_units      = nc.groups['L3_spectra'].variables['TIME'].units 
    pres            = nc.groups['L3_spectra'].variables['PRES'][:]
    pspd_rel        = nc.groups['L3_spectra'].variables['PSPD_REL'][:]
    sh_spec_clean   = nc.groups['L3_spectra'].variables['SH_SPEC_CLEAN'][:]
    spec_std        = nc.groups['L3_spectra'].variables['SPEC_STD'][:][0]
    kcyc            = nc.groups['L3_spectra'].variables['KCYC'][:]
    temp            = nc.groups['L3_spectra'].variables['TEMP'][:]   
    if param.compute_chi:
        n_temp                     = nc.dimensions['N_TEMP_SENSORS'].size  
        grad_temp_spec_clean       = nc.groups['L3_spectra'].variables['GRAD_TEMP_SPEC_CLEAN'][:] 
        grad_temp_spec_noise_clean = nc.groups['L3_spectra'].variables['GRAD_TEMP_SPEC_NOISE_CLEAN'][:] 
 
    # --> L4_dissipation variables that were saved in L2 in anticipation of usage here 
    despike_fraction_sh   = nc.groups['L4_dissipation'].variables['DESPIKE_FRACTION_SH'][:]
    despike_pass_count_sh = nc.groups['L4_dissipation'].variables['DESPIKE_PASS_COUNT_SH'][:]
    nc.close()

    param.n_estimates = time.shape[0] 

    # --- Define a few parameters and variables  
    # Viscosity of seawater 
    nu = visc35(temp)
    
    if 0: # CV 2026/01/09 :  quick check using viscosity computed with Zissou Premium 
        from scipy.io import loadmat
        mat  = loadmat(r'../DATA/VMP/MUN0/VMP_001_Profile1.mat')
        data = mat['data'] 
        nu    = np.squeeze(data['viscosity_diss'][0][0])[:-1]
        # -> this gives the same  

    # Parameters to estimate epsilon from epsilon_10, Eq (19) in Lueck et al. 2024
    a = 1.25e-9*nu**-3
    b = 5.5e-8*nu**-2.5

    kSR  = 150.                     # largest wavenumber for spectral integration that 
                                    # correponds to the spatial resolution of the probe 
    klim = param.f_limit/pspd_rel   # wavenumber of irremovable spectral corruption 
    kAA  = 0.9*param.f_AA/pspd_rel  # wavenumber corresponding to the cutoff frequency of the anti-aliasing filter 

    # --- Define variables 
    epsi         = np.zeros((n_shear,param.n_estimates)) # epsilon (dissipation)
    ku           = np.zeros((n_shear,param.n_estimates)) # upper limit wavenumber for integration  
    lnepsi_std   = np.zeros((n_shear,param.n_estimates)) # uncertainty of log(epsilon) 
    var_resolved = np.zeros((n_shear,param.n_estimates)) # fraction of variance resolved by the spectrum   
    N_s          = np.zeros((n_shear,param.n_estimates)) # number of spectral points used for integration 
    MAD          = np.zeros((n_shear,param.n_estimates)) # mean absolute difference between spectrum and model 
    FOM          = np.zeros((n_shear,param.n_estimates)) # Figure of merit, quantifies difference between spectrum and model 
    Q            = np.zeros((n_shear,param.n_estimates)) # Quality-assurance metrics 
    epsi_final   = np.zeros((param.n_estimates))*np.nan  # final epsilon (best estimate considering quality flags) 
    
    if param.compute_chi:
        chi_I         = np.zeros((n_temp,param.n_estimates)) # chi from simple spectral integral  
        chi_T         = np.zeros((n_temp,param.n_estimates)) # chi from MLE fit 
        chi_ST        = np.zeros((n_temp,param.n_estimates))*np.nan # chi using epsilon to account for missing variance 
        kB_T          = np.zeros((n_temp,param.n_estimates)) # Batchelor wavenumber from MLE fit (T only)  
        kB_S          = np.zeros((n_temp,param.n_estimates)) # Batchelor wavenumber from epsilon from shear 
        epsi_T        = np.zeros((n_temp,param.n_estimates)) # epsilon from MLE fit 
        epsi_T_max    = np.zeros((param.n_estimates))        # max epsilon that can be estimated from MLE fit 
        MAD_T         = np.zeros((n_temp,param.n_estimates)) # mean absolute difference between theoretical and actual spectrum for MLE method  
        MAD_ST        = np.zeros((n_temp,param.n_estimates)) # same but for spectra using both shear and temp  			
        MAD_crit      = np.zeros((n_temp,param.n_estimates)) # critical mean absolute difference 
        lkh_ratio     = np.zeros((n_temp,param.n_estimates)) # likelihood ratio 
        chi_flag      = np.zeros((n_temp,param.n_estimates)) # fit flag according to MAD_T and lkh_ratio 
        ik_fit        = np.zeros((n_temp,param.n_estimates,2)) # indices of min and max wavenumbers over which fit is performed 
        pp_fit        = np.zeros((n_temp,param.n_estimates,2)) # polynomial coefficients from fit 

    # --- Loop over segments  
    if param.chatty: print(' ------> Loop over segments [3.4.1 to 3.4.5] ')
    for t in range(param.n_estimates): 
        if param.chatty:      print('         ---> segment %i/%i'%(t+1,param.n_estimates))
        ktmp = np.copy(kcyc[:,t]) 
        idx_k10 = np.searchsorted(ktmp,10)-1 # index for k10 <= 10 cpm 
        # --- Loop on shear probes  
        for s in range(n_shear): 
            
            # --- Spectral integration 
            if param.superchatty: print('              ---> 3.4.1 Estimating epsilon by spectral integration ') 

            # --> Integration to 10 cpm 
            e10  = 7.5*nu[t]*np.trapz(sh_spec_clean[s,:idx_k10+1,t],ktmp[:idx_k10+1])
            e10 += 7.5*nu[t]*ktmp[1]*sh_spec_clean[s,1,t]/4. # correction for trapz at lowest wavenumber (end of 3.4.1)   

            # --> Get a first estimate of epsilon by crude integration and correction 
            epsi1 = e10*(np.sqrt(1+a[t]*e10) + np.exp(-b[t]*e10) - 1)
            k95   = 0.12*(epsi1/nu[t]**3)**0.25

            # --> Fit to spectrum in log-log space to get minimum. Note that the fit is done over [k1, min(kSR,kAA)] 
            #     as these are (so far) the potential existing upper limits of spectrum "validity"  
            kfit = min((kSR,kAA[t],k95)) # CV 2026/01/09 : added k95 as in get_diss_odas.m  
            idx_kfit = np.nanargmin(abs(ktmp-kfit)) 
            if idx_kfit > 1: # need two points for the fit 
                poly_coeffs = np.polyfit(np.log10(ktmp[1:idx_kfit]),\
                                         np.log10(sh_spec_clean[s,1:idx_kfit,t]), param.fit_order)
                spec_fit = np.power(10,np.polyval(poly_coeffs, np.log10(ktmp)))
                try:
                    kmin = ktmp[np.argmin(spec_fit[idx_k10:idx_kfit])+idx_k10]
                except: # if kfit < 10   
                    kmin = 10. 
            else:
                kmin = 10.
            
            # --> ku is the upper limit for spectral integration
            #print(k95,kmin,kSR,kAA[t],klim[t]) 
            ku[s,t] = min((k95,kmin,kSR,kAA[t],klim[t]))
            idx_ku = np.searchsorted(ktmp,ku[s,t])-1
            
            # --> Get a second (refined) estimate of epsilon by integration to ku 
            epsi2 = 7.5*nu[t]*np.trapz(sh_spec_clean[s,:idx_ku+1,t],ktmp[:idx_ku+1]) 
            epsi2 += 7.5*nu[t]*ktmp[1]*sh_spec_clean[s,1,t]/4. # correction for trapz at lowest wavenumber (end of 3.4.1)          

            # --> Get a third estimate of epsilon through refinement of integration boundary 
            #     I_L is the fraction of shear variance resolved by ending 
            #     the spectral integration to finite wavenumber ku (Lueck model) 
            ku_nondim = ku[s,t]*(nu[t]**3/epsi2)**0.25 
            I_L       = spec_integral(ku_nondim)
            epsi3     = epsi2/I_L

            # --> Repeat the operation to get a fourth estimate 
            ku_nondim = ku[s,t]*(nu[t]**3/epsi3)**0.25
            I_L       = spec_integral(ku_nondim)
            epsi_new  = epsi3/I_L

            # --> Iterate the method based on improvements from epsi3
            keep_looping = 1
            counter = 0  
            while keep_looping == 1 and counter < 5:
                ku_nondim = ku[s,t]*(nu[t]**3/epsi_new)**0.25
                I_L       = spec_integral(ku_nondim)
                epsi_old  = np.copy(epsi_new)
                epsi_new  = epsi3/I_L
                if abs(epsi_new-epsi_old)/epsi_old < 0.01: keep_looping = 0
                counter += 1 
            epsi[s,t] = np.copy(epsi_new)

            # --> Compare actual spectrum to modeled spectrum (typically Nasmyth)
            L_K                 = (nu[t]**3/epsi[s,t])**0.25 # Kolmogorov scale  
            k_nondim            = ktmp*L_K
            #spec_nasmyth_nondim = spectral_model(k_nondim,model='nasmyth')
            #spec_nasmyth        = (epsi[s,t]**3/nu[t])**0.25 * spec_nasmyth_nondim
            #spec_lueck22_nondim = spectral_model(k_nondim,model='lueck22')
            #spec_lueck22        = (epsi[s,t]**3/nu[t])**0.25 * spec_lueck22_nondim
            spec_model_nondim   = spectral_model(k_nondim,model=param.spectral_model)           
            spec_model          = (epsi[s,t]**3/nu[t])**0.25 * spec_model_nondim
 
            # --- Compute uncertainty of log(epsilon) 
            if param.superchatty: print('              ---> 3.4.3 Uncertainty of epsilon estimated by spectral integration ') 
            l_epsi = param.diss_length_sec * pspd_rel[t]
            L_f = ( l_epsi / L_K ) * I_L**0.75 # I_L is V_f in Lueck et al. (V_f is general, I_L is using Lueck model)  
            var_resolved[s,t] = I_L
            lnepsi_std[s,t] = np.sqrt(5.5 / ( 1 + (L_f/4)**(7./9) )) # uncertainty of log(epsilon)

            # --- Compute variables that will help to quality-check espilon 
            # FOM : Figure Of Merit quantifies how close the spectrum is to the model (typical <1.4 if the agreement is good)  
            # MAD : Mean Absolute Difference between actual spectrum and model spectrum 
            N_s[s,t] = np.copy(idx_ku) # number of spectral values to estimate epsilon (k=0 is excluded)  
            MAD[s,t] = (1./N_s[s,t])*np.sum(abs(np.log(sh_spec_clean[s,1:idx_ku+1,t]) - np.log(spec_model[1:idx_ku+1])))
            T_M = 0.8 + 1.25 / N_s[s,t]**0.5
            FOM[s,t] = (MAD[s,t] / spec_std) * (1./T_M)

        # --- Compute quality-assurance metrics   
        if param.superchatty: print('              ---> 3.4.5 Quality-assurance metrics ') 
        
        for s in range(n_shear):     
            # 3.4.5.1 --> Poor Figure of Merit (Q=1) 
            if FOM[s,t] > param.FOM_limit: 
                Q[s,t] = 1
            
            # 3.4.5.2 --> Large fraction of data with spikes (Q=2) 
            if despike_fraction_sh[s,t] > param.despike_shear_fraction_limit: 
                Q[s,t] += 2 

            # 3.4.5.4 --> Too many iterations of de-spiking (Q=8) 
            if despike_pass_count_sh[s,t] > param.despike_shear_iterations_limit: 
                Q[s,t] += 8
    
            # 3.4.5.5 --> Insufficient variance resolved (Q=16) 
            if var_resolved[s,t] < param.variance_resolved_limit: 
                Q[s,t] += 16

        # 3.4.5.3 --> Large disagreement between dissipation estimates from probes (Q=4)
        if Q[0,t] not in [1,3] and Q[1,t] not in [1,3]: # test only if FOM is good   
            lnepsi_std_avg = np.nanmean(lnepsi_std[:,t])
            if abs(np.log(epsi[0,t]) - np.log(epsi[1,t])) > param.diss_ratio_limit*lnepsi_std_avg: Q[:,t] += 4
  
        # --> Manual entry for shear probe failure for station KASEAOPE_1 ("MUN0") 
        if param.station_name == 'KASEAOPE_1':
            if param.file_nc[21:33] in ['009_Profile2','010_Profile1','010_Profile2','011_Profile1','011_Profile2','011_Profile3','011_Profile4','013_Profile1','013_Profile2','013_Profile3','013_Profile4','014_Profile1','014_Profile2','014_Profile3','014_Profile4','016_Profile1','016_Profile2']: 
                if param.superchatty: print('              --->       + Add manual entry for shear probe failure ') 
                Q[0,t] = 32  
 
        # Grand Finale : give best estimate of dissipation 
        if   Q[0,t]==0 and Q[1,t]==0: # both estimates are good  
            epsi_final[t] = np.mean(epsi[:,t])
        elif Q[0,t]==4 and Q[1,t]==4: # large disagreement between estimates
            epsi_final[t] = np.min((epsi[0,t],epsi[1,t]))
        elif Q[0,t]==0 and Q[1,t]!=0: # eps1 is good but not eps2  
            epsi_final[t] = epsi[0,t]
        elif Q[0,t] not in {0,4} and Q[1,t] in {0,4}: # eps1 is not good but eps2 is good 
            epsi_final[t] = epsi[1,t]
        elif Q[0,t]!=0 and Q[1,t]==0: # eps2 is good but not eps1  
            epsi_final[t] = epsi[1,t]
        elif Q[0,t] in {0,4} and Q[1,t] not in {0,4}: # eps2 is not good but eps1 is good 
            epsi_final[t] = epsi[0,t]

        if param.compute_chi:
            if param.superchatty: print('              ---> [...] Compute temperature variance dissipation ') 
            f = kcyc[:,t] * pspd_rel[t]
            for tid in range(n_temp): 
                # CV 2026/05/22 : grad_temp_spec* are wavenumber spectra of temperature vertical gradient [K^2 m^-2 cpm-1] 
                phi   = grad_temp_spec_clean[tid,:,t] / pspd_rel[t]       # wavenumber to frequency spectrum 
                phi_n = grad_temp_spec_noise_clean[tid,:,t] / pspd_rel[t] # wavenumber to frequency spectrum 
                n_fftseg = np.floor(param.diss_length / param.fft_length/2) # TO BE CHECKED. in Bieito's code it is variable 'NsegM'  
                # NB : in thermistor_fit, TRcor should be False as the correction is done in L3  
                # --> MLE method (Ruddick et al., 2000)
                if ~np.isnan(epsi_final[t]): # provide epsilon from shear to adjust grad(temp) spectrum for missing variance 
                    out = microp.thermistor_fit(f,phi,phi_n,pspd_rel[t],n_fftseg,P=pres[t],T=temp[t],K1=1,fAA=param.f_AA,\
                                            Tdis='K',channel=tid,TRcor=False,mad_factor=2,fs=param.fs_fast,plt_flag=False,epsSH=epsi_final[t]) 
                else:  
                    out = microp.thermistor_fit(f,phi,phi_n,pspd_rel[t],n_fftseg,P=pres[t],T=temp[t],K1=1,fAA=param.f_AA,\
                                            Tdis='K',channel=tid,TRcor=False,mad_factor=2,fs=param.fs_fast,plt_flag=False) 
                chi_I[tid,t]       = out[0]  # chi from simple spectral integral  
                chi_T[tid,t]       = out[1]  # chi from temperature (MLE fit)  
                chi_ST[tid,t]      = out[13] # chi from temperature and corrected from shear sensor 
                kB_T[tid,t]        = out[2]  # Batchelor wavenumber from temperature (MLE fit)  
                kB_S[tid,t]        = out[17] # Batchelor wavenumber from shear 
                epsi_T[tid,t]      = out[3]  # epsilon from MLE fit 
                MAD_T[tid,t]       = out[7]  # mean absolute difference between theoretical and actual temp spectra  
                MAD_ST[tid,t]      = out[14] # same but theoretical spectrum using shear data to compute Batchelor wavenumber 
                MAD_crit[tid,t]    = out[8]  # critical mean absolute difference 
                lkh_ratio[tid,t]   = out[10] # likelihood ratio 
                chi_flag[tid,t]    = out[4]  # fit flag according to MAD_T and lkh_ratio 
                ik_fit[tid,t,:]    = out[15] # indices of min and max K over which fit is performed 
                pp_fit[tid,t,:]    = out[16] # polynomial fit coefficients
                if tid==0:
                    epsi_T_max[t]  = out[12] # max epsilon from MLE that can be estimated  
                #print(chi_I[tid,t], chi_T[tid,t], chi_ST[tid,t]) 


    if param.chatty: print(' ------> Save data in netcdf file ')
    nc = Dataset(param.file_nc,'r+')

    # --- Create missing dimensions 
    if 'TWO' not in nc.dimensions:
        nc.createDimension('TWO',2)

    # --- Define L4 variables and their attributes   
    group = nc.groups['L4_dissipation']

    if 'TIME' not in group.variables:
        nc_time = group.createVariable('TIME', 'f8', ('TIME_SPECTRA'))
        nc_time.standard_name = "time"
        nc_time.units = time_units
        nc_time.axis = "T"
        nc_time.long_name = "Decimal day"
        nc_time.comment = "1 January noon is day 0.5."   
    else: nc_time = group.variables['TIME']

    if 'EPSI' not in group.variables:
        nc_epsi = group.createVariable('EPSI', 'f8', ('N_SHEAR_SENSORS', 'TIME_SPECTRA')) 
        nc_epsi.standard_name = "specific_turbulent_kinetic_energy_dissipation_in_sea_water" 
        nc_epsi.units = "W kg-1"
        nc_epsi.long_name = "dissipation rate of turbulent kinetic energy per unit mass in sea water estimated from individual shear probes"
    else: nc_epsi = group.variables['EPSI']

    if 'EPSI_FINAL' not in group.variables:
        nc_epsi_final = group.createVariable('EPSI_FINAL', 'f8', ('TIME_SPECTRA')) 
        nc_epsi_final.standard_name = "specific_turbulent_kinetic_energy_dissipation_in_sea_water" 
        nc_epsi_final.units = "W kg-1" 
        nc_epsi_final.long_name = "dissipation rate of turbulent kinetic energy per unit mass in sea water averaged using all accepted shear probes" 
    else: nc_epsi_final = group.variables['EPSI_FINAL']
       
    if 'KMAX' not in group.variables: 
        nc_kmax = group.createVariable('KMAX', 'f8', ('N_SHEAR_SENSORS', 'TIME_SPECTRA'))
        nc_kmax.standard_name = "maximum_wavenumber_used_for_estimating_turbulent_kinetic_energy_dissipation" 
        nc_kmax.units = "cpm" 
        nc_kmax.long_name = "maximum wavenumber for integration of shear spectrum"
    else: nc_kmax = group.variables['KMAX'] 

    if 'N_S' not in group.variables: 
        nc_n_s = group.createVariable('N_S', 'i4', ('N_SHEAR_SENSORS', 'TIME_SPECTRA'))
        nc_n_s.standard_name = "number_of_spectral_points_used_for_estimating_turbulent_kinetic_energy_dissipation"  
        nc_n_s.units = "1"  
        nc_n_s.long_name = "number of spectral points used for estimating turbulent kinetic energy dissipation"  
        nc_n_s.comment = "it is the same for the integration or the fit method" 
    else: nc_n_s = group.variables['N_S']

    if 'SECTION_NUMBER' not in group.variables: 
        nc_section_number = group.createVariable('SECTION_NUMBER', 'i4', ('TIME_SPECTRA'))
        nc_section_number.standard_name = "unique_identifier_for_each_section_of_data_from_timeseries"
        nc_section_number.units = "1"
        nc_section_number.long_name = "A unique indentifier counter defining sections of the time series extracted for dissipation estimates"
    else: nc_section_number = group.variables['SECTION_NUMBER']

    if 'PSPD_REL' not in group.variables:
        nc_pspd_rel = group.createVariable('PSPD_REL', 'f8', ('TIME_SPECTRA'))
        nc_pspd_rel.standard_name = "platform_speed_wrt_sea_water"
        nc_pspd_rel.units = "m s-1"
        nc_pspd_rel.long_name = "Platform speed with respect to sea water"
    else: nc_pspd_rel = group.variables['PSPD_REL']
 
    if 'METHOD' not in group.variables: 
        nc_method = group.createVariable('METHOD', 'i4', ('N_SHEAR_SENSORS', 'TIME_SPECTRA')) 
        nc_method.standard_name = "method_used_for_estimating_turbulent_kinetic_energy_dissipation"  
        nc_method.units = "1"   
        nc_method.long_name = "method for dissipation rate estimation: 0 is spectral integration in the viscous subrange. 1 is spectral fit to the inertial subrange."
    else: nc_method = group.variables['METHOD'] 

    if 'PRES' not in group.variables:
        nc_pres = group.createVariable('PRES', 'f8', ('TIME_SPECTRA'))
        nc_pres.standard_name = "sea_water_pressure"
        nc_pres.units = "decibar"
        nc_pres.long_name = "Sea water pressure, equals 0 at sea-level"
    else: nc_pres = group.variables['PRES']
 
    if 'TEMP' not in group.variables:
        nc_temp = group.createVariable('TEMP', 'f8', ('TIME_SPECTRA'))
        nc_temp.standard_name = "sea_water_temperature"
        nc_temp.units = "degree_Celsius"
        nc_temp.long_name = "sea water temperature in-situ ITS-90 scale"
    else: nc_temp = group.variables['TEMP']
 
    if 'FOM' not in group.variables: 
        nc_fom = group.createVariable('FOM', 'f8', ('N_SHEAR_SENSORS', 'TIME_SPECTRA')) 
        nc_fom.standard_name = "figure_of_merit" 
        nc_fom.units = "1" 
        nc_fom.long_name = "Ratio of the MAD of the spectrum to the 97.5 percentile of the expected MAD for the number of spectral points used to estimate the rate of dissipation"
        nc_fom.comment = "If the number of spectral points is N, and the expected standard deviation of the natural logarithm of spectrum of shear is sig, expected MAD is sigx(0.8+(1.25/sqrt(N))). If the spectrum is calculated using number of FFT segments Nf, and cleaned using Nv vibration signals, sig = sqrt ( 5/4 ((Nf - Nv)^(-7/9)))."
    else: nc_fom = group.variables['FOM'] 

    if 'MAD' not in group.variables: 
        nc_mad = group.createVariable('MAD', 'f8', ('N_SHEAR_SENSORS', 'TIME_SPECTRA')) 
        nc_mad.standard_name = "mean_absolute_deviation" 
        nc_mad.units = "1" 
        nc_mad.long_name = "mean absolute deviation (MAD) of the natural logarithm of the shear spectrum from the logarithm of a reference spectrum for the estimated rate of dissipation" 
        nc_mad.comment = "MAD of the spectrum is calculated using the wavenumbers up to KMAX, relative to the model spectrum, using: e.g., mean(abs(log(observed_spectrum/model_spectrum)))" 
    else: nc_mad = group.variables['MAD'] 

    if 'VAR_RESOLVED' not in group.variables: 
        nc_var_resolved = group.createVariable('VAR_RESOLVED', 'f8', ('N_SHEAR_SENSORS', 'TIME_SPECTRA'))
        nc_var_resolved.standard_name = "variance_resolved" 
        nc_var_resolved.units = "1" 
        nc_var_resolved.long_name = "variance resolved in the spectra used for the estimate rate of dissipation" 
    else: nc_var_resolved = group.variables['VAR_RESOLVED'] 

    if 'EPSI_STD' not in group.variables: 
        nc_epsi_std = group.createVariable('EPSI_STD', 'f8', ('N_SHEAR_SENSORS', 'TIME_SPECTRA'))
        nc_epsi_std.standard_name = "expected_standard_deviation_of_the_logarithm_of_the_dissipation_estimate"
        nc_epsi_std.units = "1"
        nc_epsi_std.long_name = "For estimates in the viscous subrange, EPSI_STD = sqrt(5.5 / (1 + (Lf/4)^(7/9))), whereLf = (L/Lk)(VAR_RESOLVED^(3/4)), and L is the data length in meters used for spectral estimate, Lk is the Kolmogorov length, and VAR_RESOLVED is the variance resolved in the spectra used for the estimate rate of dissipation."
    else: nc_epsi_std = group.variables['EPSI_STD'] 

    if 'KVISC' not in group.variables: 
        nc_kvisc = group.createVariable('KVISC', 'f8', ('TIME_SPECTRA')) 
        nc_kvisc.standard_name = "kinematic_viscosity_of_water"
        nc_kvisc.units = "m2 s-1"
        nc_kvisc.long_name = "Kinematic viscosity of sea water" 
    else: nc_kvisc = group.variables['KVISC'] 

    if 'EPSI_FLAGS' not in group.variables: 
        nc_epsi_flags = group.createVariable('EPSI_FLAGS', 'i4', ('N_SHEAR_SENSORS', 'TIME_SPECTRA'))
        nc_epsi_flags.standard_name = "dissipation_qc_flag" 
        nc_epsi_flags.units = "1" 
        nc_epsi_flags.long_name = "quality control coding for dissipation estimate from each shear probe" 
        nc_epsi_flags.conventions = "ATOMIX, shear probes group" 
        nc_epsi_flags.flag_values = 0., 1., 2., 4., 8., 16., 32.  
        nc_epsi_flags.flag_meanings = "0: Good, 1: Poor figure of merit (FOM>FOM_limit), 2: Large fraction of data with spikes (despike_shear_fraction > despike_shear_fraction_limit), 4: Anomalously large disagreement between dissipation estimates from probes (|log(e_max)-log(e_min)|> diss_ratio_limit * EPSI_STD), 8: Too many iterations of despiking routine (despike_shear_iterations > despike_shear_iterations_limit), 16: Insufficient variance resolved (VAR_RESOLVED<variance_resolved_limit), 32: manual flag corresponding to shear probe failure"
        nc_epsi_flags.comment = "Combination of flags are possible. For example 3 = 1 + 2: FOM and despike failure; 5 = 1 + 4: FOM failure and anomalously large disagreement between dissipation estimates, and so on."
    else: nc_epsi_flags = group.variables['EPSI_FLAGS']

    if param.compute_chi: 
        if 'CHI_I' not in group.variables: 
            nc_chi_I = group.createVariable('CHI_I','f8',('N_TEMP_SENSORS','TIME_SPECTRA')) 
            nc_chi_I.standard_name = 'temperature_variance_dissipation_from_spectral_integral' 
            nc_chi_I.units = 'K2 s-1'
            nc_chi_I.long_name = 'Temperature variance dissipation from spectral integral' 
        else: nc_chi_I = group.variables['CHI_I'] 

        if 'CHI_T' not in group.variables: 
            nc_chi_T = group.createVariable('CHI_T','f8',('N_TEMP_SENSORS','TIME_SPECTRA'))  
            nc_chi_T.standard_name = 'temperature_variance_dissipation_from_MLE_fit' 
            nc_chi_T.units = 'K2 s-1'
            nc_chi_T.long_name = 'Temperature variance dissipation from Maximum Likelihood Estimation (Ruddick et al. 2000)' 
        else: nc_chi_T = group.variables['CHI_T'] 
        
        if 'CHI_ST' not in group.variables: 
            nc_chi_ST = group.createVariable('CHI_ST','f8',('N_TEMP_SENSORS','TIME_SPECTRA'))  
            nc_chi_ST.standard_name = 'temperature_variance_dissipation_from_epsilon' 
            nc_chi_ST.units = 'K2 s-1'
            nc_chi_ST.long_name = 'Temperature variance dissipation corrected for missing variance using epsilon from shear'
        else: nc_chi_ST = group.variables['CHI_ST'] 

        if 'EPSI_T' not in group.variables: 
            nc_epsi_T = group.createVariable('EPSI_T','f8',('N_TEMP_SENSORS','TIME_SPECTRA'))  
            nc_epsi_T.standard_name = 'specific_turbulent_kinetic_energy_dissipation_in_sea_water' 
            nc_epsi_T.units = 'W kg-1' 
            nc_epsi_T.long_name = 'Turbulent kinetic energy dissipation from MLE fit of temperature' 
        else: nc_epsi_T = group.variables['EPSI_T'] 
        
        if 'EPSI_T_MAX' not in group.variables: 
            nc_epsi_T_max = group.createVariable('EPSI_T_MAX','f8',('TIME_SPECTRA'))  
            nc_epsi_T_max.standard_name = 'specific_turbulent_kinetic_energy_dissipation_in_sea_water' 
            nc_epsi_T_max.units = 'W kg-1' 
            nc_epsi_T_max.long_name = 'Maximum turbulent kinetic energy dissipation that can be estimated from MLE fit of temperature' 
        else: nc_epsi_T_max = group.variables['EPSI_T_MAX'] 

        if 'MAD_T' not in group.variables: 
            nc_mad_T = group.createVariable('MAD_T','f8',('N_TEMP_SENSORS','TIME_SPECTRA')) 
            nc_mad_T.standard_name = 'mean_absolute_deviation' 
            nc_mad_T.units = '1' 
            nc_mad_T.long_name = 'mean absolute deviation (MAD) of the temperature gradient spectrum from the logarithm of a reference (theoretical) spectrum' 
        else: nc_mad_T = group.variables['MAD_T']  
        
        if 'MAD_ST' not in group.variables: 
            nc_mad_ST = group.createVariable('MAD_ST','f8',('N_TEMP_SENSORS','TIME_SPECTRA')) 
            nc_mad_ST.standard_name = 'mean_absolute_deviation' 
            nc_mad_ST.units = '1' 
            nc_mad_ST.long_name = 'mean absolute deviation (MAD) of the temperature gradient spectrum from the logarithm of a reference (theoretical) spectrum using shear data to compute Batchelor spectrum' 
        else: nc_mad_ST = group.variables['MAD_ST']  
        
        if 'MAD_critical' not in group.variables: 
            nc_mad_crit = group.createVariable('MAD_critical','f8',('N_TEMP_SENSORS','TIME_SPECTRA')) 
            nc_mad_crit.standard_name = 'critical_mean_absolute_deviation' 
            nc_mad_crit.units = '1' 
            nc_mad_crit.long_name = 'critical mean absolute deviation (MAD) of the temperature gradient spectrum from the logarithm of a reference (theoretical) spectrum' 
        else: nc_mad_crit = group.variables['MAD_critical']  

        if 'LKH_RATIO' not in group.variables: 
            nc_lkh_ratio = group.createVariable('LKH_RATIO','f8',('N_TEMP_SENSORS','TIME_SPECTRA')) 
            nc_lkh_ratio.standard_name = 'likelihood_ratio' 
            nc_lkh_ratio.units = '1' 
            nc_lkh_ratio.long_name = 'likelihood ratio between theoretical and power-law MLE' 
        else: nc_lkh_ratio = group.variables['LKH_RATIO'] 

        if 'CHI_FLAG' not in group.variables: 
            nc_chi_flag = group.createVariable('CHI_FLAG','i4',('N_TEMP_SENSORS','TIME_SPECTRA'))  
            nc_chi_flag.standard_name = 'temperature_variance_dissipation_qc_flag' 
            nc_chi_flag.units = '1' 
            nc_chi_flag.long_name = 'quality flag for temperature variance dissipation estimated from MLE'
        else: nc_chi_flag = group.variables['CHI_FLAG'] 

        if 'KB_T' not in group.variables: 
            nc_kb_T = group.createVariable('KB_T','f8',('N_TEMP_SENSORS','TIME_SPECTRA'))  
            nc_kb_T.standard_name = 'Batchelor_wavenumber'
            nc_kb_T.units = 'cpm' 
            nc_kb_T.long_name = 'Batchelor wavenumber from MLE' 
        else: nc_kb_T = group.variables['KB_T']
        
        if 'KB_S' not in group.variables: 
            nc_kb_S = group.createVariable('KB_S','f8',('N_TEMP_SENSORS','TIME_SPECTRA'))  
            nc_kb_S.standard_name = 'Batchelor_wavenumber'
            nc_kb_S.units = 'cpm' 
            nc_kb_S.long_name = 'Batchelor wavenumber from shear' 
        else: nc_kb_S = group.variables['KB_S']
        
        if 'IK_FIT' not in group.variables: 
            nc_ik_fit = group.createVariable('IK_FIT','i4',('N_TEMP_SENSORS','TIME_SPECTRA','TWO'))  
            nc_ik_fit.standard_name = 'min_max_wavenumber_indices_for_fit'
            nc_ik_fit.units = '1' 
            nc_ik_fit.long_name = 'indices of min and max wavenumbers that set the range over which fit is performed' 
        else: nc_ik_fit = group.variables['IK_FIT']
        
        if 'PP_FIT' not in group.variables: 
            nc_pp_fit = group.createVariable('PP_FIT','f8',('N_TEMP_SENSORS','TIME_SPECTRA','TWO'))  
            nc_pp_fit.standard_name = 'polynomial_coefficients_from_fit_to_temperature_spectrum' 
            nc_pp_fit.units = '1' 
            nc_pp_fit.long_name = 'polynomial coefficients from fit to temperature spectrum'
        else: nc_pp_fit = group.variables['PP_FIT']
 
    # --- Copy variables 
    nc_time[:]           = time  
    nc_epsi[:]           = epsi  
    nc_epsi_final[:]     = epsi_final  
    nc_kmax[:]           = ku   
    nc_n_s[:]            = N_s 
    nc_section_number[:] = section_number 
    nc_pspd_rel[:]       = pspd_rel  
    nc_method[:]         = np.zeros((param.n_estimates)) # the fit method had not been implemented yet   
    nc_pres[:]           = pres  
    nc_temp[:]           = temp  
    nc_fom[:]            = FOM  
    nc_mad[:]            = MAD  
    nc_var_resolved[:]   = var_resolved  
    nc_epsi_std[:]       = lnepsi_std 
    nc_kvisc[:]          = nu 
    nc_epsi_flags[:]     = Q 
    if param.compute_chi: 
        nc_chi_I[:]         = chi_I 
        nc_chi_T[:]         = chi_T 
        nc_chi_ST[:]        = chi_ST 
        nc_epsi_T[:]        = epsi_T 
        nc_epsi_T_max[:]    = epsi_T_max  
        nc_mad_T[:]         = MAD_T 
        nc_mad_ST[:]        = MAD_ST 
        nc_mad_crit[:]      = MAD_crit
        nc_lkh_ratio[:]     = lkh_ratio
        nc_chi_flag[:]      = chi_flag  
        nc_kb_T[:]          = kB_T
        nc_kb_S[:]          = kB_S
        nc_ik_fit[:]        = ik_fit    
        nc_pp_fit[:]        = pp_fit    
    nc.close()
    return 


