'''
CV 2025/11/05 : Read mat file exported by Zissou Premium with signals converted to physical units  
                and write data in netcdf file correponding to L1 processing 
                Running explore_rockland_mat_file.py gives the variables that can be extracted 
CV 2026/04/29 : add gradT to further compute temperature variance dissipation (chi) 

contact : clement.vic@ifremer.fr
'''
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import loadmat
from netCDF4 import Dataset
from datetime import datetime
import os 

# --- User defined parameters (adjusted to original dataset) 
station         = 'STY3'
path_files      = r'../DATA/VMP/'+station+'/'
files           = os.listdir(path_files) 
list_file_mat   = [f for f in files if f.endswith('.mat')]
#list_file_mat.sort()  # Sorts alphabetically
#list_file_mat = [str(list_file_mat[0])] # to test function with only one element  
#list_file_mat = ['VMP_011_Profile2.mat'] # MUN0 
#list_file_mat = ['VMP_010_Profile5.mat'] # MUN2 
#list_file_mat = ['VMP_040_Profile1.mat'] # MUN1 

print(list_file_mat) 
file_coord      = path_files+station+'_lonlat.txt'

n_shear_sensors = 2        # number of shear sensors 
n_acc_sensors   = 2        # number of acceleration / vibration sensors 
n_temp_sensors  = 2        # microstructure (not CTD)  
lat_min,lat_max = -24,-22  # lat min and max -- not critical, used to convert pressure to depth 

print(' ------> Read VMP lon,lat ')
ff = open(file_coord,'r')
lines = ff.readlines()
nlines = len(lines)
mat_vmp = []; lat_vmp = []; lon_vmp = []
for ll in range(1,nlines):              
    mat_vmp.append(lines[ll].split(' ')[0])
    lon_vmp.append(float(lines[ll].split(' ')[1]))
    lat_vmp.append(float(lines[ll].split(' ')[2]))

# Special treatment for some stations where two files correspond to 
# a single profile that has been split
if station == 'MUN0':
    mat_vmp.append('VMP_013_Profile3.mat') 
    lon_vmp.append(lon_vmp[mat_vmp.index('VMP_013_Profile2.mat')]) 
    lat_vmp.append(lat_vmp[mat_vmp.index('VMP_013_Profile2.mat')]) 
if station == 'MUN1':
    mat_vmp.append('VMP_050_Profile4.mat') 
    lon_vmp.append(lon_vmp[mat_vmp.index('VMP_050_Profile3.mat')]) 
    lat_vmp.append(lat_vmp[mat_vmp.index('VMP_050_Profile3.mat')]) 


print(' ------> Loop on files in directory ') 
npro = len(list_file_mat)
for i in range(npro):
    print('         --> '+list_file_mat[i])
    mat = loadmat(path_files+list_file_mat[i])
    data = mat['data']
    # --- Extract variables converted to physical units  
    file_start_date = data['file_start_date'][0][0][0]
    file_start_time = data['file_start_time'][0][0][0]
    t_fast          = np.squeeze(data['t_fast'][0][0])   # seconds since file start date and time  
    t_slow          = np.squeeze(data['t_slow'][0][0])
    p_fast          = np.squeeze(data['p_fast'][0][0]) 
    p_slow          = np.squeeze(data['p_slow'][0][0])
    sh1             = np.squeeze(data['sh1'][0][0])      # converted to physical units but w/o further processing 
    sh2             = np.squeeze(data['sh2'][0][0]) 
    Ax              = np.squeeze(data['Ax'][0][0]) 
    Ay              = np.squeeze(data['Ay'][0][0])  
    T1              = np.squeeze(data['T1'][0][0])       # fast temperature  
    T2              = np.squeeze(data['T2'][0][0])       # fast temperature  
    gradT1          = np.squeeze(data['gradT1'][0][0])   
    gradT2          = np.squeeze(data['gradT2'][0][0])   
    JAC_T           = np.squeeze(data['JAC_T'][0][0])    # CTD temperature 
    JAC_C           = np.squeeze(data['JAC_C'][0][0])    # CTD conductivity 
    salinity        = np.squeeze(data['salinity'][0][0]) # fast salinity (reconstructed from CTD and fast temp ?)  
    fs_fast         = data['fs_fast'][0][0][0][0]  
    fs_slow         = data['fs_slow'][0][0][0][0]  

    #print(data['cfgobj'])      
    #print(Ax.shape,T1.shape,sh1.shape,salinity.shape,t_slow.shape) 


    # --- Convert t_slow and t_fast in day of year (doy) 
    start_date_time    = datetime.strptime(file_start_date+' '+file_start_time, "%Y-%m-%d %H:%M:%S")
    start_date_time_ns = np.datetime64(start_date_time, 'ns')   # nanosecond precision 
    t_slow_ns       = (t_slow * 1e9).astype(np.int64)  # Convert seconds to nanoseconds 
    t_slow_datetime = start_date_time_ns + t_slow_ns.astype('timedelta64[ns]')
    t_slow_doy      = ((t_slow_datetime - t_slow_datetime.astype('datetime64[Y]')) / np.timedelta64(1, 'D')).astype(float)
    t_fast_ns       = (t_fast * 1e9).astype(np.int64)  # Convert seconds to nanoseconds 
    t_fast_datetime = start_date_time_ns + t_fast_ns.astype('timedelta64[ns]')
    t_fast_doy      = ((t_fast_datetime - t_fast_datetime.astype('datetime64[Y]')) / np.timedelta64(1, 'D')).astype(float)
    print(t_slow_doy[:5])
    print(t_fast_doy[:5])
 
    # --- Note that there might be variables processed w/ Rockland standard parameters 
    #sh1_raw   = np.squeeze(data['sh1_raw'][0][0])    # by segment onto which dissipation is computed   
    #sh1_clean = np.squeeze(data['sh1_clean'][0][0])  # by segment onto which dissipation is computed   
    #print(sh1.shape,sh1_raw.shape,sh1_clean.shape) 

    # --- Save in netcdf file 
    #nc = Dataset(path_files+list_file_mat[i][:-3]+'nc','r+')
    nc = Dataset(path_files+list_file_mat[i][:-4]+'_gradT.nc','r+')
    if 'L1_converted' not in nc.groups:
        group = nc.createGroup('L1_converted')  
    if 'TIME' not in nc.dimensions:
        nc.createDimension('TIME',t_fast.shape[0])
    if 'TIME_SLOW' not in nc.dimensions:
        nc.createDimension('TIME_SLOW',t_slow.shape[0])
    if 'N_SHEAR_SENSORS' not in nc.dimensions:
        nc.createDimension('N_SHEAR_SENSORS',n_shear_sensors) 
    if 'N_ACC_SENSORS' not in nc.dimensions:
        nc.createDimension('N_ACC_SENSORS',n_acc_sensors) 
    if 'N_TEMP_SENSORS' not in nc.dimensions:
        nc.createDimension('N_TEMP_SENSORS',n_temp_sensors) 

    # --- Copy longitude and latitude as attributes 
    nc.longitude_profile = lon_vmp[mat_vmp.index(list_file_mat[i])] 
    nc.latitude_profile  = lat_vmp[mat_vmp.index(list_file_mat[i])] 

    # --- Define L1 variables and their attributes   
    group = nc.groups['L1_converted']
    
    if 'TIME' not in group.variables:
        nc_time = group.createVariable('TIME', 'f8', ('TIME',))
        nc_time.standard_name = "time"
        nc_time.units = "Days since "+str(file_start_date[:4])+"-01-01T00:00:00Z"
        nc_time.axis = "T"
        nc_time.long_name = "Decimal day"
        nc_time.comment = "1 January noon is day 0.5."
    else: nc_time = group.variables['TIME']
    
    if 'TIME_SLOW' not in group.variables:
        nc_time_slow = group.createVariable('TIME_SLOW', 'f8', ('TIME_SLOW',))
        nc_time_slow.standard_name = "time"
        nc_time_slow.units = "Days since "+str(file_start_date[:4])+"-01-01T00:00:00Z"
        nc_time_slow.axis = "T"
        nc_time_slow.long_name = "Decimal day"
        nc_time_slow.comment = "1 January noon is day 0.5."
    else: nc_time_slow = group.variables['TIME_SLOW']
    
    if 'PRES' not in group.variables:
        nc_pres = group.createVariable('PRES','f8',('TIME',))
        nc_pres.standard_name = "sea_water_pressure"
        nc_pres.units = "decibars"
        nc_pres.long_name = "Sea water pressure, equals 0 at sea-level"
    else: nc_pres = group.variables['PRES']
    
    if 'PRES_SLOW' not in group.variables:
        nc_pres_slow = group.createVariable('PRES_SLOW','f8',('TIME_SLOW',))
        nc_pres_slow.standard_name = "sea_water_pressure"
        nc_pres_slow.units = "decibar"
        nc_pres_slow.long_name = "Sea water pressure, equals 0 at sea-level"
    else: nc_pres_slow = group.variables['PRES_SLOW']
    
    if 'SHEAR' not in group.variables:
        nc_shear = group.createVariable('SHEAR', 'f8', ('N_SHEAR_SENSORS','TIME'))
        nc_shear.standard_name = "sea_water_velocity_shear"
        nc_shear.units = "s-1"
        nc_shear.long_name = "rate of change of cross axis sea water velocity along transect measured by shear probes"
    else: nc_shear = group.variables['SHEAR']
        
    if 'ACC' not in group.variables:
        nc_acc = group.createVariable('ACC','f8',('N_ACC_SENSORS','TIME'))
        nc_acc.standard_name = "platform_acceleration" 
        nc_acc.units = "m s-2" 
        nc_acc.long_name = "platform acceleration detected by accelerometers"
    else: nc_acc = group.variables['ACC']  
    
    if 'TEMP' not in group.variables:
        nc_temp = group.createVariable('TEMP','f8',('N_TEMP_SENSORS','TIME'))
        nc_temp.standard_name = "sea_water_temperature"
        nc_temp.units = "degree_Celsius"
        nc_temp.long_name = "sea water temperature in-situ ITS-90 scale"
    else: nc_temp = group.variables['TEMP']
    
    if 'TEMP_CTD' not in group.variables:
        nc_temp_ctd = group.createVariable('TEMP_CTD','f8',('TIME_SLOW',))
        nc_temp_ctd.standard_name = "sea_water_temperature"
        nc_temp_ctd.units = "degree_Celsius"
        nc_temp_ctd.long_name = "sea water temperature in-situ ITS-90 scale"
    else: nc_temp_ctd = group.variables['TEMP_CTD']
    
    if 'COND_CTD' not in group.variables:
        nc_cond_ctd = group.createVariable('COND_CTD','f8',('TIME_SLOW',))
        nc_cond_ctd.standard_name = "sea_water_electrical_conductivity"
        nc_cond_ctd.units = "mS cm-1"
        nc_cond_ctd.long_name = "sea water electrical conductivity"
    else: nc_cond_ctd = group.variables['COND_CTD'] 
    
    if 'SALINITY' not in group.variables:
        nc_salinity = group.createVariable('SALINITY','f8',('TIME',))
        nc_salinity.standard_name = "sea_water_salinity"
        nc_salinity.units = "psu"
        nc_salinity.long_name = "sea water salinity"
    else: nc_salinity = group.variables['SALINITY'] 
  
    # CV 2026/04/29 : add gradT to process microstructure temperature for chi (temperature variance dissipation)  
    if 'GRAD_TEMP' not in group.variables:
        nc_grad_temp = group.createVariable('GRAD_TEMP','f8',('N_TEMP_SENSORS','TIME'))
        nc_grad_temp.standard_name = "sea_water_temperature_gradient"
        nc_grad_temp.units = "degree_Celsius_s-1"
        nc_grad_temp.long_name = "sea water temperature gradient in-situ ITS-90 scale"
    else: nc_grad_temp = group.variables['GRAD_TEMP']
 
    # --- Copy variables in the netcdf file 
    nc_time[:]      = t_fast_doy 
    nc_time_slow[:] = t_slow_doy
    nc_pres[:]      = p_fast 
    nc_pres_slow[:] = p_slow 
    nc_shear[:]     = np.stack((sh1,sh2), axis=0) 
    nc_acc[:]       = np.stack((Ax,Ay), axis=0) 
    nc_temp[:]      = np.stack((T1,T2), axis=0) 
    nc_temp_ctd[:]  = JAC_T
    nc_cond_ctd[:]  = JAC_C
    nc_salinity[:]  = salinity 
    nc_grad_temp[:] = np.stack((gradT1,gradT2), axis=0) 
    
    # --- Copy parameters in the netcdf file 
    nc.fs_fast            = fs_fast 
    nc.fs_slow            = fs_slow 
    nc.geospatial_lat_min = lat_min 
    nc.geospatial_lat_max = lat_max 
    
    nc.close() 
