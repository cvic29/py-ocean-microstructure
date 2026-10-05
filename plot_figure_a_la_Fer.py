'''
CV 2026/01/30 : plot VMP-measured fields such as Figure 2 in Fer et al. (2024)

contact : clement.vic@ifremer.fr 
'''
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
import matplotlib.gridspec as gridspec
from netCDF4 import Dataset
from spectral_analysis import spectral_model  
import warnings; warnings.filterwarnings('ignore')

# - KASEAOPE-3 - 
path_data = '/Users/cv1m15/Documents/Campagnes/2025_Kaseaope-3/LEG2/DATA/VMP/'
station   = 'MUN1'
file_name = 'VMP_036_Profile1.nc'
kspec     = [39, 155] # indices for showing spectra in panel (f)  

# - plot options - 
fs   = 8 
lw   = 0.5
ms   = 5 # marker size for scatter plots 
my_bbox = dict(fc='w',ec='k',pad=2,lw=0.4,alpha=0.7)
xlet1,ylet1 = 0.01,0.94 # for time series 
xlet2,ylet2 = 0.03,0.97 # for square plots 

print(' ------> Extract variables in netcdf file ')
nc = Dataset(path_data+station+'/'+file_name,'r')
pres      = nc.groups['L1_converted'].variables['PRES'][:]
time_L1   = nc.groups['L1_converted'].variables['TIME'][:]
pspd_rel  = nc.groups['L2_cleaned'].variables['PSPD_REL'][:]
shear     = nc.groups['L2_cleaned'].variables['SHEAR'][:] 
acc       = nc.groups['L2_cleaned'].variables['ACC'][:] 
kcyc          = nc.groups['L3_spectra'].variables['KCYC'][:] 
sh_spec_clean = nc.groups['L3_spectra'].variables['SH_SPEC_CLEAN'][:] 
eps           = nc.groups['L4_dissipation'].variables['EPSI'][:]
eps_final     = nc.groups['L4_dissipation'].variables['EPSI_FINAL'][:]
eps_std       = nc.groups['L4_dissipation'].variables['EPSI_STD'][:]
time_L4       = nc.groups['L4_dissipation'].variables['TIME'][:]
kvisc         = nc.groups['L4_dissipation'].variables['KVISC'][:]
flag          = nc.groups['L4_dissipation'].variables['EPSI_FLAGS'][:] # (N_SHEAR_SENSORS, TIME_SPECTRA) 
fom           = nc.groups['L4_dissipation'].variables['FOM'][:] # (N_SHEAR_SENSORS, TIME_SPECTRA) 
nc.close()


kspec[0] = np.nanargmin(eps_final) 
kspec[1] = np.nanargmax(eps_final) 

print(f' ===== Spectra shown for epsilon [W kg-1] = {eps_final[kspec[0]]:.1e} and {eps_final[kspec[1]]:.1e}')
print(f' ===== corresponding FOM are for eps1 : {fom[0,kspec[0]]:.2e} and {fom[0,kspec[1]]:.2e}') 
print(f'                             for eps2 : {fom[1,kspec[0]]:.2e} and {fom[1,kspec[1]]:.2e}') 

if 0:
    plt.figure()
    plt.plot(flag[0,:],'r+')
    plt.plot(flag[0,:],'b+')
    plt.savefig('tmp.pdf') 
    exit()

pspd_rel[0] = pspd_rel[1]
time = (time_L1 - time_L1[0])*86400 
time_eps = (time_L4 - time_L1[0])*86400 # same origin as time in L1
   
print(' ------> Make plot ')
plt.figure(figsize=(5,7))
gs = gridspec.GridSpec(6,2,height_ratios=[1,1,1,1,0.3,2],wspace=0.3) # 4*time series + blank + square plots 
ax = plt.subplot(gs[0,:]) # ------------------------------- pressure and platform speed 
plt.text(xlet1,ylet1,'a',fontsize=fs,ha='left',va='top',transform=ax.transAxes,bbox=my_bbox)
plt.plot(time,pres,'b',lw=lw)  
plt.ylabel('$P$\n[dbar]',fontsize=fs) 
ax.set_xticklabels(())  
ax.tick_params(labelsize=fs)
axt = ax.twinx() # --- platform speed 
plt.plot(time,pspd_rel,'r',lw=lw)
axt.tick_params(labelsize=fs,axis='y',colors='r') 
plt.ylabel('$W$ [m s$^{-1}$]',fontsize=fs,color='r') 

ax = plt.subplot(gs[1,:]) # ------------------------------- shear
plt.text(xlet1,ylet1,'b',fontsize=fs,ha='left',va='top',transform=ax.transAxes,bbox=my_bbox)
plt.plot(time,shear[0,:],'b',lw=lw)
plt.plot(time,shear[1,:]+2,'r',lw=lw)
plt.ylabel('$\partial u_i/\partial z$\n[s$^{-1}$]',fontsize=fs)  
ax.set_xticklabels(())  
ax.tick_params(labelsize=fs)

ax = plt.subplot(gs[2,:]) # ------------------------------- acceleration 
plt.text(xlet1,ylet1,'c',fontsize=fs,ha='left',va='top',transform=ax.transAxes,bbox=my_bbox)
plt.plot(time,acc[0,:],'b',lw=lw)
plt.plot(time,acc[1,:]+500,'r',lw=lw)
plt.ylabel('$A_x, A_y$\n[counts]',fontsize=fs)  
ax.set_xticklabels(())  
ax.tick_params(labelsize=fs)

ax = plt.subplot(gs[3,:]) # ------------------------------- epsilon 1 and 2 
plt.text(xlet1,ylet1,'d',fontsize=fs,ha='left',va='top',transform=ax.transAxes,bbox=my_bbox)
plt.scatter(time_eps,eps[0,:],marker='o',facecolors='b',edgecolors='b',s=ms,linewidths=lw,label=r'$\varepsilon_1$') 
plt.scatter(time_eps,eps[1,:],marker='o',facecolors='r',edgecolors='r',s=ms,linewidths=lw,label=r'$\varepsilon_2$')
plt.scatter(time_eps,eps_final,marker='s',facecolors='w',edgecolors='k',s=ms,linewidths=lw,label=r'$\varepsilon$')
plt.plot([time_eps[kspec[0]],time_eps[kspec[0]]],[1e-10,1e-5],'gray',lw=lw) # eps whose shear spec is shown in (f)  
plt.plot([time_eps[kspec[1]],time_eps[kspec[1]]],[1e-10,1e-5],'gray',lw=lw) 
plt.legend(bbox_to_anchor=(0.5,0.95),loc='center',prop={'size':fs},ncol=3,handletextpad=0.01,columnspacing=0.4,framealpha=1)
plt.yscale('log')
ax.set_yticks([1e-9,1e-8,1e-7,1e-6])  
plt.ylim(1e-10,1e-5)
plt.ylabel(r'$\varepsilon$'+'\n[W kg$^{-1}$]',fontsize=fs)  
plt.xlabel('time [second]',fontsize=fs)  
ax.tick_params(labelsize=fs)

ax = plt.subplot(gs[5,0]) # ------------------------------- scatter epsilon 1 vs epsilon 2 
plt.text(xlet2,ylet2,'e',fontsize=fs,ha='left',va='top',transform=ax.transAxes,bbox=my_bbox)
plt.scatter(eps[0,:],eps[1,:],marker='o',facecolors='none',edgecolors='k',s=ms,linewidths=lw,zorder=4) 
# - statistical uncertainty - 
xx = np.power(10,np.linspace(-10,-5,100))
plt.fill_between(xx,np.exp(-2.77*np.mean(eps_std))*xx,np.exp(+2.77*np.mean(eps_std))*xx,color='lightgray',edgecolor='none',alpha=0.5)
plt.plot(xx,xx,'w',lw=lw) 
# - bad flag -
eps_copy = np.copy(eps) 
eps_copy[flag==0] = np.nan 
#eps_copy[flag==4] = np.nan # one of the epsilon is rejected  
plt.scatter(eps_copy[0,:],eps_copy[1,:],marker='x',facecolors='r',edgecolors='r',s=ms,linewidths=lw,zorder=4) 

plt.xscale('log'); plt.yscale('log') 
plt.xlim(1e-10,9e-6); plt.ylim(1e-10,9e-6) 
ax.set_aspect('equal', 'box')
plt.xlabel(r'$\varepsilon_1$',fontsize=fs) 
plt.ylabel(r'$\varepsilon_2$',fontsize=fs) 
ax.tick_params(labelsize=fs)

ax = plt.subplot(gs[5,1]) # ------------------------------- spectra
plt.text(xlet2,ylet2,'f',fontsize=fs,ha='left',va='top',transform=ax.transAxes,bbox=my_bbox)
plt.plot(kcyc[:,kspec[0]],sh_spec_clean[0,:,kspec[0]],'b',lw=lw,label='$\Psi_{1c}$') # first index  
plt.plot(kcyc[:,kspec[0]],sh_spec_clean[1,:,kspec[0]],'r',lw=lw,label='$\Psi_{2c}$') 
plt.plot(kcyc[:,kspec[1]],sh_spec_clean[0,:,kspec[1]],'b',lw=lw) # second index  
plt.plot(kcyc[:,kspec[1]],sh_spec_clean[1,:,kspec[1]],'r',lw=lw) 
# - Lueck model (first index) - 
L_K = (kvisc[kspec[0]]**3/eps_final[kspec[0]])**0.25 # Kolmogorov scale
kcyc_nondim = kcyc[:,kspec[0]]*L_K 
spec_model_nondim = spectral_model(kcyc_nondim,model='Lueck') 
spec_model        = (eps_final[kspec[0]]**3/kvisc[kspec[0]])**0.25 * spec_model_nondim
plt.plot(kcyc[:,kspec[0]],spec_model,'gray',lw=2*lw,alpha=0.5,label='Lueck')
# - Lueck model (second index) - 
L_K = (kvisc[kspec[1]]**3/eps_final[kspec[1]])**0.25 # Kolmogorov scale
kcyc_nondim = kcyc[:,kspec[1]]*L_K 
spec_model_nondim = spectral_model(kcyc_nondim,model='Lueck') 
spec_model        = (eps_final[kspec[1]]**3/kvisc[kspec[1]])**0.25 * spec_model_nondim
plt.plot(kcyc[:,kspec[1]],spec_model,'gray',lw=2*lw,alpha=0.5)
 
plt.legend(bbox_to_anchor=(0.0,0.0),loc='lower left',prop={'size':fs},ncol=1,handletextpad=0.05,framealpha=1)
plt.xscale('log'); plt.yscale('log')
plt.ylabel('$\Psi$ [s$^{-2}$ cpm$^{-1}$]',fontsize=fs) 
plt.xlabel('$k$ [cpm]',fontsize=fs) 
plt.xlim(5e-1,5e2) 
plt.ylim(1e-9,1e-2) 
ax.tick_params(labelsize=fs)


plt.savefig(f'../Figures/{station}_{file_name[:-3]}_synthesis.pdf',bbox_inches='tight') 

