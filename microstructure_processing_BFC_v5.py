'''
CV 2026/10/05 : set of functions written by Bieito Fernandez Castro (univ Southampton) 
                to process microstructure temperature to get temperature variance dissipation 
 
contact : B.Fernandez-Castro@soton.ac.uk 
'''
import numpy as np
import glob
import scipy.io as io
from scipy.stats import chi2
from scipy.special import erf
import matplotlib.pyplot as plt

#General functions
def moving_average(x,n, window = "flat"):
    if n%2 == 0:
        n+=1
    N = x.size
    cx = np.full(x.size, np.nan)
    for i in range(N):
        ii = np.arange(i-n//2, i+n//2+1,1)
        if window == "flat":
            ww = np.ones(ii.size)
        elif window == "gauss":
            xx = ii - i

            ww = np.exp(- xx**2/(float(n)/4)**2 )
        elif window == "hanning":
            ww = np.hanning(ii.size)
        ww = ww[ (ii>=0) & (ii<N)]
        ii = ii[ (ii>=0) & (ii<N)]

        #print(ii)
        kk = np.isfinite(x[ii])
        if np.sum(kk)<0.25*ii.size:
            continue
        cx[i] = np.sum(x[ii[kk]]*ww[kk])/np.sum(ww[kk])
    return cx

def first_centered_differences(x, y, fill = False):
    if x.size != y.size:
        print("first-centered differences: vectors do not have the same size")
    dy = np.full( x.size, np.nan )
    iif = np.where( (np.isfinite(x)) & (np.isfinite(y))) [0]
    if iif.size == 0:
        return dy
    x0 = x[iif]
    y0 = y[iif]
    dy0 = np.full( x0.size, np.nan )
    #calculates differences
    dy0[0] = (y0[1] - y0[0])/(x0[1]-x0[0])
    dy0[-1] = (y0[-1] - y0[-2])/(x0[-1]-x0[-2])
    dy0[1:-1] = (y0[2:] - y0[0:-2])/(x0[2:]- x0[0:-2])

    dy[iif] = dy0

    if fill:
        dy[0:iif[0]] = dy[iif[0]]
        dy[iif[-1]+1:] = dy[iif[-1]]
    return dy

#simple noise models
def FP07noise(fr, params):
    b1 = params[0]
    m1 = params[1]

    Sn1=(10**b1)*fr**m1;

    if len(params)>2:
        fAA = params[2]
        Sn2 = (1 + (fr/fAA)**10)**-2;
    else:
        Sn2 = 1
    Sn = Sn1*Sn2
    return Sn


def FP07noise0(fr, b1=-10.5,m1=-0.60):
    Sn=(10**b1)*fr**m1;
    return Sn


#fitting routines
def viscosity(T):
    """
    Seawater viscosity calculation.
    
    Parameters:
    T : float or array-like
        Temperature in degrees Celsius.

    Returns:
    v : float or array-like
        Viscosity in m^2/s.
    """
    v = 1.792747 - 0.052126103 * T + 0.0005918645 * T**2
    v *= 10**-6
    return v

def thermal_diffusivity(T):
    """
    Calculate the thermal diffusivity of seawater as a function of temperature.

    Parameters:
    T : float or numpy array
        Temperature in degrees Celsius.

    Returns:
    alpha : float or numpy array
        Thermal diffusivity in m^2/s.
    """
    # Coefficients for thermal diffusivity (empirical fit)
    alpha0 = 1.4e-7   # Base thermal diffusivity at 0°C (m²/s)
    alpha1 = 1.0e-9   # Linear coefficient for temperature
    alpha2 = -7.0e-12 # Quadratic coefficient for temperature
    
    # Calculate thermal diffusivity
    alpha = alpha0 + alpha1 * T + alpha2 * T**2
    
    return alpha

def epsilon_from_KB(visco, kmol, kb):
    return visco*kmol**2*(2*np.pi*kb)**4;

def KB_from_epsilon(visco, kmol, epsilon):
    return (epsilon/visco/kmol**2)**(1./4)/(2*np.pi)

def cost_MLE(Sobs, Steo, dof=6, Sn=0):
    # Add a small value to avoid division by zero (equivalent to eps in MATLAB)
    eps = np.finfo(float).eps
    
    # Compute the cost function
    C11 = np.log(dof / (Steo + Sn + eps) * chi2.pdf(dof * Sobs / (Steo + Sn + eps), dof))
    
    # Remove any non-finite values and sum the result
    C11 = -np.sum(C11[np.isfinite(C11)])
    
    return C11


def cost_T_fit(K, Sobs, KB, iKfit, Tdis, dof=6, Sn=None, Kmol = 1.44e-7):
    if Sn is None:
        Sn = np.zeros_like(Sobs)
    
    # Find indices similar to the MATLAB implementation
    iKB = np.where(K <= KB)[0]
    if iKB.size == 0:
        iKB = K.size-1
    else:
        iKB = iKB[-1]

    iKnoi = np.where(Sobs[iKfit[0]:] < 2 * Sn[iKfit[0]:])[0]
    if iKnoi.size >0:
        iKnoi = iKnoi[0]
    else:
        iKnoi = Sobs[iKfit[0]:].size-1
    iKnoi+= iKfit[0]  # First index where Sobs < 2 * Sn
    
    iKcorr = np.arange(iKfit[0], min([iKB, iKnoi, iKfit[-1]]) + 1)
    
    # Initial Xi guess
    Xi0 = 6 * Kmol * np.sum((Sobs[iKcorr[1:]] + Sobs[iKcorr[:-1]]) * (K[iKcorr[1:]] - K[iKcorr[:-1]])) / 2
    XiN0 = 6 * Kmol * np.sum((Sn[iKcorr[1:]] + Sn[iKcorr[:-1]]) * (K[iKcorr[1:]] - K[iKcorr[:-1]])) / 2
    Xi0 -= XiN0
    
    # Correct for loss variance
    Steo = Tspec(Tdis, Xi0, KB, K, k = Kmol)
    XiT = 6 * Kmol * np.sum((Steo[iKcorr[1:]] + Steo[iKcorr[:-1]]) * (K[iKcorr[1:]] - K[iKcorr[:-1]])) / 2
    XiC = Xi0 * Xi0 / XiT
    
    # Recalculate Steo with corrected XiC
    Steo = Tspec(Tdis, XiC, KB, K, k = Kmol)
    
    # Calculate C11 using the previously defined cost_MLE function
    C11 = cost_MLE(Sobs[iKfit], Steo[iKfit], dof, Sn[iKfit])
    
    return C11, XiC

def Tspec(spc, a, b=None, c=None, k = 1.44e-7):
    if c is not None:
        Xi = a
        KB = b
        K = c
    else:
        Xi = a[0]
        KB = a[1]
        K = b

    if spc == 'K':
        # Roget 2 - Kraichnan spectrum
        q = 5.26
        phi = KB / np.sqrt(2 * q)
        y = K / phi
        f = y * np.exp(-np.sqrt(3) * y)
        SK = Xi / (2 * k * KB) * np.sqrt(2 * q) * f
    elif spc == 'B':
        # Roget 2 - Batchelor spectrum
        q = 3.9
        k = 1.44e-7
        phi = KB / np.sqrt(2 * q)
        y = K / phi
        f = y * (np.exp(-y**2 / 2) - y * np.sqrt(np.pi / 2) * (1 - erf(y / np.sqrt(2))))
        SK = Xi / (2 * k * KB) * np.sqrt(2 * q) * f
    else:
        raise ValueError("Invalid value for spc. Use 'K' or 'B'.")

    return SK

def meanabsdev(Sobs, Steo, Sn=None):
    """
    Calculate the mean absolute deviation.
    
    Parameters:
        Sobs : array_like
            Observed data.
        Steo : array_like
            Theoretical model data.
        Sn : array_like, optional
            Noise data. Defaults to an array of zeros with the same size as Steo.

    Returns:
        float
            Mean absolute deviation.
    """
    if Sn is None:
        Sn = np.zeros_like(Steo)

    MAD = np.mean(np.abs((Sobs / (Steo + Sn)) - np.mean(Sobs / (Steo + Sn))))
    
    return MAD


def thermistor_fit(f,phi, phi_n,W, Ns, P = np.nan,T =10., K1=0.1,  Tdis = "K", channel = "x",TRcor = False, mad_factor = 2., fs = 256, plt_flag=False, **kargs):
    #gets degrees of freedom
    dof = 1.9*Ns
    
    Kmol = thermal_diffusivity(T)
    visco = viscosity(T)
    Pr = visco/Kmol
    if Tdis == "K":
        q = 5.26
    elif Tdis =="B":
        q = 3.9
    else:
        raise ValueError("Invalid value for spc. Use 'K' or 'B'.")
    if "noise_cutoff" in kargs:
        noise_cutoff = kargs["noise_cutoff"]
    else:
        noise_cutoff = 2.
    if "low_cutoff" in kargs:
        low_cutoff = kargs["low_cutoff"]
    else:
        low_cutoff = 2.
    if "tau" in kargs:
        tau = kargs["tau"]
    else:
        tau = -999.
    if "pole" in kargs:
        if kargs["pole"] == "single" or kargs["pole"] == "double" or kargs["pole"]=="RBR":
            pole = kargs["pole"]
    else:
        pole = "double"
    if "fAA" in kargs:
        fAA = kargs["fAA"]
    else:
        fAA = 0.2*fs
    if "search_minimum" in kargs:
        search_minimum = kargs["search_minimum"]
    else:
        search_minimum = False
    if "do_Hcutoff" in kargs:
        do_Hcutoff = kargs["do_Hcutoff"]
    else:
        do_Hcutoff = True
    if "Hcutoff" in kargs:
        Hcutoff = kargs["Hcutoff"]
    else:
        Hcutoff = 0.1

    MADc = np.sqrt(2/dof)
    
    Xiv = np.nan;Xif=np.nan;KBT=np.nan;epsilon = np.nan;sXif = np.nan; sKBT = np.nan;  MADf = np.nan;MLKH=np.nan; LKHratio = np.nan; fit_flag = 0; Xic = np.nan; MADsh = np.nan
    
    
    
    W = np.abs(W)
    K = f/W;
    PSD = W*phi
    Sn = W*phi_n
    iK1 = np.where(K>=K1)[0][0]
    PSD0 = np.copy(PSD)
    if TRcor:
        #variance correction
        if tau == -999:
            F0 = 25*np.sqrt(W);
            tau = (2*np.pi*F0/np.sqrt(np.sqrt(2)-1))**(-1)
        if pole == "double":
            H = 1./(1 + (2*np.pi*tau*f)**2)**2;
        elif pole == "single":
            H = 1./(1 + (2*np.pi*tau*f)**2);
        elif pole == "RBR":
            #From Mathieu Dever
            tau1 = 0.075
            tau2 = 0.793
            taueff = 0.94*tau2 + 0.06*tau1
            H = (1 + (2*np.pi*f*taueff)**2) / ((1 + (2*np.pi*f*tau1)**2)*(1 + (2*np.pi*f*tau2)**2))
        PSD = PSD/H;
        Sn = Sn/H
        

    iAA = np.where(f<=fAA)[0]
    if iAA.size == 0:
        iAA = PSD.size-1
    else:
        iAA = iAA[-1]



    #maximum epsilon based on the maximum resolved wavenumber
    #KBmax = K[iKnM0]
    #epsilon_max = epsilon_from_KB(visco,Kmol, KBmax)
    #KBmax = np.max(K)
    # CV 2026/07/03 : I added the following 3 lines : H is unknown if TRcor is False
    F0 = 25*np.sqrt(W);
    tau = (2*np.pi*F0/np.sqrt(np.sqrt(2)-1))**(-1)
    H = 1./(1 + (2*np.pi*tau*f)**2)**2 
    icut = np.where(H<Hcutoff)[0]
    if icut.size > 0:
        icut = icut[0]
        KBmax = K[icut]
    else:
        KBmax = np.max(K)
        icut = K.size

    epsilon_max = epsilon_from_KB(visco,Kmol, KBmax)

    #cut either at iAA or icut
    icutcut = min([icut,iAA])
    
    #print(noise_cutoff)
    iin = np.where(PSD[iK1:] < noise_cutoff * Sn[iK1:])[0]
    if iin.size==0: ###It used to be 2*Sn!!!!
        #looks for the PSD minimum, otherwise, the last K
        #iminPSD = np.nanargmin(PSD)
        #if PSD[iminPSD]<10*noise_cutoff*Sn:
        #    iKnM0 = np.copy(iminPSD)
        #else:
        #
        #iKnM0 = PSD.size-1
        iKnM0 = iAA
    else:
        if search_minimum:
            #print(search_minimum)
            #30 Jan 2026 added minimum search otherwise use what is under else
            imin = np.nanargmin(PSD[iK1:icutcut+1])
            if (imin<4) or (imin == PSD[iK1:icutcut+1].size-1):
                iKnM0 = iin[0] + iK1 - 1
            else:
                iKnM0 = imin+iK1-1 #
        else:
            iKnM0 = iin[0] + iK1 - 1

    #does not integrate when time-response correction is large (added 8/7/2026)
    if do_Hcutoff:
        iKnM0 = min([iKnM0, icutcut])
            

    # First guess
    cont = False
    ikcor = np.arange(iK1, iKnM0+1)
  

    Xiv = 6 * Kmol * np.sum((PSD[ikcor[1:]] + PSD[ikcor[:-1]]) * (K[ikcor[1:]] - K[ikcor[:-1]])) / 2
    Xin = 6 * Kmol * np.sum((Sn[ikcor[1:]] + Sn[ikcor[:-1]]) * (K[ikcor[1:]] - K[ikcor[:-1]])) / 2



    if Xiv > 1.3*Xin:
        cont = True
    else:
        Xiv = 0

    Xiv -= Xin


    #fig, ax = plt.subplots()
    #ax.loglog(K, PSD)
    #ax.loglog(K[ikcor], PSD[ikcor], "ro")
    #ax.loglog(K, Sn)

    if cont:
        #if epsilon from shear is available, use it to get missing variance
        if "epsSH" in kargs:
            KBsh = KB_from_epsilon(visco,Kmol, kargs["epsSH"])
            BATsh = Tspec(Tdis, Xiv, KBsh, K, k = Kmol)
            XiT = 6 * Kmol * np.sum((BATsh[ikcor[1:]] + BATsh[ikcor[:-1]]) * (K[ikcor[1:]] - K[ikcor[:-1]])) / 2
            Xic = Xiv*Xiv/XiT
            BATsh2 = Tspec(Tdis, Xic, KBsh, K, k = Kmol)
            MADsh = meanabsdev( PSD[ikcor], BATsh2[ikcor], Sn[ikcor])


        # Fit parameters in the noise-free region
        #[Search 1]
        KF = 2 * K[iKnM0]
        Ktest = np.linspace(max([5 * K1, KF / 5]), KF * 5, 40)
        cost = np.full_like(Ktest, np.nan)
        Xif0 = np.full_like(Ktest, np.nan)
        dK = Ktest[1] - Ktest[0]

        for i in range(len(Ktest)):
            ks = 0.04 * Pr**(-0.5) * Ktest[i]
            Kf1 = max([ks, K1])
            iKf1 = np.where(K >= Kf1)[0][0]
            #ikfit = np.arange(iKf1, K.size)
            ikfit = np.arange(iKf1,iKnM0+1) #changed in 25 december 2025
            cost[i], Xif0[i] = cost_T_fit(K, PSD, Ktest[i], ikfit, Tdis, dof, Sn, Kmol = Kmol)


        valid = np.isfinite(cost)
        Ktest = Ktest[valid]
        cost = cost[valid]
        LKHtest = -cost

        ML = np.max(LKHtest)
        iML = np.argmax(LKHtest)
        KB00 = Ktest[iML]
        
        
        if low_cutoff>0:
            #print("Tessst")
            #[Search 2] (over the same range but excluding lower frequency)

            #add an aditional constrain on the lower wavenumber for stratified contitions, 
            #where there is an spectral decay at low wavenumbers (2024-09-26)
            BATf0 = Tspec(Tdis, Xif0[iML], Ktest[iML], K, k = Kmol)    
            #if plt_flag:
            #    fig, ax = plt.subplots()
            #    ax.loglog(K, PSD)
            #    ax.loglog(K,BATf0)
            #    ax.loglog(K,Sn)            
            iKinter = np.where(PSD<=low_cutoff*BATf0)[0]
            if iKinter.size>0:
                Kinter = K[iKinter[0]]
                #avoids large values of Kinter above the spectral through
                ikmin = np.nanargmin(PSD)
                kmin = K[ikmin]
                if Kinter>=kmin/2.:
                    Kinter = np.copy(K1)
            else:
                Kinter =  np.copy(K1)

            cost = np.full_like(Ktest, np.nan)
            Xif0 = np.full_like(Ktest, np.nan)
            dK = Ktest[1] - Ktest[0]

            for i in range(len(Ktest)):
                ks = 0.04 * Pr**(-0.5) * Ktest[i]
                Kf1 = max([ks, K1,Kinter])
                iKf1 = np.where(K >= Kf1)[0][0]
                #ikfit = np.arange(iKf1, K.size)
                ikfit = np.arange(iKf1,iKnM0+1) #changed in 25 december 2025
                cost[i], Xif0[i] = cost_T_fit(K, PSD, Ktest[i], ikfit, Tdis, dof, Sn, Kmol = Kmol)


            valid = np.isfinite(cost)
            Ktest = Ktest[valid]
            cost = cost[valid]
            LKHtest = -cost

            ML = np.max(LKHtest)
            iML = np.argmax(LKHtest)
            KB00 = Ktest[iML]
        
        
        #[Search 3] (restricted range)
        #estimate the range for next search
        ks = 0.04 * Pr**(-0.5) * KB00
        Kf1 = max([ks, K1])
        iKf1 = np.where(K >= Kf1)[0][0]
        #iKf1 = np.where(K >= K1)[0][0]
        #ikfit = np.arange(iKf1, K.size)
        ikfit = np.arange(iKf1,iKnM0+1) #changed in 25 december 2025
        MLp, Xip = cost_T_fit(K, PSD, KB00 + dK, ikfit, Tdis, dof, Sn, Kmol = Kmol)
        MLm, Xim = cost_T_fit(K, PSD, KB00 - dK, ikfit, Tdis, dof, Sn, Kmol = Kmol)
        MLp, MLm = -MLp, -MLm
        deltak0 = abs((2 * dK) / np.sqrt(2 * ML - MLm - MLp))

        deltak = np.nanmax([deltak0, dK])

        #add an aditional constrain on the lower wavenumber for stratified contitions, 
        #where there is an spectral decay at low wavenumbers (2024-09-26)
        BATf0 = Tspec(Tdis, Xif0[iML], Ktest[iML], K, k = Kmol)    
        #if plt_flag:
        #    fig, ax = plt.subplots()
        #    ax.loglog(K, PSD)
        #    ax.loglog(K,BATf0)
        #    ax.loglog(K,Sn)            
        iKinter = np.where(PSD<=2*BATf0)[0]
        if iKinter.size>0:
            Kinter = K[iKinter[0]]
            #avoids large values of Kinter above the spectral through
            ikmin = np.nanargmin(PSD)
            kmin = K[ikmin]
            if Kinter>=kmin/2.:
                Kinter = np.copy(K1)
        else:
            Kinter =  np.copy(K1)
        
        kmin2 = max([KB00 - deltak, K[iK1 + 1]])
        kmax2 = min([KB00 + deltak, 5 * K[-1]])
        Ktest = np.linspace(kmin2, kmax2, 40)
        cost = np.full_like(Ktest, np.nan)
        Xif0 = np.full_like(Ktest, np.nan)

        for i in range(len(Ktest)):
            ks = 0.04 * Pr**(-0.5) * Ktest[i]
            #Kf1 = max([ks, K1]) #until 2024-09-26
            Kf1 = max([ks, K1, Kinter])
            iKf1 = np.where(K >= Kf1)[0][0]
            #ikfit = np.arange(iKf1, K.size)
            ikfit = np.arange(iKf1,iKnM0+1) #changed in 25 december 2025
            cost[i], Xif0[i] = cost_T_fit(K, PSD, Ktest[i], ikfit, Tdis, dof, Sn, Kmol = Kmol)



        valid = np.isfinite(cost)
        Ktest = Ktest[valid]
        Xif0 = Xif0[valid]
        cost = cost[valid]
        LKHtest = -cost



        MLKH = np.max(LKHtest)
        iML = np.argmax(LKHtest)
        
        #gets the optimal KB
        Xif = Xif0[iML]
        KBT = Ktest[iML]
        epsilon  = epsilon_from_KB(visco,Kmol, KBT)
        
        BATf = Tspec(Tdis, Xif, KBT, K, k = Kmol)
        ks = 0.04 * Pr**(-0.5) * KBT

        
        #Kf1 = max([ks, K1])
        #again additional constrain
        if iKinter.size>0:
            #avoids large values of Kinter above the spectral through
            Kinter = K[iKinter[0]]
            ikmin = np.nanargmin(PSD)
            kmin = K[ikmin]
            if Kinter>=kmin/2.:
                Kinter = np.copy(K1)
        else:
            Kinter =  np.copy(K1)
        
        Kf1 = max([ks, K1, Kinter])
        iKf1 = np.where(K >= Kf1)[0][0]
        #ikfit = np.arange(iKf1, K.size)
        ikfit = np.arange(iKf1,iKnM0+1) #changed in 25 december 2025


        # Uncertainties in fitting parameters
        MLp, Xip = cost_T_fit(K, PSD, KBT + deltak, ikfit, Tdis, dof, Sn, Kmol = Kmol)
        MLm, Xim = cost_T_fit(K, PSD, KBT - deltak, ikfit, Tdis, dof, Sn, Kmol = Kmol)
        MLp, MLm = -MLp, -MLm
        sKBT = abs((2 * deltak) / np.sqrt(2 * MLKH - MLm - MLp))
        sXif = abs((Xip - Xim) / np.sqrt(2 * MLKH - MLm - MLp))

        # Polynomial fit (avoiding noisy part)
        #cross_noise = np.where(BATf < 1.5 * Sn)[0]
        #if cross_noise.size>0:
        #    cross_noise = cross_noise[0]
        #else:
        #    cross_noise = BATf.size-1    
        ikfitA = np.copy(ikfit)#np.arange(ikfit[0], min([ikfit[-1], cross_noise])) ##changed this 26 December 2025, because created trouble with LKHR
        logK = np.log(K[ikfitA])
        logS = np.log(PSD[ikfitA])
        pp = np.polyfit(logK, logS, 1)
        Sm = np.exp(np.polyval(pp, np.log(K)))
        LKHpol = -cost_MLE(PSD[ikfitA], Sm[ikfitA], dof, Sn[ikfitA])
        #recalculates MLKH to exclude noise
        #MLKH = -cost_MLE(PSD[ikfitA], BATf[ikfitA], dof, Sn[ikfitA]) # CV 2026/07/13 : this line was commented but could make a difference
                                                                      # --> it actually does not, so I keep it commented 
        LKHratio = (MLKH - LKHpol)
        LKHratio = np.log10(np.exp(1)) * LKHratio
        MADf = meanabsdev(PSD[ikfitA], BATf[ikfitA], Sn[ikfitA])

        fit_flag = False
        if LKHratio > 0 and MADf < mad_factor*MADc  and abs(sKBT) < 0.5 * abs(KBT):
            fit_flag = True

        # Plotting section
        if plt_flag != 0:
            fig, ax = plt.subplots(figsize = (5,4))
            #plt.clf()

            ax.loglog(K, PSD,color = "k")
            ax.loglog(K, PSD0,color = "gray")
            ax.loglog(K, Sn, color='red', linewidth=1, linestyle='--')
            #ax.loglog(K[ikcor], PSD[ikcor], 'ko', markersize=5, mfc = "r")

            if cont:
                ax.loglog(K, BATf, color="b", linestyle='--')
                ax.loglog(K, BATf+Sn, color='b', linestyle='dotted', lw = 2)
                ax.loglog(K[ikfitA], PSD[ikfitA], 'ko', markersize=5, mfc = "w")
                ax.loglog(K[ikfitA], Sm[ikfitA] + Sn[ikfitA], color = 'darkgreen')
                ax.axvline(KBT, color='black')
                if "epsSH" in kargs:
                    plt.loglog(K, BATsh2 + Sn, color = 'saddlebrown')
            else:
                ax.loglog(K[ikcor], PSD[ikcor], 'ko', markersize=5, mfc = "r")


            ax.set_ylim([1e-9, 1e-0])
            #ax.set_xlim([0.7, 1000])
            ax.set_xlim([1, 1000])
            ax.set_xlabel("K (cpm)")
            ax.set_ylabel("PSD (K$^2$ m$^{-2}$ cpm$^{-1}$)")
            ax.set_title("Channel %02d, pressure = %1.1f dbar"%(channel, P))
            ax.grid(True)
            string = "$\chi_{\\theta}^v = %1.2e$ K$^2$/s\n$\chi_{\\theta}^f = %1.2e$ K$^2$/s\n$\\varepsilon = %1.2e$ W/Kg"%(Xiv,Xif,epsilon)
            string += "\n$K_B = %1.1f$ cpm\nLKHR = %1.1f\nMAD = %1.2f (%1.2f)\nW=%1.2f m/s"%(KBT,LKHratio, MADf, mad_factor*MADc,W)
            ax.annotate(string, xy = (0.98,0.98), ha = "right", va = "top", xycoords = "axes fraction", fontsize = 7)
            # Annotations
            
            plt.tight_layout()
            plt.show()
            
    else:
        if plt_flag != 0:
            fig, ax = plt.subplots(figsize = (5,4))
            #plt.clf()
            ax.loglog(K, PSD,color = "k")
            ax.loglog(K, PSD0,color = "gray")
            ax.loglog(K, Sn, color='red', linewidth=1, linestyle='--')
            ax.set_ylim([1e-9, 1e-1])
            ax.set_xlim([0.7, 1000])
            ax.set_xlabel("K (cpm)")
            ax.set_ylabel("PSD (K$^2$ m$^{-2}$ cpm$^{-1}$)")
            ax.set_title("Channel %02d, pressure = %1.1f dbar"%(channel, P))
            ax.grid(True)
  
        print("Skip at P = %1.1f dbar because low signal to noise"%(P))
           
           
    # CV 2026/07/08 : I added ikcor, pp and KBsh to function's outputs to make plots afterwards  
    if ikcor.size==0: ikcor = np.nan*np.zeros(2) 
    try:
        pp
    except: 
        pp=np.nan
    try: 
        KBsh
    except:
        KBsh=np.nan
    return Xiv,Xif,KBT,epsilon,fit_flag, sXif,sKBT, MADf,MADc,MLKH,LKHratio, KBmax, epsilon_max, Xic, MADsh, [ikcor[0],ikcor[-1]], pp, KBsh
 

def merge_sensors(sn1,sn2,fl1=0,fl2=0, factor = 2.7):
    if type(fl1) == int:
        fl1 = np.full(sn1.shape, True)
    if type(fl2) == int:
        fl2 = np.full(sn2.shape, True)
    SN = np.full(sn1.shape, np.nan)
    d1 = sn1.size
    for i in range(d1):

        if (fl1[i] == 0 or np.isnan(sn1[i])) and (fl2[i] == 0 or np.isnan(sn2[i])):
            #print "case 1"
            SN[i] == np.nan
        elif fl1[i] == 0 or np.isnan(sn1[i]) or sn1[i]>factor*sn2[i]:
            #print "case 2"
            SN[i] = sn2[i]
        elif fl2[i] == 0 or np.isnan(sn2[i]) or sn2[i]>factor*sn1[i]:
            #print "case 3"
            SN[i] = sn1[i]
        else:
            #print "case 4"
            SN[i] = 0.5*(sn1[i]+sn2[i])
    return SN



###SHEAR
def viscosity(T):
    """
    Seawater viscosity calculation.
    
    Parameters:
    T : float or array-like
        Temperature in degrees Celsius.

    Returns:
    v : float or array-like
        Viscosity in m^2/s.
    """
    v = 1.792747 - 0.052126103 * T + 0.0005918645 * T**2
    v *= 10**-6
    return v


def nasmyth(eps,nu,k):
    # function psd =  nasmythspec(eps,nu,k)
    #
    # eps: epsilon in m2/s3
    # nu:  kinematic viscosity in m2/s
    # k:   wave number in cpm
    # psd: power spectral density of Nasmyth spectrum
    #      in 1/s2/cpm
    #
    # last changed 09/02/20 TF

    kkol = (eps/nu**3)**0.25; # 1/m in radian
    kdimless = k/kkol; # mixed dimless factor (in 1/rad),
    # this is due to a strange taste of Lueck (no literature.)
    pdimless = 8.05*kdimless**(1/3)/(1+(20*kdimless)**3.7);
    # pdimless in rad, (eps^3/nu)^0.25 in m/s2/rad
    psd = pdimless*(eps**3/nu)**0.25; # in 1/s2/cpm
    return psd

def shear_fit(f,phi, W, Ns, P = np.nan,T =10., K1=1, K2 = 1,Kn= 50., channel = "x", mad_factor = 2., fs = 256, plt_flag=False):
    #gets degrees of freedom
    dof = 1.9*Ns
    MADc = np.sqrt(2/dof)
    
    visco = viscosity(T)

    W = np.abs(W)
    K = f/W;
    PSD = W*phi
 
            
    #applies spatial response correction
    Hsr = 1/(1+(K/48)**2);
    PSD = PSD/Hsr

    #searches wavenumbers
    iK1 = np.where(K>=K1)[0][0]
    iK2 = np.where(K>=K2)[0][0]
    if iK2 == iK1:
        iK2+=2
    iKn = np.where(K>=Kn)[0][0]



    # EPSILON ITERATIVE CALCULATION
    iK3 = np.copy(iK2)
    K3 = K[iK3]
    Kc0 = np.copy(K2)
    flag = 0
    #i = 0
    while flag == 0:
        #print(i)
        #i+=1
        psd_hi = PSD[iK1 + 1 : iK3 + 1]
        psd_lo = PSD[iK1     : iK3]
        k_hi   = K[iK1 + 1   : iK3 + 1]
        k_lo   = K[iK1       : iK3]
        
        eps = 7.5 * visco * np.sum((psd_hi + psd_lo) * (k_hi - k_lo)) / 2.0

        Kc = (1.0 / (2.0 * np.pi)) * (eps / (visco**3))**0.25

        if Kc<=K[0]:
            kcm = K[0]
        else:
            ikcm = np.where(K<=Kc)[0][-1]
            kcm = K[ikcm]
        if Kc>K[-1]:
            kcp = K[-1]
        else:
            ikcp = np.where(K>=Kc)[0][0]
            kcp = K[ikcp]
        dK = kcp-kcm

        
        if abs(Kc - Kc0) <= dK:
            flag = 1
            iK3 = int(np.searchsorted(K, Kc, side='left'))
            # optional: clamp to valid range
            #iK3 = min(max(iK3, 0), len(K) - 1)
            
        elif Kc <= Kn:
            diff = (Kc - K3)
            # replicate MATLAB sign(diff) behavior, but avoid division by zero
            #inc = 0 if diff == 0 else (diff / abs(diff))  # will be +1 or -1
            inc = diff/np.abs(diff)
            iK3 = int(iK3 + inc)
            iK3 = min(max(iK3, 0), len(K) - 1)

            K3 = K[iK3]
            Kc0 = np.copy(Kc)

        else:
            iK3 = iKn
            flag = 2


    # Final recomputation 
    psd_hi = PSD[iK1 + 1 : iK3 + 1]
    psd_lo = PSD[iK1     : iK3]
    k_hi   = K[iK1 + 1   : iK3 + 1]
    k_lo   = K[iK1       : iK3]
    
    eps = 7.5 * visco * np.sum((psd_hi + psd_lo) * (k_hi - k_lo)) / 2.0
    Kc  = (1.0 / (2.0 * np.pi)) * (eps / (visco**3))**0.25
    NAS = nasmyth(eps,visco,K)
    varianceN=sum( (NAS[iK1+1:iK3+1]+NAS[iK1:iK3])* (K[iK1+1:iK3+1]-K[iK1:iK3]) )/2
    epsUN=7.5*visco*varianceN

    if np.abs(eps-epsUN)/epsUN>0.05:
        eps=eps*eps/epsUN;
        Kc=1/(2*np.pi)*(eps/visco**3)**(0.25);       
    NAS = nasmyth(eps,visco,K)
    
    MAD =  np.mean(np.abs(np.log10(PSD[iK1:iK3+1]/NAS[iK1:iK3+1])))
    #MAD =  meanabsdev(PSD[iK1:iK3], NAS[iK1:iK3], np.zeros(NAS[iK1:iK3].shape))
    fit_flag = False
    if MAD<2*MADc:
        fit_flag = True
    
        
        
    if plt_flag == True:
        fig, ax = plt.subplots(num=1, clear=True,figsize = (5,4))

        #theoretical spectrum
        k_grid = np.logspace(0, 3, 100)          # 10^0 ... 10^3, 100 pts
        eps_list = np.logspace(-11, -4, 9)       # 10^-11 ... 10^-3, 9 pts
        for eps_i in eps_list:
            ax.loglog(k_grid, nasmyth(eps_i, visco, k_grid),
                      color="skyblue", lw = 0.5)
            
        x_text = 1.5
        k_two = np.array([1.5, 2.0])
        for i in range(1, 9):  # 1..9 inclusive
            eps_i = 10.0 ** (-12 + i)  # matches MATLAB: 10^(-12+i)
            y_text = np.min(nasmyth(eps_i, visco, k_two))
            ax.text(x_text, y_text,
                    rf'$10^{{{(-12 + i)}}}$ W/kg',
                    color="blue",fontsize = 8)
            
        # --- measured/model spectra ---
        
        ax.loglog(K, PSD,  color='k')
        ax.loglog(K[:iK3+1], PSD[:iK3+1], 'ko', markersize=5, mfc = "w")
        ax.loglog(K, NAS,  color="blue", linewidth=2)

        ax.set_ylim(1e-9, 0)
        ax.set_yticks(10.**np.arange(-9,0))
        ax.set_xlim(1, 1e3)

        ax.axvline(Kc, color='k')
        ax.text(1.1 * Kc, 0.1, '$K_c = %1.1f$ cpm'%(Kc))
        
        ax.axvline(Kn, color='k', linestyle='--')
        ax.text(1.2, 1e-8,
                r'$\epsilon = $' + f'{eps:1.3e} W/kg\nMAD = ' + f'{MAD:1.2f}' + '(' + f'{(MADc*2):1.2f}' + ')',
                ha='left')
        


        # --- Title/labels (MATLAB: title/xlabel/ylabel) ---
        ax.set_title("Channel %02d, shear, pressure = %1.1f dbar"%(channel, P))
        ax.set_xlabel('k (cpm)')
        ax.set_ylabel(r'$S_{sh}$ (s$^{-2}$ cpm$^{-1}$)')
        #ax.grid(True)

    return eps,MAD, MADc,Kc,fit_flag





def mixed_layer_depth(z0, den0, Dd = 0.03, crit = "diff", z_min = 30., intrp = True):
    #Mixed layer calculation
    if crit != "diff" and crit != "grad" and crit != "DO":
        crit = "diff"
        print("Incorrect criterion, set to diff")
    c,f = den0.shape
    MLD = np.full(f, np.nan)
    for i in range(f):
        if z0.ndim ==1:
            z = np.copy(z0)
        else:
            z = z0[:,i]
        #den = np.sort(den0[:,i])
        den = den0[:,i]
        iif = np.isfinite(den+z)
        if np.sum(iif)<=1:
            continue
        den = den[iif]
        z = z[iif]
        if np.min(z0)>z_min:
            continue

        if crit == "diff":
            sden = den[0]
            denp = den-sden
            imld = np.where( denp>=Dd )[0]
            if imld.size == 0:
                MLD[i] = np.max(z)
            elif imld[0]>0:
                imld = imld[0]
                z2 = z[imld]
                z1 = z[imld-1]
                denp2 = denp[imld]
                denp1 = denp[imld-1]
                if intrp:
                    MLD[i] = (z2-z1)/(denp2-denp1)*(Dd - denp1) + z1
                else:
                    MLD[i] = (z1+z2)*0.5
            else:
                MLD[i] = np.max(z)
                #MLD[i] = z0[0,i]

        elif crit == "grad":
            grden = np.abs(first_centered_differences(z, den))
            imld = np.where(grden>=Dd)[0]
            if imld.size == 0:
                MLD[i] = np.max(z)
            elif imld[0]>0:
                imld = imld[0]
                z2 = z[imld]
                z1 = z[imld-1]
                grd2 = grden[imld]
                grd1 = grden[imld-1]
                if intrp:
                    MLD[i] = (z2-z1)/(grd2-grd1)*(Dd - grd1) + z1
                else:
                    MLD[i] = 0.5*(z1+z2)
            else:
                MLD[i] = z[0]

        if crit == "DO":
            sden = den[0]
            denp = den-sden
            imld = np.where( np.abs(denp)>=Dd )[0]
            if imld.size == 0:
                MLD[i] = np.max(z)
            elif imld[0]>0:
                imld = imld[0]
                z2 = z[imld]
                z1 = z[imld-1]
                MLD[i] = z1
            else:
                MLD[i] = np.max(z)
                #MLD[i] = z0[0,i]

    return MLD
