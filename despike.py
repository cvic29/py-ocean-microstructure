'''
CV 2026/10/05 : Despike a microstructure shear or acceleration signal. 
                
contact : clement.vic@ifremer.fr 
''' 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def despike(signal, params, var='shear', debug=False, single_pass=False):
    # Here are the parameters that are used in this routine 
    Fs     = params.fs_fast
    if var=='shear': 
        thresh = params.despike_sh[0]  
        LP_cut = params.despike_sh[1]  
        N      = int(params.despike_sh[2]*Fs)
    elif var=='acc':
        thresh = params.despike_A[0]  
        LP_cut = params.despike_A[1]  
        N      = int(params.despike_A[2]*Fs)

    # Define a filter  
    def butterworth_filter(data, cutoff, fs, order=1, btype='low'):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype=btype)
        return filtfilt(b, a, data)

    # Padding to avoid boundary effects
    pad_len = min(len(signal), 2 * int(Fs / LP_cut))
    padded_signal = np.pad(signal, pad_len, mode='constant', constant_values=0)

    # High-pass filter to remove low-frequency noise
    cutoff_hp = 0.5  # [Hz] in ODAS, 0.5 Hz is hard-coded whereas Lueck et al. recommends 0.1 Hz 
    signal_hp = butterworth_filter(padded_signal, cutoff_hp, Fs, order=1, btype='high')

    # Rectify the high-pass filtered signal
    signal_rectified = np.abs(signal_hp)

    # Apply smoothing using Butterworth low-pass filter
    signal_smoothed = butterworth_filter(signal_rectified, LP_cut, Fs)

    # Detect spikes based on the threshold
    ratio_signal = signal_rectified / signal_smoothed

    # Detect spikes only within the range of the original signal (after padding)
    spikes = np.where(ratio_signal[pad_len:-pad_len] > thresh)[0] + pad_len  # Adjust spike indices

    if 0: # quick check on filtering
        plt.figure();xlim=[3,5]
        time = np.arange(len(padded_signal))/Fs
        plt.subplot(311);plt.plot(time,padded_signal,'k',lw=0.4);plt.xlim(xlim);plt.ylim(-4,4)
        plt.subplot(312);plt.plot(time,signal_rectified,'k',lw=0.4) 
        plt.plot(time,signal_smoothed,'r',lw=0.4);plt.xlim(xlim);plt.ylim(0,3)   
        plt.subplot(313);plt.plot(time,ratio_signal,'k',lw=0.4)
        plt.plot(xlim,[thresh,thresh],'g',lw=0.4);plt.xlim(xlim);plt.ylim(0,15)          
        plt.savefig('tmp.pdf') 
        exit() 
 
    # Initialize the despiked signal
    y = np.copy(padded_signal)
    pass_count = 0

    # Calculate the neighborhood size (Fs / (4 * LP_cut))
    neighborhood_size = int(Fs / (4 * LP_cut))

    while len(spikes) > 0:
        pass_count += 1

        # Replace spikes in the original signal range with local mean
        for spike_idx in spikes:
            start = max(pad_len, spike_idx - N//2)  # Stay within the padded signal range
            stop = min(len(padded_signal), spike_idx + N)
            
            # Neighborhood sampling, avoid the padding region at both ends
            neighborhood = np.concatenate((
                padded_signal[max(pad_len, start - neighborhood_size):start], 
                padded_signal[stop:min(len(padded_signal), stop + neighborhood_size)]
            ))
            # Replace the original signal range only
            y[start:stop] = np.mean(neighborhood) if len(neighborhood) > 0 else 0

            #plt.figure()
            #plt.plot(signal,'k',lw=0.4) 
            #plt.plot(y,'r',lw=0.4)
            #plt.xlim(spike_idx-100,spike_idx+100) 
            #plt.savefig('tmp.pdf') 
            #exit()  

        # Recalculate signals for the next pass
        signal_hp = butterworth_filter(y, cutoff_hp, Fs, order=1, btype='high')
        signal_rectified = np.abs(signal_hp)
        signal_smoothed = butterworth_filter(signal_rectified, LP_cut, Fs)

        ratio_signal = signal_rectified / signal_smoothed
        ratio_signal = np.maximum(ratio_signal, 0)  # Ensure valid range

        # Recalculate spikes only in the original range
        spikes = np.where(ratio_signal[pad_len:-pad_len] > thresh)[0] + pad_len

        if single_pass:
            break

    # Remove padding
    y = y[pad_len:-pad_len]

    # Compute ratio of replaced data
    ratio = len(np.where(y != signal)[0]) / len(signal)

    if debug:
        plt.figure()
        plt.plot(np.arange(len(signal)) / Fs, signal, lw=0.4, label='Original Signal')
        plt.plot(np.arange(len(y)) / Fs, y, lw=0.4, label='Despiked Signal')
        #plt.xlim(880,900);plt.ylim(-5,5)
        plt.legend()
        plt.savefig('tmp_despike.pdf')
    
    return y, spikes - pad_len, pass_count, ratio

