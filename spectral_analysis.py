'''
CV 2026/10/05 : Set of functions for spectral analysis. 
                The core function compute_spectrum is heavily inspired by csd_matrix_odas.m 
                contact : clement.vic@ifremer.fr 
''' 
import numpy as np
from scipy.fftpack import fft
from scipy.signal import get_window, detrend

def compute_spectrum(x, y=None, n_fft=256, rate=1.0, window_type='cosine', overlap=None, detrend_method='linear'):
    """
    Estimate the cross- and auto-spectrum of one or two real matrices.

    Parameters:
    x (np.ndarray): Matrix of signals to analyze (shape: channels x samples).
    y (np.ndarray, optional): Second matrix of signals (same shape as x).
                              If None, only auto-spectra of x are computed.
    n_fft (int): Length of FFT segments.
    rate (float): Sampling rate of the signals.
    window_type (str or np.ndarray): Window function name or custom window array.
    overlap (int, optional): Number of points to overlap between segments (default is n_fft/2).
    detrend_method (str): Detrending method ('none', 'constant', 'linear', 'parabolic').

    Returns:
    Cxy (np.ndarray): Cross-spectrum matrix of x and y (shape: x.shape[0], y.shape[0], NN).
    F (np.ndarray): Frequency vector.
    Cxx (np.ndarray): Auto-spectrum matrix of x (shape: x.shape[0], x.shape[0], NN).
    Cyy (np.ndarray or None): Auto-spectrum matrix of y (shape: y.shape[0], y.shape[0], NN), if y is provided.
    """
    
    if y is None:
        y = x  # Auto-spectrum case

    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same number of time samples (columns).")
    
    # Define window
    if isinstance(window_type, str):
        window = get_window(window_type, n_fft)
    else:
        window = np.asarray(window_type)
        if len(window) != n_fft:
            raise ValueError("Custom window must have length n_fft.")
    window = window / np.sqrt(np.mean(window**2)) # force to have a mean square of 1 

    # Define overlap
    if overlap is None:
        overlap = n_fft // 2

    num_segments = (x.shape[1] - overlap) // (n_fft - overlap)
    NN = n_fft // 2 + 1  # Number of positive frequency bins
    
    Cxx = np.zeros((x.shape[0], x.shape[0], NN), dtype=np.complex_)
    Cyy = np.zeros((y.shape[0], y.shape[0], NN), dtype=np.complex_) if y is not None else None
    Cxy = np.zeros((x.shape[0], y.shape[0], NN), dtype=np.complex_)
    
    for i in range(num_segments):
        start_idx = i * (n_fft - overlap)
        end_idx = start_idx + n_fft
        
        x_seg = x[:, start_idx:end_idx]
        y_seg = y[:, start_idx:end_idx]

        # Apply detrending
        if detrend_method in ['constant','linear']:
            x_seg = detrend(x_seg, axis=1, type=detrend_method)
            y_seg = detrend(y_seg, axis=1, type=detrend_method)
        elif detrend_method == 'parabolic':
            # Fit a quadratic polynomial and remove it
            t = np.arange(x_seg.shape[1])
            for m in range(x.shape[0]): 
                poly_coeffs = np.polyfit(t, x_seg[m,:], 2)
                trend = np.polyval(poly_coeffs, t)
                x_seg[m,:] -= trend
            for n in range(y.shape[0]): 
                poly_coeffs = np.polyfit(t, y_seg[n,:], 2)
                trend = np.polyval(poly_coeffs, t)
                y_seg[n,:] -= trend
        elif detrend_method == 'none': 
            x_seg = x_seg; y_seg = y_seg 
        else:
            raise ValueError(f"Unknown detrending method: {method}")

        # Apply window
        x_seg *= window[np.newaxis, :]
        y_seg *= window[np.newaxis, :]

        # Compute FFT
        fft_x = fft(x_seg, axis=1)[:, :NN]
        fft_y = fft(y_seg, axis=1)[:, :NN]

        # Compute cross- and auto-spectra
        for m in range(x.shape[0]):
            for n in range(y.shape[0]):
                Cxy[m, n, :] += fft_x[m, :] * np.conj(fft_y[n, :])
        
        for m in range(x.shape[0]):
            for n in range(x.shape[0]):
                Cxx[m, n, :] += fft_x[m, :] * np.conj(fft_x[n, :])

        if y is not None:
            for m in range(y.shape[0]):
                for n in range(y.shape[0]):
                    Cyy[m, n, :] += fft_y[m, :] * np.conj(fft_y[n, :])

    # Normalize spectra
    Cxy /= num_segments * n_fft * rate / 2
    Cxx /= num_segments * n_fft * rate / 2
    if y is not None:
        Cyy /= num_segments * n_fft * rate / 2

    # Generate frequency vector
    F = np.linspace(0, rate / 2, NN)

    return Cxy, F, Cxx, Cyy if y is not None else None


def clean_shear_spec(U, A, n_fft, rate):
    """
    Remove acceleration contamination from shear probe signals using spectral processing.
    
    Parameters:
        A (np.ndarray): Acceleration signals (columns = Ax, Ay, Az).
        U (np.ndarray): Shear probe signals (columns = probes).
        n_fft (int): Length of FFT used for auto- and cross-spectrum calculation.
        rate (float): Sampling rate in Hz.

    Returns:
        clean_UU (np.ndarray): Cleaned shear spectra.
        AA (np.ndarray): Acceleration auto-spectrum.
        UU (np.ndarray): Shear auto-spectrum before noise removal.
        UA (np.ndarray): Cross-spectrum of shear and acceleration.
        F (np.ndarray): Frequency vector.
    """
    if A.shape[1] != U.shape[1]:
        raise ValueError("Acceleration and shear matrices must have the same number of rows.")

    # Compute auto- and cross-spectra 
    UA, F, UU, AA = compute_spectrum(U, A, n_fft, rate)

    clean_UU = np.zeros_like(UU, dtype=np.complex_)

    # Apply Goodman noise removal method
    for i in range(len(F)):
        clean_UU[:,:,i] = UU[:,:,i] - (UA[:,:,i] @ np.linalg.pinv(AA[:,:,i])) @ UA[:,:,i].conj().T  

    clean_UU = np.real(clean_UU).squeeze()
    UU = np.real(UU).squeeze()
    AA = np.real(AA).squeeze()
    UA = np.real(UA).squeeze()

    # Correct for bias due to finite number of FFT segments
    num_segments = (2 * U.shape[1]) // n_fft - 1
    vibration_signals = A.shape[0]
    R = 1 / (1 - 1.02 * vibration_signals / num_segments)

    clean_UU *= R

    return clean_UU, F, AA, UU, UA

def spec_integral(ku_nondim):
    # ku_nondim is the nondimensional upper wavenumber for spectral integration 
    # Lueck (2022a) model, as in Eq (9) in Lueck et al. (2024)
    ku_nondim = ku_nondim**(4./3.)  
    IL = np.tanh(65.5*ku_nondim) - 9.*ku_nondim*np.exp(-54.5*ku_nondim) 
    return IL 

def spectral_model(k_nondim,model='nasmyth'):
    if model=='Nasmyth': 
        psi = 8.05*(k_nondim)**(1./3) / ( 1 + (20.6*k_nondim)**3.715 ) 
    if model=='Lueck': 
        y = (k_nondim/0.015)**2 
        f1 = 8.048*k_nondim**(1./3) / ( 1 + (21.7*k_nondim)**3 )
        f2 = 1 / ( 1 + (6.6*k_nondim)**(5./2) )
        f3 = 1 + 0.36*y / ( (y-1)**2 + 2*y )
        psi = f1 * f2 * f3  
    return psi

