import numpy as np
from PyEMD.EMD import EMD
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew
import pywt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def compute_spectral_flatness(signal):
    
    fft_mag = np.abs(np.fft.fft(signal))
    fft_mag = fft_mag[:len(fft_mag)//2]  # doar frecvente pozitive
    fft_mag = fft_mag[fft_mag > 1e-10]   # evita log(0)
    
    if len(fft_mag) < 2:
        return 1.0  # considerc zgomot daca nu avem date
    
    # geometric mean (in log space pentru stabilitate numerica)
    log_mean = np.mean(np.log(fft_mag))
    geometric_mean = np.exp(log_mean)
    
    # Arithmetic mean
    arithmetic_mean = np.mean(fft_mag)
    
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
    
    return spectral_flatness


def compute_effective_bandwidth(signal, fs=173.61):
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    positive_mask = freqs >= 0
    freqs = freqs[positive_mask]
    power = np.abs(fft_vals[positive_mask])**2
    
    total_power = np.sum(power) + 1e-10
    
    # frecventa medie ponderata
    mean_freq = np.sum(freqs * power) / total_power
    
    # bandwidth = deviatia standard a frecventei
    bandwidth = np.sqrt(np.sum((freqs - mean_freq)**2 * power) / total_power)
    
    return bandwidth, mean_freq


def adaptive_imf_selection_with_spectral_flatness(IMFs, signal, fs=173.61):
    selected_imfs = []
    selection_info = {
        'total_imfs': len(IMFs),
        'selected_indices': [],
        'rejection_reasons': [],
        'spectral_flatness_values': [],
        'correlation_values': [],
        'bandwidth_values': []
    }
    
    total_energy = np.sum(signal**2) + 1e-10
    
    # pre-calculeaza statistici pentru threshold-uri adaptive
    correlations = []
    for imf in IMFs:
        if len(imf) == len(signal):
            try:
                corr = abs(np.corrcoef(signal, imf)[0, 1])
                if not np.isnan(corr):
                    correlations.append(corr)
            except:
                continue

    corr_threshold = np.percentile(correlations, 25) if len(correlations) > 0 else 0.05
    corr_threshold = max(corr_threshold, 0.05)  # minim 0.05
    
    for idx, imf in enumerate(IMFs):
        rejection_reason = None
        
        if len(imf) != len(signal):
            rejection_reason = "length_mismatch"
            selection_info['rejection_reasons'].append((idx, rejection_reason))
            continue
        
        try:
            # corelatie
            correlation = abs(np.corrcoef(signal, imf)[0, 1])
            if np.isnan(correlation):
                correlation = 0
            selection_info['correlation_values'].append((idx, correlation))
            
            if correlation < corr_threshold:
                rejection_reason = f"low_correlation ({correlation:.3f} < {corr_threshold:.3f})"
                selection_info['rejection_reasons'].append((idx, rejection_reason))
                continue
            
            # energie
            imf_energy = np.sum(imf**2)
            energy_ratio = imf_energy / total_energy
            
            if energy_ratio < 0.005:
                rejection_reason = f"low_energy ({energy_ratio:.4f})"
                selection_info['rejection_reasons'].append((idx, rejection_reason))
                continue
            
            # spectral flatness
            spectral_flatness = compute_spectral_flatness(imf)
            selection_info['spectral_flatness_values'].append((idx, spectral_flatness))
            
            if spectral_flatness > 0.7:  # prea aproape de zgomot alb
                rejection_reason = f"high_spectral_flatness ({spectral_flatness:.3f} > 0.7)"
                selection_info['rejection_reasons'].append((idx, rejection_reason))
                continue
            
            bandwidth, mean_freq = compute_effective_bandwidth(imf, fs)
            selection_info['bandwidth_values'].append((idx, bandwidth))
            
            if bandwidth < 0.5:  # prea ingust
                rejection_reason = f"low_bandwidth ({bandwidth:.2f} Hz)"
                selection_info['rejection_reasons'].append((idx, rejection_reason))
                continue
            
            if not (0.3 < mean_freq < 60):  # in afara benzii EEG
                rejection_reason = f"freq_out_of_range ({mean_freq:.1f} Hz)"
                selection_info['rejection_reasons'].append((idx, rejection_reason))
                continue
            
            selected_imfs.append(imf)
            selection_info['selected_indices'].append(idx)
            
        except Exception as e:
            rejection_reason = f"error: {str(e)}"
            selection_info['rejection_reasons'].append((idx, rejection_reason))
            continue
    
    if len(selected_imfs) == 0:
        selected_imfs = [IMFs[i] for i in range(min(5, len(IMFs))) if len(IMFs[i]) == len(signal)]
        selection_info['fallback_used'] = True
    
    if len(selected_imfs) > 8:
        energies = [np.sum(imf**2) for imf in selected_imfs]
        top_indices = np.argsort(energies)[-8:]
        selected_imfs = [selected_imfs[i] for i in sorted(top_indices)]
    
    selection_info['final_count'] = len(selected_imfs)
    
    return selected_imfs, selection_info



def hjorth_parameters(signal):
    try:
        activity = np.var(signal)
        first_deriv = np.diff(signal)
        mobility = np.sqrt(np.var(first_deriv) / (activity + 1e-10))
        second_deriv = np.diff(first_deriv)
        complexity = np.sqrt(np.var(second_deriv) / (np.var(first_deriv) + 1e-10)) / (mobility + 1e-10)
        return activity, mobility, complexity
    except:
        return 0, 0, 0



def sample_entropy_fast(signal, m=2, r_factor=0.2):
    try:
        N = len(signal)
        if N < 10:
            return 0
        
        r = r_factor * np.std(signal)
        if r < 1e-10:
            return 0
        
        
        templates_m = np.array([signal[i:i+m] for i in range(N-m)])
        templates_m1 = np.array([signal[i:i+m+1] for i in range(N-m-1)])
        
        # calculeaza distante Chebyshev
        def count_matches(templates, r):
            count = 0
            n = len(templates)
            for i in range(n):
                for j in range(i+1, n):
                    if np.max(np.abs(templates[i] - templates[j])) < r:
                        count += 2 
            return count
        
        B = count_matches(templates_m, r)
        A = count_matches(templates_m1, r)
        
        if B == 0:
            return 0
        
        return -np.log((A + 1e-10) / (B + 1e-10))
    except:
        return 0



def extract_wavelet_features(signal, wavelet='db4', level=5):
    # DWT features
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        features = []
        for coeff in coeffs:
            features.append(np.sum(coeff**2))
            prob = np.abs(coeff) / (np.sum(np.abs(coeff)) + 1e-10)
            features.append(-np.sum(prob * np.log(prob + 1e-10)))
            features.append(np.std(coeff))
        return features
    except:
        return [0] * 18



def extract_time_domain_features(signal):
    try:
        features = []
        mad = np.mean(np.abs(signal - np.mean(signal)))
        features.append(mad)
        rms = np.sqrt(np.mean(signal**2))
        features.append(rms)
        features.append(np.ptp(signal))
        features.append(np.max(np.abs(signal)) / (rms + 1e-10))
        features.append(rms / (np.mean(np.abs(signal)) + 1e-10))
        features.append(np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-10))
        features.append(np.max(np.abs(signal)) / (np.mean(np.sqrt(np.abs(signal)))**2 + 1e-10))
        return features
    except:
        return [0] * 7



def hurst_exponent(signal):
    try:
        N = len(signal)
        if N < 20:
            return 0.5
        
        max_lag = min(N // 4, 20)
        lags = range(2, max_lag)
        tau = []
        
        for lag in lags:
            tau.append(np.std(signal[lag:] - signal[:-lag]))
        
        if len(tau) > 2:
            tau = np.array(tau)
            lags = np.array(list(lags))
            valid = tau > 0
            if np.sum(valid) > 2:
                poly = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)
                return poly[0] * 0.5
        return 0.5
    except:
        return 0.5


def detrended_fluctuation_analysis(signal):
    try:
        N = len(signal)
        if N < 16:
            return 1.0
       
        y = np.cumsum(signal - np.mean(signal))
        
        scales = np.unique(np.logspace(0.5, np.log10(N//4), 8).astype(int))
        scales = scales[scales >= 4]
        
        fluctuations = []
        valid_scales = []
        
        for scale in scales:
            n_segments = N // scale
            if n_segments < 1:
                continue
            
            rms_list = []
            for i in range(n_segments):
                segment = y[i*scale:(i+1)*scale]
                if len(segment) < 2:
                    continue
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                rms_list.append(np.sqrt(np.mean((segment - trend)**2)))
            
            if len(rms_list) > 0:
                fluctuations.append(np.mean(rms_list))
                valid_scales.append(scale)
        
        if len(fluctuations) > 2:
            coeffs = np.polyfit(np.log(valid_scales), np.log(fluctuations), 1)
            return coeffs[0]
        return 1.0
    except:
        return 1.0



def extract_band_energy(signal, fs=173.61):
    try:
        fft_vals = np.fft.fft(signal)
        fft_freqs = np.fft.fftfreq(len(signal), 1/fs)
        power_spectrum = np.abs(fft_vals)**2
        
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50),
            'high_gamma': (50, 80)
        }
        
        band_energies = {}
        for band_name, (low, high) in bands.items():
            band_mask = (np.abs(fft_freqs) >= low) & (np.abs(fft_freqs) <= high)
            band_energies[band_name] = np.sum(power_spectrum[band_mask])
        
        return band_energies
    except:
        return {b: 0 for b in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']}


def extract_am_hht_features(signal, fs=173.61):
    
    features = []
    
    try:
        emd = EMD()
        emd.FIXE_H = 0.05 
        IMFs_raw = emd.emd(signal)
        
        if IMFs_raw is None:
            return np.zeros(80)
        
        if IMFs_raw.ndim == 1:
            IMFs_all = [IMFs_raw]
        else:
            IMFs_all = [IMFs_raw[i] for i in range(IMFs_raw.shape[0])]
        
        IMFs, selection_info = adaptive_imf_selection_with_spectral_flatness(
            IMFs_all, signal, fs
        )
        
    except Exception as e:
        return np.zeros(80)
    
    for i in range(8):
        if i < len(IMFs):
            imf = IMFs[i]
            try:
                energy = np.sum(imf**2)
                features.append(energy)
                prob = np.abs(imf) / (np.sum(np.abs(imf)) + 1e-10)
                entropy = -np.sum(prob * np.log(prob + 1e-10))
                features.append(entropy)
                features.append(np.mean(np.abs(imf)))
            except:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0])
    
    band_energies = extract_band_energy(signal, fs)
    total_band = sum(band_energies.values()) + 1e-10
    
    features.append(band_energies['theta'] / (band_energies['beta'] + 1e-10))   # Theta/Beta
    features.append(band_energies['delta'] / (band_energies['alpha'] + 1e-10))  # Delta/Alpha
    features.append(band_energies['gamma'] / total_band)                         # Gamma ratio
    features.append(band_energies['alpha'] / (band_energies['beta'] + 1e-10))   # Alpha/Beta
    features.append(band_energies['theta'] / (band_energies['alpha'] + 1e-10))  # Theta/Alpha
    features.append(band_energies['high_gamma'] / total_band)                    # High Gamma ratio
    
    for i in range(5):
        if i < len(IMFs):
            try:
                analytic = hilbert(IMFs[i])
                phase = np.unwrap(np.angle(analytic))
                inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
                features.append(np.var(inst_freq))
            except:
                features.append(0)
        else:
            features.append(0)
    
    for i in range(5):
        if i < len(IMFs):
            try:
                zcr = np.sum(np.diff(np.sign(IMFs[i])) != 0)
                features.append(zcr)
            except:
                features.append(0)
        else:
            features.append(0)
    
    try:
        features.append(kurtosis(IMFs[0]) if len(IMFs) > 0 else 0)
        features.append(skew(IMFs[0]) if len(IMFs) > 0 else 0)
        features.append(np.std(IMFs[0]) if len(IMFs) > 0 else 0)
        features.append(kurtosis(IMFs[1]) if len(IMFs) > 1 else 0)
        features.append(skew(IMFs[1]) if len(IMFs) > 1 else 0)
    except:
        features.extend([0, 0, 0, 0, 0])
    
    activity, mobility, complexity = hjorth_parameters(signal)
    features.extend([activity, mobility, complexity])
    
    features.append(sample_entropy_fast(signal, m=2))
    features.append(sample_entropy_fast(signal, m=3))
    
    wavelet_feats = extract_wavelet_features(signal)
    features.extend(wavelet_feats)
    
    time_feats = extract_time_domain_features(signal)
    features.extend(time_feats)
    
    features.append(hurst_exponent(signal))
    features.append(detrended_fluctuation_analysis(signal))
    
    features.append(compute_spectral_flatness(signal))
    
    bw, mean_f = compute_effective_bandwidth(signal, fs)
    features.append(bw)
    features.append(mean_f)
    
    features = [f if np.isfinite(f) else 0 for f in features]
    features = features[:80]
    while len(features) < 80:
        features.append(0)
    
    return np.array(features, dtype=np.float64)

def main():
    try:
        X = np.load('X_signals.npy')
        y_binary = np.load('y_binary.npy')
    except FileNotFoundError:
        return
    
    print(f"Loaded: {X.shape[0]} signals, {X.shape[1]} samples each")
    print(f"Seizure: {np.sum(y_binary == 1)}, Normal: {np.sum(y_binary == 0)}")
    
    n_samples = min(2000, np.sum(y_binary == 1), np.sum(y_binary == 0))
    print(f"\nExtracting features for {n_samples} samples per class")
    
    seizure_indices = np.where(y_binary == 1)[0][:n_samples]
    normal_indices = np.where(y_binary == 0)[0][:n_samples]
    
    features_seizure = []
    for idx in tqdm(seizure_indices, desc="   Seizure"):
        feat = extract_am_hht_features(X[idx])
        features_seizure.append(feat)
    features_seizure = np.array(features_seizure)
    
    features_normal = []
    for idx in tqdm(normal_indices, desc="   Normal"):
        feat = extract_am_hht_features(X[idx])
        features_normal.append(feat)
    features_normal = np.array(features_normal)
    
    X_features = np.vstack([features_seizure, features_normal])
    y_labels = np.hstack([np.ones(len(seizure_indices)), np.zeros(len(normal_indices))])
    
    valid_mask = np.all(np.isfinite(X_features), axis=1)
    valid_mask &= np.sum(np.abs(X_features), axis=1) > 0
    
    X_features = X_features[valid_mask]
    y_labels = y_labels[valid_mask]
    
    np.save('X_adaptive_features.npy', X_features)
    np.save('y_adaptive_labels.npy', y_labels)
    
    print(f"Features extracted: {X_features.shape}")
    print(f"Valid samples: {np.sum(valid_mask)}/{len(valid_mask)}")
    print(f"X_adaptive_features.npy, y_adaptive_labels.npy")
   


if __name__ == '__main__':
    main()