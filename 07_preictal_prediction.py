import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from tqdm import tqdm

def extract_band_energy(signal, fs=173.61):
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(len(signal), 1/fs)
    power_spectrum = np.abs(fft_vals)**2
    
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    band_energies = {}
    for band_name, (low, high) in bands.items():
        band_mask = (np.abs(fft_freqs) >= low) & (np.abs(fft_freqs) <= high)
        band_energies[band_name] = np.sum(power_spectrum[band_mask])
    
    return band_energies

def extract_window_features(window):
    band_energies = extract_band_energy(window)
    
    theta_beta_ratio = band_energies['theta'] / (band_energies['beta'] + 1e-10)
    theta_energy = band_energies['theta']
    total_energy = sum(band_energies.values())
    variance = np.var(window)
    delta_alpha_ratio = band_energies['delta'] / (band_energies['alpha'] + 1e-10)
    
    return {
        'theta_beta_ratio': theta_beta_ratio,
        'theta_energy': theta_energy,
        'total_energy': total_energy,
        'variance': variance,
        'delta_alpha_ratio': delta_alpha_ratio
    }

def sliding_window_analysis(signal, fs=173.61, window_size=30, hop_size=5):
    features_list = []
    times = []
    
    for i in range(0, len(signal) - window_size, hop_size):
        window = signal[i:i+window_size]
        features = extract_window_features(window)
        features_list.append(features)
        
        time = (i + window_size/2) / fs
        times.append(time)
    
    return features_list, np.array(times)

def compute_trend(values, window=3):
    if len(values) < window:
        return 0
    
    recent = values[-window:]
    x = np.arange(window)
    slope = np.polyfit(x, recent, 1)[0]
    return slope

def detect_transition_to_seizure(features_list, times, baseline_window=3):
    if len(features_list) < baseline_window + 3:
        return False, None, 0.0
    
    theta_beta_thresh = 0.10 
    theta_energy_thresh = 0.15 
    trend_thresh = 0.02      
    
    
    baseline_theta_beta = np.mean([f['theta_beta_ratio'] for f in features_list[:baseline_window]])
    baseline_theta_energy = np.mean([f['theta_energy'] for f in features_list[:baseline_window]])
    baseline_variance = np.mean([f['variance'] for f in features_list[:baseline_window]])
    
    theta_beta_history = [f['theta_beta_ratio'] for f in features_list[:baseline_window]]
    theta_energy_history = [f['theta_energy'] for f in features_list[:baseline_window]]
    
    max_confidence = 0.0
    best_detection_idx = None
    
    for i in range(baseline_window, len(features_list) - 1):
        current = features_list[i]
        
        theta_beta_history.append(current['theta_beta_ratio'])
        theta_energy_history.append(current['theta_energy'])
        
        theta_beta_increase = (current['theta_beta_ratio'] - baseline_theta_beta) / (baseline_theta_beta + 1e-10)
        theta_energy_increase = (current['theta_energy'] - baseline_theta_energy) / (baseline_theta_energy + 1e-10)
        
        theta_beta_trend = compute_trend(theta_beta_history, window=3)
        theta_energy_trend = compute_trend(theta_energy_history, window=3)
        
        theta_beta_trend_norm = theta_beta_trend / (baseline_theta_beta + 1e-10)
        theta_energy_trend_norm = theta_energy_trend / (baseline_theta_energy + 1e-10)
        
        variance_increase = current['variance'] / (baseline_variance + 1e-10)
        
        confidence = 0.0
        
        if theta_beta_increase > theta_beta_thresh:
            confidence += 0.4
        if theta_energy_increase > theta_energy_thresh:
            confidence += 0.3
        if theta_beta_trend_norm > trend_thresh:
            confidence += 0.2
        if variance_increase > 1.1:  
            confidence += 0.1
        
        if i < len(features_list) * 0.3: 
            confidence += 0.1
        
        if confidence > max_confidence:
            max_confidence = confidence
            best_detection_idx = i
    
    detected = max_confidence > 0.4
    
    return detected, best_detection_idx, max_confidence

def main():
    X_signals = np.load('X_signals.npy')
    y_binary = np.load('y_binary.npy')
    
    seizure_signals = X_signals[y_binary == 1][:8]
    
    fs = 173.61
    
    print(f"✓ Loaded {len(seizure_signals)} seizure signals")
    print(f"✓ Signal length: {len(seizure_signals[0])} samples ({len(seizure_signals[0])/fs:.2f}s)")
    
    results = []
    
    for idx, signal in enumerate(tqdm(seizure_signals, desc="Analyzing")):
        features_list, times = sliding_window_analysis(
            signal, fs=fs, window_size=30, hop_size=5
        )
        
        detected, detection_idx, confidence = detect_transition_to_seizure(
            features_list, times, baseline_window=3
        )
        
        results.append({
            'signal_idx': idx,
            'features': features_list,
            'times': times,
            'detected': detected,
            'detection_idx': detection_idx,
            'confidence': confidence,
            'signal': signal
        })
    
    fig, axes = plt.subplots(4, 2, figsize=(18, 16))
    axes = axes.flatten()
    
    detection_count = sum([r['detected'] for r in results])
    
    for idx in range(len(results)):
        result = results[idx]
        signal = result['signal']
        
        ax = axes[idx]
        ax2 = ax.twinx()
        
        time_signal = np.arange(len(signal)) / fs
        ax.plot(time_signal, signal, 'k-', linewidth=0.8, alpha=0.7, label='EEG')
        
        theta_beta_values = [f['theta_beta_ratio'] for f in result['features']]
        ax2.plot(result['times'], theta_beta_values, 'b-', linewidth=2, label='Theta/Beta')
        
        theta_energy_values = [f['theta_energy'] for f in result['features']]
        theta_energy_norm = np.array(theta_energy_values) / (np.max(theta_energy_values) + 1e-10)
        ax2.plot(result['times'], theta_energy_norm, 'r--', linewidth=1.5, label='Theta Energy', alpha=0.7)
        
        if result['detected']:
            detect_time = result['times'][result['detection_idx']]
            confidence = result['confidence']
            
            ax2.axvline(detect_time, color='lime', linestyle='--', linewidth=3, label='TRANSITION')
            ax2.scatter(detect_time, theta_beta_values[result['detection_idx']], 
                       s=400, color='lime', marker='*', zorder=5, edgecolors='black', linewidths=2)
            
            ax.annotate(f'DETECTED\nConf: {confidence:.2f}', 
                       xy=(detect_time, ax.get_ylim()[1]*0.8),
                       fontsize=10, fontweight='bold', color='darkgreen',
                       ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            title_text = f'Signal {idx+1}: DETECTED (Conf: {confidence:.2f})'
            color = 'green'
        else:
            title_text = f'Signal {idx+1}: NOT DETECTED (Conf: {result["confidence"]:.2f})'
            color = 'orange'
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax2.set_ylabel('Features', fontsize=11, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        ax.set_title(title_text, fontweight='bold', fontsize=12, color=color)
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time_signal[-1]])
    
    plt.suptitle(f'Seizure Transition Detection (Original Signals)\nDetected: {detection_count}/{len(results)}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('07_seizure_transition_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Rezultate:")
    print(f"Detected: {detection_count}/{len(results)} ({detection_count/len(results)*100:.1f}%)")
    print(f"Average confidence: {np.mean([r['confidence'] for r in results]):.2f}")
    print(f"\n07_seizure_transition_detection.png")

if __name__ == '__main__':
    main()