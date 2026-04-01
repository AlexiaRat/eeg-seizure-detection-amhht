import numpy as np
import matplotlib.pyplot as plt
from PyEMD.CEEMDAN import CEEMDAN
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter

def create_hh_spectrogram_improved(signal, fs=173.61, n_bins_time=200, n_bins_freq=150):
   
    # descompunere CEEMDAN
    ceemdan = CEEMDAN(trials=50, epsilon=0.005)
    IMFs = ceemdan.ceemdan(signal)
    
    if IMFs.ndim == 1:
        IMFs = [IMFs]
    else:
        IMFs = [IMFs[i] for i in range(IMFs.shape[0])]
    
    print(f"Număr de IMF-uri: {len(IMFs)}")
    
    # 2. extrage toate datele timp-frecvență-amplitudine
    time = np.arange(len(signal)) / fs
    all_times = []
    all_freqs = []
    all_amps = []
    
    # foloseste toate IMFurile
    for idx, imf in enumerate(IMFs):
        try:
            analytic_signal = hilbert(imf)
            amplitude = np.abs(analytic_signal)
            
            phase = np.unwrap(np.angle(analytic_signal))
            instant_freq = np.diff(phase) / (2.0 * np.pi) * fs
            instant_freq = np.append(instant_freq, instant_freq[-1])
            
            # filtrare 
            valid_mask = (instant_freq >= 0.5) & (instant_freq <= 50)
            
            n_valid = np.sum(valid_mask)
            if n_valid > 10:
                all_times.extend(time[valid_mask])
                all_freqs.extend(instant_freq[valid_mask])
                all_amps.extend(amplitude[valid_mask])
                
                print(f"  IMF {idx+1}: {n_valid} puncte, freq: {instant_freq[valid_mask].min():.1f}-{instant_freq[valid_mask].max():.1f} Hz")
        except Exception as e:
            print(f"  IMF {idx+1}: Eroare - {e}")
            continue
    
    all_times = np.array(all_times)
    all_freqs = np.array(all_freqs)
    all_amps = np.array(all_amps)
    
    print(f"\nTotal puncte valide: {len(all_times)}")
    
    if len(all_times) < 100:
        print("Prea putine puncte!")
        return np.linspace(0, time[-1], n_bins_time), \
               np.linspace(0, 50, n_bins_freq), \
               np.zeros((n_bins_freq, n_bins_time))
    
    # histograma 2D ponderat
    heatmap, xedges, yedges = np.histogram2d(
        all_times, all_freqs,
        bins=[n_bins_time, n_bins_freq],
        range=[[0, time[-1]], [0, 50]],
        weights=all_amps
    )
    
    counts, _, _ = np.histogram2d(
        all_times, all_freqs,
        bins=[n_bins_time, n_bins_freq],
        range=[[0, time[-1]], [0, 50]]
    )
    
    heatmap = np.divide(heatmap.T, counts.T, 
                       out=np.zeros_like(heatmap.T), 
                       where=counts.T > 0)
    
    # smoothing
    heatmap_smooth = gaussian_filter(heatmap, sigma=1.2)
    
    
    heatmap_db = 20 * np.log10(heatmap_smooth + 1e-8)
    heatmap_db = np.clip(heatmap_db, -40, 60)
    
    return xedges, yedges, heatmap_db


def main():
    X = np.load('X_signals.npy')
    y_binary = np.load('y_binary.npy')
    
    print(f"Shape X: {X.shape}")
    print(f"Seizure samples: {np.sum(y_binary == 1)}")
    print(f"Normal samples: {np.sum(y_binary == 0)}")

    # alege semnale
    seizure_idx = np.where(y_binary == 1)[0][0]
    normal_idx = np.where(y_binary == 0)[0][0]
    
    seizure_signal = X[seizure_idx]
    normal_signal = X[normal_idx]
    
    print(f"\nSemnal seizure: index {seizure_idx}")
    print(f"Range: [{seizure_signal.min():.1f}, {seizure_signal.max():.1f}]")
    print(f"Mean: {seizure_signal.mean():.2f}, Std: {seizure_signal.std():.2f}")
    
    print(f"\nSemnal normal: index {normal_idx}")
    print(f"Range: [{normal_signal.min():.1f}, {normal_signal.max():.1f}]")
    print(f"Mean: {normal_signal.mean():.2f}, Std: {normal_signal.std():.2f}")

    ceemdan = CEEMDAN(trials=50, epsilon=0.005)
    IMFs = ceemdan.ceemdan(seizure_signal)
    
    if IMFs.ndim == 1:
        IMFs = [IMFs]
    else:
        IMFs = [IMFs[i] for i in range(IMFs.shape[0])]

    n_imfs = len(IMFs)
    print(f"✓ {n_imfs} IMF-uri generate")
    
    fig, axes = plt.subplots(n_imfs+1, 1, figsize=(14, 2*n_imfs))

    axes[0].plot(seizure_signal, 'k', linewidth=1)
    axes[0].set_title('Original Seizure Signal (Class 5)', fontweight='bold', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('Amplitude')

    for i, imf in enumerate(IMFs):
        axes[i+1].plot(imf, linewidth=0.8)
        axes[i+1].set_title(f'IMF {i+1}', fontsize=10)
        axes[i+1].grid(True, alpha=0.3)
        axes[i+1].set_ylabel('Amplitude')

    axes[-1].set_xlabel('Sample Point')
    plt.tight_layout()
    plt.savefig('04_CEEMDAN_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("04_CEEMDAN_decomposition.png")

    
    time_n, freq_n, amp_n = create_hh_spectrogram_improved(normal_signal, fs=173.61)
    
    time_s, freq_s, amp_s = create_hh_spectrogram_improved(seizure_signal, fs=173.61)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    vmin = min(np.percentile(amp_n, 2), np.percentile(amp_s, 2))
    vmax = max(np.percentile(amp_n, 98), np.percentile(amp_s, 98))
    
    im1 = ax1.imshow(amp_n, aspect='auto', origin='lower',
                     extent=[time_n[0], time_n[-1], freq_n[0], freq_n[-1]],
                     cmap='jet', interpolation='bilinear',
                     vmin=vmin, vmax=vmax)
    ax1.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax1.set_title('(a) Normal EEG Signal', fontsize=16, fontweight='bold')
    ax1.set_xlim([0, time_n[-1]])
    ax1.set_ylim([0, 50])

    im2 = ax2.imshow(amp_s, aspect='auto', origin='lower',
                     extent=[time_s[0], time_s[-1], freq_s[0], freq_s[-1]],
                     cmap='jet', interpolation='bilinear',
                     vmin=vmin, vmax=vmax)
    ax2.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax2.set_title('(b) Seizure EEG Signal', fontsize=16, fontweight='bold')
    ax2.set_xlim([0, time_s[-1]])
    ax2.set_ylim([0, 50])
    
    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation='vertical', 
                       pad=0.02, fraction=0.046)
    cbar.set_label('Amplitude (dB)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('05_HH_spectrogram_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("05_HH_spectrogram_comparison.png")
    
    
    fs = 173.61
    time = np.arange(len(seizure_signal)) / fs

    plt.figure(figsize=(14, 8))

    for i, imf in enumerate(IMFs[:5]):
        analytic_signal = hilbert(imf)
        amplitude = np.abs(analytic_signal)
        
        phase = np.unwrap(np.angle(analytic_signal))
        instant_freq = np.diff(phase) / (2.0 * np.pi) * fs
        instant_freq = np.append(instant_freq, instant_freq[-1])
        
        plt.scatter(time, instant_freq, c=amplitude, s=50, cmap='jet', alpha=0.9, 
                   vmin=0, vmax=np.percentile(amplitude, 95))

    plt.colorbar(label='Instantaneous Amplitude')
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Instantaneous Frequency (Hz)', fontsize=14)
    plt.title('Hilbert-Huang Spectrum - Seizure Signal (Scatter)', fontsize=16, fontweight='bold')
    plt.ylim([0, 50])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('05_HH_spectrum_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("05_HH_spectrum_scatter.png")

if __name__ == '__main__':
    main()