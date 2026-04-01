import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, 
                            precision_score, f1_score, roc_curve, auc)
from PyEMD.EMD import EMD
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew
import pywt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def extract_fft_features(signal, fs=173.61):
    features = []
    
    N = len(signal)
    fft_vals = fft(signal)
    fft_freqs = fftfreq(N, 1/fs)
    
    positive_mask = fft_freqs >= 0
    fft_magnitude = np.abs(fft_vals[positive_mask])
    freqs = fft_freqs[positive_mask]
    power_spectrum = fft_magnitude ** 2
    total_power = np.sum(power_spectrum) + 1e-10
    
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    band_powers = {}
    for band_name, (low, high) in bands.items():
        band_mask = (freqs >= low) & (freqs <= high)
        band_power = np.sum(power_spectrum[band_mask])
        band_powers[band_name] = band_power
        features.append(band_power / total_power)
    
    features.append(band_powers['theta'] / (band_powers['beta'] + 1e-10))
    features.append(band_powers['alpha'] / (band_powers['beta'] + 1e-10))
    features.append(band_powers['delta'] / (band_powers['alpha'] + 1e-10))
    
    spectral_centroid = np.sum(freqs * power_spectrum) / total_power
    features.append(spectral_centroid)
    
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power_spectrum) / total_power)
    features.append(spectral_spread)
    
    prob = power_spectrum / total_power
    prob = prob[prob > 0]
    spectral_entropy = -np.sum(prob * np.log(prob + 1e-10))
    features.append(spectral_entropy)
    
    peak_freq = freqs[np.argmax(power_spectrum)] if len(power_spectrum) > 0 else 0
    features.append(peak_freq)
    
    mean_freq = np.sum(freqs * power_spectrum) / total_power
    features.append(mean_freq)
    
    bandwidth = np.sqrt(np.sum(((freqs - mean_freq) ** 2) * power_spectrum) / total_power)
    features.append(bandwidth)
    
    return np.array(features[:15])


def extract_dwt_features(signal, wavelet='db4', level=5):
    """Features bazate pe DWT"""
    features = []
    
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        for coeff in coeffs:
            energy = np.sum(coeff ** 2)
            features.append(energy)
            
            prob = np.abs(coeff) / (np.sum(np.abs(coeff)) + 1e-10)
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            features.append(entropy)
            
            features.append(np.std(coeff))
        
        total_energy = sum([np.sum(c ** 2) for c in coeffs]) + 1e-10
        for coeff in coeffs[:3]:
            features.append(np.sum(coeff ** 2) / total_energy)
    except:
        features = [0] * 21
    
    return np.array(features[:15])


def extract_hht_features(signal, fs=173.61, n_imfs=5):
    features = []
    
    try:
        emd = EMD()
        IMFs = emd.emd(signal)
        
        if IMFs is None:
            return np.zeros(15)
        
        if IMFs.ndim == 1:
            IMFs = [IMFs]
        else:
            IMFs = [IMFs[i] for i in range(min(IMFs.shape[0], n_imfs))]
    except:
        return np.zeros(15)
    
    for i in range(n_imfs):
        if i < len(IMFs):
            imf = IMFs[i]
            try:
                energy = np.sum(imf ** 2)
                features.append(energy)
                
                prob = np.abs(imf) / (np.sum(np.abs(imf)) + 1e-10)
                entropy = -np.sum(prob * np.log(prob + 1e-10))
                features.append(entropy)
                
                features.append(np.mean(np.abs(imf)))
            except:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0])
    
    return np.array(features[:15])


def compute_spectral_flatness(signal):
    fft_mag = np.abs(np.fft.fft(signal))
    fft_mag = fft_mag[:len(fft_mag)//2]
    fft_mag = fft_mag[fft_mag > 1e-10]
    
    if len(fft_mag) < 2:
        return 1.0
    
    geometric_mean = np.exp(np.mean(np.log(fft_mag)))
    arithmetic_mean = np.mean(fft_mag)
    
    return geometric_mean / (arithmetic_mean + 1e-10)


def adaptive_imf_selection(IMFs, signal, fs=173.61):
    selected = []
    total_energy = np.sum(signal**2) + 1e-10
    
    for imf in IMFs:
        if len(imf) != len(signal):
            continue
        
        try:
            corr = abs(np.corrcoef(signal, imf)[0, 1])
            if np.isnan(corr) or corr < 0.05:
                continue
            
            energy_ratio = np.sum(imf**2) / total_energy
            if energy_ratio < 0.005:
                continue
            
            sf = compute_spectral_flatness(imf)
            if sf > 0.7:
                continue
            
            selected.append(imf)
        except:
            continue
    
    if len(selected) == 0:
        selected = [IMFs[i] for i in range(min(5, len(IMFs))) if len(IMFs[i]) == len(signal)]
    
    return selected[:8]


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


def sample_entropy_fast(signal, m=2, r=0.2):
    try:
        N = len(signal)
        if N < 10:
            return 0
        r = r * np.std(signal)
        if r < 1e-10:
            return 0
        
        count_m = 0
        count_m1 = 0
        
        for i in range(N - m):
            for j in range(i + 1, N - m):
                if np.max(np.abs(signal[i:i+m] - signal[j:j+m])) < r:
                    count_m += 1
                    if i < N - m - 1 and j < N - m - 1:
                        if abs(signal[i+m] - signal[j+m]) < r:
                            count_m1 += 1
        
        if count_m == 0:
            return 0
        return -np.log((count_m1 + 1) / (count_m + 1))
    except:
        return 0


def extract_am_hht_features(signal, fs=173.61):
    features = []
    
    try:
        emd = EMD()
        IMFs_raw = emd.emd(signal)
        
        if IMFs_raw is None:
            return np.zeros(80)
        
        if IMFs_raw.ndim == 1:
            IMFs_all = [IMFs_raw]
        else:
            IMFs_all = [IMFs_raw[i] for i in range(IMFs_raw.shape[0])]
        
        IMFs = adaptive_imf_selection(IMFs_all, signal, fs)
    except:
        return np.zeros(80)
    
    for i in range(8):
        if i < len(IMFs):
            imf = IMFs[i]
            try:
                features.append(np.sum(imf**2))
                prob = np.abs(imf) / (np.sum(np.abs(imf)) + 1e-10)
                features.append(-np.sum(prob * np.log(prob + 1e-10)))
                features.append(np.mean(np.abs(imf)))
            except:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0])
    
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(len(signal), 1/fs)
    power = np.abs(fft_vals)**2
    
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 
             'beta': (13, 30), 'gamma': (30, 50), 'high_gamma': (50, 80)}
    
    band_energies = {}
    for name, (low, high) in bands.items():
        mask = (np.abs(fft_freqs) >= low) & (np.abs(fft_freqs) <= high)
        band_energies[name] = np.sum(power[mask])
    
    total = sum(band_energies.values()) + 1e-10
    features.append(band_energies['theta'] / (band_energies['beta'] + 1e-10))
    features.append(band_energies['delta'] / (band_energies['alpha'] + 1e-10))
    features.append(band_energies['gamma'] / total)
    features.append(band_energies['alpha'] / (band_energies['beta'] + 1e-10))
    features.append(band_energies['theta'] / (band_energies['alpha'] + 1e-10))
    features.append(band_energies['high_gamma'] / total)
    
    for i in range(5):
        if i < len(IMFs):
            try:
                analytic = hilbert(IMFs[i])
                phase = np.unwrap(np.angle(analytic))
                inst_freq = np.diff(phase) / (2 * np.pi) * fs
                features.append(np.var(inst_freq))
            except:
                features.append(0)
        else:
            features.append(0)
    
    for i in range(5):
        if i < len(IMFs):
            features.append(np.sum(np.diff(np.sign(IMFs[i])) != 0))
        else:
            features.append(0)
    
    try:
        features.append(kurtosis(IMFs[0]) if len(IMFs) > 0 else 0)
        features.append(skew(IMFs[0]) if len(IMFs) > 0 else 0)
        features.append(np.std(IMFs[0]) if len(IMFs) > 0 else 0)
        features.append(kurtosis(IMFs[1]) if len(IMFs) > 1 else 0)
        features.append(skew(IMFs[1]) if len(IMFs) > 1 else 0)
    except:
        features.extend([0] * 5)
    
    act, mob, comp = hjorth_parameters(signal)
    features.extend([act, mob, comp])
    
    features.append(sample_entropy_fast(signal, m=2))
    features.append(sample_entropy_fast(signal, m=3))
    
    try:
        coeffs = pywt.wavedec(signal, 'db4', level=5)
        for coeff in coeffs:
            features.append(np.sum(coeff**2))
            prob = np.abs(coeff) / (np.sum(np.abs(coeff)) + 1e-10)
            features.append(-np.sum(prob * np.log(prob + 1e-10)))
            features.append(np.std(coeff))
    except:
        features.extend([0] * 18)
    
    features.append(np.mean(np.abs(signal - np.mean(signal))))
    rms = np.sqrt(np.mean(signal**2))
    features.append(rms)
    features.append(np.ptp(signal))
    features.append(np.max(np.abs(signal)) / (rms + 1e-10))
    features.append(rms / (np.mean(np.abs(signal)) + 1e-10))
    features.append(np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-10))
    features.append(np.max(np.abs(signal)) / (np.mean(np.sqrt(np.abs(signal)))**2 + 1e-10))
    
    try:
        lags = range(2, 20)
        tau = [np.std(signal[l:] - signal[:-l]) for l in lags]
        hurst = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0] * 0.5
    except:
        hurst = 0.5
    features.append(hurst)
    
    features.append(1.0)
    
    features.append(compute_spectral_flatness(signal))
    
    pos_mask = fft_freqs >= 0
    freqs_pos = fft_freqs[pos_mask]
    power_pos = power[pos_mask]
    total_p = np.sum(power_pos) + 1e-10
    mean_f = np.sum(freqs_pos * power_pos) / total_p
    features.append(mean_f)
    bw = np.sqrt(np.sum((freqs_pos - mean_f)**2 * power_pos) / total_p)
    features.append(bw)
    
    features = [f if np.isfinite(f) else 0 for f in features]
    features = features[:80]
    while len(features) < 80:
        features.append(0)
    
    return np.array(features)


def calculate_metrics(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'sensitivity': tp / (tp + fn + 1e-10),
        'specificity': tn / (tn + fp + 1e-10),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }
    
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics['auc'] = auc(fpr, tpr)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
    
    return metrics


def main():
    try:
        X_signals = np.load('X_signals.npy')
        y_binary = np.load('y_binary.npy')
    except:
        return
    
    n_samples = 1000
    
    seizure_idx = np.where(y_binary == 1)[0][:n_samples]
    normal_idx = np.where(y_binary == 0)[0][:n_samples]
    
    X_seizure = X_signals[seizure_idx]
    X_normal = X_signals[normal_idx]
    
    X_all = np.vstack([X_seizure, X_normal])
    y_all = np.hstack([np.ones(len(X_seizure)), np.zeros(len(X_normal))])
    
    print(f"Seizure: {len(X_seizure)}, Normal: {len(X_normal)}")
    print(f"Total: {len(y_all)} samples")
    
    print("\nFFT")
    X_fft = np.array([extract_fft_features(s) for s in tqdm(X_all, desc="      FFT")])
    
    print("\nDWT")
    X_dwt = np.array([extract_dwt_features(s) for s in tqdm(X_all, desc="      DWT")])
    
    print("\nHHT Standard")
    X_hht = np.array([extract_hht_features(s) for s in tqdm(X_all, desc="      HHT")])
    
    print("\nAM-HHT (cu Spectral Flatness)")
    X_am = np.array([extract_am_hht_features(s) for s in tqdm(X_all, desc="   AM-HHT")])
    
    print(f"\nFFT: {X_fft.shape[1]} features")
    print(f"DWT: {X_dwt.shape[1]} features")
    print(f"HHT: {X_hht.shape[1]} features")
    print(f"AM-HHT: {X_am.shape[1]} features")
    
    methods = {
        'FFT': (X_fft, 'steelblue'),
        'DWT': (X_dwt, 'darkorange'),
        'HHT': (X_hht, 'forestgreen'),
        'AM-HHT': (X_am, 'crimson')
    }
    
    results = {}
    
    for name, (X_feat, color) in methods.items():
        print(f"\n  → {name}...")
        
        X_feat = np.nan_to_num(X_feat, nan=0, posinf=0, neginf=0)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_feat, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        metrics['y_test'] = y_test
        metrics['y_pred'] = y_pred
        metrics['color'] = color
        metrics['n_features'] = X_feat.shape[1]
        
        results[name] = metrics
        
        print(f"Accuracy:{metrics['accuracy']:.4f}")
        print(f"Sensitivity:{metrics['sensitivity']:.4f}")
        print(f"Specificity:{metrics['specificity']:.4f}")
        print(f"F1-Score:{metrics['f1_score']:.4f}")
        print(f"AUC:{metrics['auc']:.4f}")
    
    fig = plt.figure(figsize=(20, 12))
    
    method_names = ['FFT', 'DWT', 'HHT', 'AM-HHT']
    cmaps = ['Blues', 'Oranges', 'Greens', 'Reds']
    
    for i, name in enumerate(method_names):
        ax = plt.subplot(2, 4, i + 1)
        cm = confusion_matrix(results[name]['y_test'], results[name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[i], ax=ax,
                   xticklabels=['Normal', 'Seizure'],
                   yticklabels=['Normal', 'Seizure'],
                   annot_kws={'size': 11, 'weight': 'bold'})
        
        acc = results[name]['accuracy']
        sens = results[name]['sensitivity']
        spec = results[name]['specificity']
        nf = results[name]['n_features']
        
        ax.set_title(f'{name} ({nf} feat)\nAcc:{acc:.3f} | Sens:{sens:.3f} | Spec:{spec:.3f}',
                    fontweight='bold', fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    ax5 = plt.subplot(2, 4, 5)
    metrics_names = ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc']
    x = np.arange(len(metrics_names))
    width = 0.2
    colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
    
    for i, name in enumerate(method_names):
        vals = [results[name][m] for m in metrics_names]
        ax5.bar(x + i * width, vals, width, label=name, color=colors[i], edgecolor='black')
    
    ax5.set_ylabel('Score', fontweight='bold')
    ax5.set_title('Comparație Metrici', fontweight='bold')
    ax5.set_xticks(x + 1.5 * width)
    ax5.set_xticklabels(['Acc', 'Sens', 'Spec', 'F1', 'AUC'])
    ax5.legend(loc='lower right', fontsize=8)
    ax5.set_ylim([0.7, 1.0])
    ax5.grid(True, alpha=0.3, axis='y')
    
    ax6 = plt.subplot(2, 4, 6)
    for name in method_names:
        ax6.plot(results[name]['fpr'], results[name]['tpr'], 
                color=results[name]['color'], linewidth=2,
                label=f"{name} (AUC={results[name]['auc']:.3f})")
    ax6.plot([0, 1], [0, 1], 'k--')
    ax6.set_xlabel('False Positive Rate')
    ax6.set_ylabel('True Positive Rate')
    ax6.set_title('ROC Curves', fontweight='bold')
    ax6.legend(loc='lower right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    ax7 = plt.subplot(2, 4, 7)
    baseline = results['FFT']['accuracy']
    improvements = [(results[m]['accuracy'] - baseline) * 100 for m in ['DWT', 'HHT', 'AM-HHT']]
    bars = ax7.bar(['DWT', 'HHT', 'AM-HHT'], improvements, 
                   color=['darkorange', 'forestgreen', 'crimson'], edgecolor='black')
    ax7.axhline(0, color='black', linewidth=1)
    ax7.set_ylabel('Improvement vs FFT (%)', fontweight='bold')
    ax7.set_title('Îmbunătățire față de FFT', fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    for bar, imp in zip(bars, improvements):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'+{imp:.1f}%', ha='center', fontweight='bold')
    
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    table_data = [['Metric', 'FFT', 'DWT', 'HHT', 'AM-HHT', 'BEST']]
    for m in ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc']:
        row = [m.upper()]
        vals = [results[n][m] for n in method_names]
        for v in vals:
            row.append(f'{v:.4f}')
        row.append(method_names[np.argmax(vals)])
        table_data.append(row)
    table_data.append(['Features', '15', '15', '15', '80', '-'])
    
    table = ax8.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.18, 0.14, 0.14, 0.14, 0.14, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    for i in range(6):
        table[(0, i)].set_facecolor('#1976D2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax8.set_title('Summary', fontweight='bold', pad=10)
    
    plt.suptitle('COMPARAȚIE COMPLETĂ: FFT vs DWT vs HHT Standard vs AM-HHT\n'
                '(AM-HHT include Spectral Flatness IMF Selection - CONTRIBUȚIE ORIGINALĂ)',
                fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('10_FINAL_comparison_all_methods.png', dpi=300, bbox_inches='tight')
    plt.close()
    
      
    print(f"\n{'Metric':<15} {'FFT':>10} {'DWT':>10} {'HHT':>10} {'AM-HHT':>10} {'BEST':>10}")
 
    
    for m in ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc']:
        vals = [results[n][m] for n in method_names]
        best_idx = np.argmax(vals)
        
        row = f"{m.upper():<15}"
        for i, v in enumerate(vals):
            if i == best_idx:
                row += f" {v:>9.4f}*"
            else:
                row += f" {v:>10.4f}"
        row += f" {method_names[best_idx]:>10}"
        print(row)
    
    
    print(f"{'Features':<15} {'15':>10} {'15':>10} {'15':>10} {'80':>10}")
    
    
    # Winner
    winner = max(method_names, key=lambda n: results[n]['accuracy'])
    
    print(f"imbunatatire vs FFT: +{(results['AM-HHT']['accuracy'] - results['FFT']['accuracy'])*100:.2f}%")
    print(f"imbunatatire vs HHT: +{(results['AM-HHT']['accuracy'] - results['HHT']['accuracy'])*100:.2f}%")
    
    
    print("10_FINAL_comparison_all_methods.png")
    


if __name__ == '__main__':
    main()