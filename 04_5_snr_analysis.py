import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_snr_db(signal, noise):
    
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return np.inf
    
    snr = signal_power / noise_power
    snr_db = 10 * np.log10(snr)
    
    return snr_db

def calculate_signal_noise_from_features(X_features):
    snr_values = []
    
    for features in X_features:
        # energy features
        imf1_energy = features[0]  
        imf2_energy = features[3]  
        imf4_energy = features[9]  
        imf5_energy = features[12] 
        
        signal_power = imf1_energy + imf2_energy
        noise_power = imf4_energy + imf5_energy + 1e-10
        
        snr = signal_power / noise_power
        snr_db = 10 * np.log10(snr + 1e-10)
        snr_values.append(snr_db)
    
    return np.array(snr_values)

def main():
    X_features = np.load('X_hht_features.npy')
    y_labels = np.load('y_hht_labels.npy')
    
    # calculeaza SNR pentru fiecare semnal
    print("\n📊 Calculare SNR...")
    snr_values = calculate_signal_noise_from_features(X_features)
    
    # separa SNR pentru seizure si normal
    snr_seizure = snr_values[y_labels == 1]
    snr_normal = snr_values[y_labels == 0]
    
    
    print(f"Seizure signals:")
    print(f"Mean SNR: {np.mean(snr_seizure):.2f} dB")
    print(f"Std SNR:  {np.std(snr_seizure):.2f} dB")
    print(f"Min SNR:  {np.min(snr_seizure):.2f} dB")
    print(f"Max SNR:  {np.max(snr_seizure):.2f} dB")
    
    print(f"\nNormal signals:")
    print(f"Mean SNR: {np.mean(snr_normal):.2f} dB")
    print(f"Std SNR:  {np.std(snr_normal):.2f} dB")
    print(f"Min SNR:  {np.min(snr_normal):.2f} dB")
    print(f"Max SNR:  {np.max(snr_normal):.2f} dB")
    
    print(f"\nMean SNR: {np.mean(snr_seizure) - np.mean(snr_normal):.2f} dB")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
  
    ax = axes[0, 0]
    ax.hist(snr_normal, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Normal')
    ax.hist(snr_seizure, bins=30, alpha=0.7, color='red', edgecolor='black', label='Seizure')
    ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('SNR Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    box_data = [snr_normal, snr_seizure]
    bp = ax.boxplot(box_data, labels=['Normal', 'Seizure'], patch_artist=True,
                    widths=0.6)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    ax.set_ylabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title('SNR Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 0]
    indices_normal = np.where(y_labels == 0)[0]
    indices_seizure = np.where(y_labels == 1)[0]
    
    ax.scatter(indices_normal, snr_normal, c='blue', alpha=0.6, s=20, label='Normal')
    ax.scatter(indices_seizure, snr_seizure, c='red', alpha=0.6, s=20, label='Seizure')
    ax.axhline(np.mean(snr_normal), color='blue', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean Normal: {np.mean(snr_normal):.1f} dB')
    ax.axhline(np.mean(snr_seizure), color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean Seizure: {np.mean(snr_seizure):.1f} dB')
    ax.set_xlabel('Signal Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title('SNR per Signal', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'Normal', 'Seizure', 'Δ'],
        ['Mean SNR (dB)', f'{np.mean(snr_normal):.2f}', f'{np.mean(snr_seizure):.2f}', f'{np.mean(snr_seizure)-np.mean(snr_normal):.2f}'],
        ['Std SNR (dB)', f'{np.std(snr_normal):.2f}', f'{np.std(snr_seizure):.2f}', '—'],
        ['Min SNR (dB)', f'{np.min(snr_normal):.2f}', f'{np.min(snr_seizure):.2f}', '—'],
        ['Max SNR (dB)', f'{np.max(snr_normal):.2f}', f'{np.max(snr_seizure):.2f}', '—'],
        ['Median SNR (dB)', f'{np.median(snr_normal):.2f}', f'{np.median(snr_seizure):.2f}', '—'],
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.23, 0.23, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('SNR Statistics Summary', fontweight='bold', fontsize=14, pad=20)
    
    plt.suptitle('Signal-to-Noise Ratio (SNR) Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('04.5_snr_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n04.5_snr_analysis.png")

    np.save('snr_values.npy', snr_values)
    print("snr_values.npy")


if __name__ == '__main__':
    main()