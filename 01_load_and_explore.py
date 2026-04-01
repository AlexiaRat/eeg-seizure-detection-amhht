import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
df = df.drop('Unnamed', axis=1)

X = df.drop('y', axis=1).values
y = df['y'].values

y_binary = (y == 5).astype(int)


print(f"Date încarcate: {X.shape}")
print(f"Healthy (0): {np.sum(y_binary == 0)} samples")
print(f"Seizure (1): {np.sum(y_binary == 1)} samples")
print(f"Balance ratio: 1:{np.sum(y_binary == 0) / np.sum(y_binary == 1):.1f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

unique_classes = np.unique(y)
axes[0].bar(unique_classes, [np.sum(y == cls) for cls in unique_classes], 
            color=['blue', 'red'])
axes[0].set_xlabel('Class', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Original Classes (1 + 5)', fontweight='bold')
axes[0].set_xticks([1, 5])
axes[0].set_xticklabels(['1\n(Healthy)', '5\n(Seizure)'])
axes[0].grid(True, alpha=0.3)

axes[1].bar(['Healthy\n(class 1)', 'Seizure\n(class 5)'], 
            [np.sum(y_binary == 0), np.sum(y_binary == 1)],
            color=['blue', 'red'])
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Binary Classification', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 3 semnale seizure
for i in range(3):
    seizure_idx = np.where(y_binary == 1)[0][i]
    signal = X[seizure_idx]
    
    axes[0, i].plot(signal, 'r', linewidth=0.8)
    axes[0, i].set_title(f'Seizure Sample {i+1}', 
                        fontweight='bold', color='red')
    axes[0, i].set_xlabel('Sample Point')
    axes[0, i].set_ylabel('Amplitude')
    axes[0, i].grid(True, alpha=0.3)

# 3 semnale healthy
for i in range(3):
    healthy_idx = np.where(y_binary == 0)[0][i]
    signal = X[healthy_idx]
    
    axes[1, i].plot(signal, 'b', linewidth=0.8)
    axes[1, i].set_title(f'Healthy Sample {i+1}', 
                        fontweight='bold', color='blue')
    axes[1, i].set_xlabel('Sample Point')
    axes[1, i].set_ylabel('Amplitude')
    axes[1, i].grid(True, alpha=0.3)

plt.suptitle('Raw EEG Signals: Seizure (Class 5) vs Healthy (Class 1)', 
            fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('02_raw_signals.png', dpi=300, bbox_inches='tight')
plt.close()

# distributii amplitudine
seizure_signals = X[y_binary == 1]
healthy_signals = X[y_binary == 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(seizure_signals.flatten(), bins=50, color='red', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Amplitude')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Seizure (Class 5) Amplitude Distribution', fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].hist(healthy_signals.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Amplitude')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Healthy (Class 1) Amplitude Distribution', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_amplitude_distribution.png', dpi=300, bbox_inches='tight')
plt.close()


np.save('X_signals.npy', X)
np.save('y_binary.npy', y_binary)

print("X_signals.npy, y_binary.npy")
