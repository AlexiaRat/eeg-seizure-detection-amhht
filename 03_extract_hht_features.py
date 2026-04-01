import numpy as np
from PyEMD.EMD import EMD
from tqdm import tqdm

def extract_hht_features(signal, n_imfs=5):
    emd = EMD()
    
    try:
        IMFs = emd.emd(signal)
        if IMFs.ndim == 1:
            IMFs = [IMFs]
        else:
            IMFs = [IMFs[i] for i in range(IMFs.shape[0])]
    except:
        return np.zeros(n_imfs * 3)
    
    features = []
    
    for i in range(min(n_imfs, len(IMFs))):
        imf = IMFs[i]
        
        energy = np.sum(imf**2)
        features.append(energy)
        
        prob = np.abs(imf) / (np.sum(np.abs(imf)) + 1e-10)
        entr = -np.sum(prob * np.log(prob + 1e-10))
        features.append(entr)
        
        mean_amp = np.mean(np.abs(imf))
        features.append(mean_amp)
    
    while len(features) < n_imfs * 3:
        features.append(0)
    
    return np.array(features[:n_imfs * 3])

def main():
    X = np.load('X_signals.npy')
    y_binary = np.load('y_binary.npy')

    n_samples = 1000

    X_seizure = X[y_binary == 1][:n_samples]
    X_normal = X[y_binary == 0][:n_samples]

    features_seizure = []
    for sig in tqdm(X_seizure, desc="Seizure"):
        features = extract_hht_features(sig)
        features_seizure.append(features)
    features_seizure = np.array(features_seizure)

    features_normal = []
    for sig in tqdm(X_normal, desc="Normal"):
        features = extract_hht_features(sig)
        features_normal.append(features)
    features_normal = np.array(features_normal)

    X_features = np.vstack([features_seizure, features_normal])
    y_labels = np.hstack([np.ones(n_samples), np.zeros(n_samples)])

    np.save('X_hht_features.npy', X_features)
    np.save('y_hht_labels.npy', y_labels)

if __name__ == '__main__':
    main()