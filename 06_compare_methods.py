
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def train_and_evaluate(X, y, method_name):
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    if len(X) == 0 or len(y) == 0:
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_test': y_test,
        'y_pred': y_pred,
        'model': rf,
        'features': X.shape[1]
    }


def main():
    try:
        X_hht = np.load('X_hht_features.npy')
        y_hht = np.load('y_hht_labels.npy')
        
    except:
        return
    
    try:
        X_adaptive = np.load('X_adaptive_features.npy')
        y_adaptive = np.load('y_adaptive_labels.npy')
    except:
        return
    
    if X_adaptive.shape[0] == 0:
        return
    
    results_hht = train_and_evaluate(X_hht, y_hht, "HHT Standard")
    results_adaptive = train_and_evaluate(X_adaptive, y_adaptive, "AM-HHT")
    
    if results_hht is None or results_adaptive is None:
        return
    
    
    print(f"HHT Standard: {results_hht['accuracy']:.4f}")
    print(f"AM-HHT:       {results_adaptive['accuracy']:.4f}")
    improvement = (results_adaptive['accuracy'] - results_hht['accuracy']) * 100
    print(f"Improvement:  {improvement:+.2f}%")
    
    fig = plt.figure(figsize=(16, 10))
    
    ax1 = plt.subplot(2, 3, 1)
    cm_hht = confusion_matrix(results_hht['y_test'], results_hht['y_pred'])
    sns.heatmap(cm_hht, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Normal', 'Seizure'], yticklabels=['Normal', 'Seizure'])
    ax1.set_title(f'HHT Standard\nAccuracy: {results_hht["accuracy"]:.3f}', fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    ax2 = plt.subplot(2, 3, 2)
    cm_am = confusion_matrix(results_adaptive['y_test'], results_adaptive['y_pred'])
    sns.heatmap(cm_am, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=['Normal', 'Seizure'], yticklabels=['Normal', 'Seizure'])
    ax2.set_title(f'AM-HHT (Spectral Flatness)\nAccuracy: {results_adaptive["accuracy"]:.3f}', fontweight='bold')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    ax3 = plt.subplot(2, 3, 3)
    methods = ['HHT\nStandard', 'AM-HHT\n(Original)']
    accuracies = [results_hht['accuracy'], results_adaptive['accuracy']]
    colors = ['steelblue', 'forestgreen']
    bars = ax3.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Accuracy', fontweight='bold')
    ax3.set_ylim([0.8, 1.0])
    ax3.set_title('Accuracy Comparison', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accuracies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', fontweight='bold')
    
    ax4 = plt.subplot(2, 3, 4)
    feature_names_hht = []
    for i in range(1, 6):
        feature_names_hht.extend([f'IMF{i}_E', f'IMF{i}_Ent', f'IMF{i}_MA'])
    
    importances = results_hht['model'].feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    
    ax4.barh(range(10), importances[top_idx], color='steelblue', edgecolor='black')
    ax4.set_yticks(range(10))
    ax4.set_yticklabels([feature_names_hht[i] if i < len(feature_names_hht) else f'F{i}' for i in top_idx])
    ax4.set_xlabel('Importance')
    ax4.set_title('HHT - Top 10 Features', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    ax5 = plt.subplot(2, 3, 5)
    
    feature_names_am = []
    for i in range(1, 9):
        feature_names_am.extend([f'IMF{i}_E', f'IMF{i}_Ent', f'IMF{i}_MA'])
    feature_names_am.extend(['θ/β', 'δ/α', 'γ%', 'α/β', 'θ/α', 'hγ%'])
    feature_names_am.extend([f'FV{i}' for i in range(1, 6)])
    feature_names_am.extend([f'ZCR{i}' for i in range(1, 6)])
    feature_names_am.extend(['Kurt1', 'Skew1', 'Std1', 'Kurt2', 'Skew2'])
    feature_names_am.extend(['H_Act', 'H_Mob', 'H_Comp'])
    feature_names_am.extend(['SampEn2', 'SampEn3'])
    for i in range(1, 7):
        feature_names_am.extend([f'W{i}_E', f'W{i}_Ent', f'W{i}_Std'])
    feature_names_am.extend(['MAD', 'RMS', 'PtP', 'Crest', 'Shape', 'Impulse', 'Clear'])
    feature_names_am.extend(['Hurst', 'DFA', 'SF', 'MeanF', 'BW'])
    
    importances_am = results_adaptive['model'].feature_importances_
    top_idx_am = np.argsort(importances_am)[-10:][::-1]
    
    ax5.barh(range(10), importances_am[top_idx_am], color='forestgreen', edgecolor='black')
    ax5.set_yticks(range(10))
    ax5.set_yticklabels([feature_names_am[i] if i < len(feature_names_am) else f'F{i}' for i in top_idx_am])
    ax5.set_xlabel('Importance')
    ax5.set_title('AM-HHT - Top 10 Features', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary = [
        ['Metric', 'HHT', 'AM-HHT', 'Δ'],
        ['Accuracy', f"{results_hht['accuracy']:.4f}", f"{results_adaptive['accuracy']:.4f}",
         f"{improvement:+.2f}%"],
        ['CV Score', f"{results_hht['cv_mean']:.4f}", f"{results_adaptive['cv_mean']:.4f}",
         f"{(results_adaptive['cv_mean']-results_hht['cv_mean'])*100:+.2f}%"],
        ['Features', f"{results_hht['features']}", f"{results_adaptive['features']}",
         f"+{results_adaptive['features']-results_hht['features']}"],
    ]
    
    table = ax6.table(cellText=summary, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Performance Summary', fontweight='bold', pad=15)
    
    plt.suptitle('HHT Standard vs AM-HHT (cu Spectral Flatness - CONTRIBUȚIE ORIGINALĂ)',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('08_comparison_hht_vs_adaptive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n08_comparison_hht_vs_adaptive.png")


if __name__ == '__main__':
    main()