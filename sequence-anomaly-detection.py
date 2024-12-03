import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import tensorflow.keras as keras
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SequenceAnomalyDetector:
    """Comprehensive anomaly detection for biological sequences"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.lof = LocalOutlierFactor(novelty=True, contamination=0.1)
        self.autoencoder = self._build_autoencoder()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% variance
        
    def _build_autoencoder(self):
        """Build autoencoder for sequence anomaly detection"""
        encoding_dim = 32
        
        encoder = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(encoding_dim, activation='relu')
        ])
        
        decoder = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(256, activation='sigmoid')
        ])
        
        autoencoder = keras.Sequential([encoder, decoder])
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def extract_sequence_features(self, sequence):
        """Extract comprehensive feature set from sequence"""
        features = {}
        
        # Basic composition
        features['gc_content'] = (sequence.count('G') + sequence.count('C')) / len(sequence)
        features['at_content'] = (sequence.count('A') + sequence.count('T')) / len(sequence)
        
        # Dinucleotide frequencies
        dinucleotides = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 
                        'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']
        for di in dinucleotides:
            features[f'di_{di}'] = sequence.count(di) / (len(sequence) - 1)
        
        # Sequence complexity measures
        features['entropy'] = self._calculate_entropy(sequence)
        features['complexity'] = self._calculate_complexity(sequence)
        
        return features
    
    def _calculate_entropy(self, sequence):
        """Calculate Shannon entropy of sequence"""
        freq = [sequence.count(base) / len(sequence) for base in 'ATCG']
        return -sum(f * np.log2(f) if f > 0 else 0 for f in freq)
    
    def _calculate_complexity(self, sequence):
        """Calculate linguistic complexity of sequence"""
        observed_kmers = set()
        max_possible = 0
        
        for k in range(1, min(len(sequence), 5) + 1):
            max_possible += min(4**k, len(sequence) - k + 1)
            for i in range(len(sequence) - k + 1):
                observed_kmers.add(sequence[i:i+k])
                
        return len(observed_kmers) / max_possible
    
    def fit(self, sequences):
        """Train anomaly detection models"""
        # Extract features from all sequences
        feature_vectors = []
        for seq in sequences:
            features = self.extract_sequence_features(seq)
            feature_vectors.append(list(features.values()))
        
        # Convert to numpy array
        X = np.array(feature_vectors)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Reduce dimensionality
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Train models
        self.isolation_forest.fit(X_pca)
        self.lof.fit(X_pca)
        self.autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=0)
        
        # Calculate baseline reconstruction error
        self.reconstruction_threshold = self._calculate_reconstruction_threshold(X_scaled)
        
        return self
    
    def _calculate_reconstruction_threshold(self, X_scaled):
        """Calculate threshold for autoencoder reconstruction error"""
        reconstructed = self.autoencoder.predict(X_scaled)
        errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
        return np.percentile(errors, 95)  # 95th percentile
    
    def detect_anomalies(self, sequences):
        """Detect anomalies in sequences"""
        # Extract features
        feature_vectors = []
        for seq in sequences:
            features = self.extract_sequence_features(seq)
            feature_vectors.append(list(features.values()))
        
        X = np.array(feature_vectors)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        # Get predictions from each model
        if_predictions = self.isolation_forest.predict(X_pca)
        lof_predictions = self.lof.predict(X_pca)
        
        # Get autoencoder reconstruction error
        reconstructed = self.autoencoder.predict(X_scaled)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
        ae_predictions = reconstruction_errors > self.reconstruction_threshold
        
        # Combine predictions
        results = []
        for i in range(len(sequences)):
            anomaly_scores = {
                'sequence': sequences[i],
                'isolation_forest': if_predictions[i] == -1,
                'lof': lof_predictions[i] == -1,
                'autoencoder': ae_predictions[i],
                'reconstruction_error': reconstruction_errors[i],
                'features': self.extract_sequence_features(sequences[i])
            }
            
            # Calculate consensus
            anomaly_votes = sum([
                anomaly_scores['isolation_forest'],
                anomaly_scores['lof'],
                anomaly_scores['autoencoder']
            ])
            anomaly_scores['is_anomaly'] = anomaly_votes >= 2
            
            # Add specific anomaly types
            anomaly_scores['anomaly_types'] = self._identify_anomaly_types(
                anomaly_scores['features']
            )
            
            results.append(anomaly_scores)
        
        return results
    
    def _identify_anomaly_types(self, features):
        """Identify specific types of anomalies"""
        anomaly_types = []
        
        # GC content anomalies
        if features['gc_content'] > 0.7 or features['gc_content'] < 0.3:
            anomaly_types.append('Extreme GC content')
        
        # Complexity anomalies
        if features['complexity'] < 0.6:
            anomaly_types.append('Low sequence complexity')
        
        # Entropy anomalies
        if features['entropy'] < 1.5:
            anomaly_types.append('Low sequence entropy')
        
        # Dinucleotide anomalies
        dinuc_values = [v for k, v in features.items() if k.startswith('di_')]
        if any(v > 0.4 for v in dinuc_values):
            anomaly_types.append('Unusual dinucleotide frequency')
        
        return anomaly_types

    def visualize_anomalies(self, anomaly_results):
        """Visualize anomaly detection results"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Extract reconstruction errors
        errors = [result['reconstruction_error'] for result in anomaly_results]
        is_anomaly = [result['is_anomaly'] for result in anomaly_results]
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Reconstruction error distribution
        plt.subplot(1, 2, 1)
        sns.histplot(errors, bins=30)
        plt.axvline(self.reconstruction_threshold, color='r', linestyle='--')
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Reconstruction Error')
        
        # Plot 2: Anomaly types
        plt.subplot(1, 2, 2)
        anomaly_types = []
        for result in anomaly_results:
            if result['is_anomaly']:
                anomaly_types.extend(result['anomaly_types'])
        
        type_counts = pd.Series(anomaly_types).value_counts()
        sns.barplot(x=type_counts.values, y=type_counts.index)
        plt.title('Anomaly Types Distribution')
        
        plt.tight_layout()
        return plt.gcf()

def main():
    """Example usage of anomaly detection"""
    # Sample sequences
    normal_sequences = [
        "ATCGATCGATCG",
        "GCTAGCTAGCTA",
        # ... more normal sequences
    ]
    
    test_sequences = [
        "ATCGATCGATCG",  # Normal
        "AAAAAAAAAAAAA",  # Low complexity
        "GCGCGCGCGCGC",  # Extreme GC
        # ... more test sequences
    ]
    
    # Initialize and train detector
    detector = SequenceAnomalyDetector()
    detector.fit(normal_sequences)
    
    # Detect anomalies
    results = detector.detect_anomalies(test_sequences)
    
    # Print results
    for result in results:
        print(f"\nSequence: {result['sequence']}")
        print(f"Is anomaly: {result['is_anomaly']}")
        if result['is_anomaly']:
            print(f"Anomaly types: {result['anomaly_types']}")
            print(f"Reconstruction error: {result['reconstruction_error']:.4f}")

if __name__ == "__main__":
    main()
