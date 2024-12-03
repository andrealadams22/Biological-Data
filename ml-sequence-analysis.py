import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
from Bio import SeqIO
import datetime
import jinja2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class SequenceFeatureExtractor:
    """Extract features from DNA sequences for ML analysis"""
    
    def __init__(self, kmer_sizes=[2, 3, 4]):
        self.kmer_sizes = kmer_sizes
        self.feature_names = []
        self._initialize_features()
    
    def _initialize_features(self):
        """Initialize feature names"""
        bases = ['A', 'T', 'C', 'G']
        
        # Add basic composition features
        self.feature_names.extend(['gc_content', 'at_content'])
        
        # Add k-mer features
        for k in self.kmer_sizes:
            kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
            self.feature_names.extend(kmers)
    
    def transform(self, sequences):
        """Transform sequences into feature matrix"""
        features = []
        for seq in sequences:
            seq_features = []
            
            # Basic composition
            gc_content = (seq.count('G') + seq.count('C')) / len(seq)
            at_content = (seq.count('A') + seq.count('T')) / len(seq)
            seq_features.extend([gc_content, at_content])
            
            # K-mer frequencies
            for k in self.kmer_sizes:
                kmer_counts = self._get_kmer_frequencies(seq, k)
                seq_features.extend(kmer_counts.values())
            
            features.append(seq_features)
        
        return np.array(features)
    
    def _get_kmer_frequencies(self, sequence, k):
        """Calculate k-mer frequencies"""
        kmers = {}
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            kmers[kmer] = kmers.get(kmer, 0) + 1
        return kmers

class SequenceAnalyzer:
    """ML-based sequence analysis"""
    
    def __init__(self):
        self.feature_extractor = SequenceFeatureExtractor()
        self.rf_model = RandomForestClassifier(n_estimators=100)
        self.nn_model = self._build_nn_model()
    
    def _build_nn_model(self):
        """Build neural network model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def train(self, real_sequences, generated_sequences):
        """Train models on real vs generated sequences"""
        # Prepare data
        X_real = self.feature_extractor.transform(real_sequences)
        X_gen = self.feature_extractor.transform(generated_sequences)
        
        X = np.vstack([X_real, X_gen])
        y = np.array([1] * len(real_sequences) + [0] * len(generated_sequences))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train random forest
        self.rf_model.fit(X_train, y_train)
        rf_score = self.rf_model.score(X_test, y_test)
        
        # Train neural network
        self.nn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
        nn_score = self.nn_model.evaluate(X_test, y_test)[1]
        
        return {
            'rf_score': rf_score,
            'nn_score': nn_score,
            'feature_importance': self._get_feature_importance()
        }
    
    def _get_feature_importance(self):
        """Get feature importance from random forest"""
        importance = self.rf_model.feature_importances_
        features = self.feature_extractor.feature_names
        return dict(zip(features, importance))

class ReportGenerator:
    """Generate analysis reports"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_env = self._setup_jinja()
    
    def _setup_jinja(self):
        """Setup Jinja2 environment"""
        template_loader = jinja2.FileSystemLoader(searchpath="./templates")
        template_env = jinja2.Environment(loader=template_loader)
        return template_env
    
    def generate_report(self, analysis_results, sequences, filename="analysis_report.html"):
        """Generate HTML report"""
        template = self.template_env.get_template("report_template.html")
        
        # Generate plots
        plot_paths = self._generate_plots(analysis_results, sequences)
        
        # Prepare report context
        context = {
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'analysis_results': analysis_results,
            'plot_paths': plot_paths,
            'sequence_stats': self._calculate_sequence_stats(sequences)
        }
        
        # Render report
        html_output = template.render(context)
        
        # Save report
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            f.write(html_output)
        
        return report_path
    
    def _generate_plots(self, analysis_results, sequences):
        """Generate plots for report"""
        plot_paths = {}
        
        # Feature importance plot
        plt.figure(figsize=(12, 6))
        importance = pd.Series(analysis_results['feature_importance']).sort_values(ascending=False)
        importance[:20].plot(kind='bar')
        plt.title('Top 20 Important Features')
        plt.xticks(rotation=45)
        path = self.output_dir / 'feature_importance.png'
        plt.savefig(path, bbox_inches='tight')
        plot_paths['feature_importance'] = path
        
        # Add more plots as needed
        
        return plot_paths
    
    def _calculate_sequence_stats(self, sequences):
        """Calculate basic sequence statistics"""
        lengths = [len(seq) for seq in sequences]
        return {
            'count': len(sequences),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths)
        }

def main():
    """Example usage of the analysis pipeline"""
    # Load sequences
    real_sequences = ["ATCG..."] # Your real sequences
    generated_sequences = ["ATCG..."] # Your generated sequences
    
    # Initialize components
    analyzer = SequenceAnalyzer()
    reporter = ReportGenerator("output_reports")
    
    # Perform analysis
    analysis_results = analyzer.train(real_sequences, generated_sequences)
    
    # Generate report
    report_path = reporter.generate_report(analysis_results, generated_sequences)
    
    print(f"Analysis complete. Report generated at: {report_path}")

if __name__ == "__main__":
    main()
