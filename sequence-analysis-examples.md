# Sequence Analysis Examples and Statistical Analysis Guide

## 1. Example Outputs with Interpretations

### A. GC Content Analysis
```python
def analyze_gc_distribution(sequence, window_size=100):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    # Calculate GC content in sliding windows
    gc_content = []
    positions = []
    for i in range(0, len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        gc = (window.count('G') + window.count('C')) / window_size
        gc_content.append(gc)
        positions.append(i)
    
    # Statistical analysis
    gc_mean = np.mean(gc_content)
    gc_std = np.std(gc_content)
    gc_zscore = stats.zscore(gc_content)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # GC content plot
    ax1.plot(positions, gc_content, 'b-', label='GC Content')
    ax1.axhline(y=gc_mean, color='r', linestyle='--', label=f'Mean ({gc_mean:.2f})')
    ax1.fill_between(positions, gc_mean - gc_std, gc_mean + gc_std, 
                     alpha=0.2, color='gray', label=f'±1 SD ({gc_std:.2f})')
    ax1.set_title('GC Content Distribution')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('GC Content')
    ax1.legend()
    
    # Z-score plot
    ax2.plot(positions, gc_zscore, 'g-', label='Z-score')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title('GC Content Z-scores')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Z-score')
    
    plt.tight_layout()
    return {'mean': gc_mean, 'std': gc_std, 'zscore': gc_zscore}
```

**Example Interpretation:**
- Normal GC content: 40-60% range
- Z-score peaks (>2): Potential regulatory regions
- Sudden changes: Possible domain boundaries
- Consistent patterns: Well-maintained sequence properties

### B. K-mer Analysis
```python
def analyze_kmers(sequence, k=3, top_n=10):
    from collections import Counter
    import pandas as pd
    import seaborn as sns
    
    # Generate all k-mers
    kmers = [sequence[i:i+k] for i in range(len(sequence)-k+1)]
    kmer_counts = Counter(kmers)
    
    # Calculate expected frequencies
    total_kmers = len(kmers)
    expected_freq = 1/(4**k)  # For DNA sequences
    
    # Chi-square analysis
    from scipy.stats import chisquare
    observed_freqs = [count/total_kmers for count in kmer_counts.values()]
    expected_freqs = [expected_freq] * len(kmer_counts)
    chi2_stat, p_value = chisquare(observed_freqs, expected_freqs)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    top_kmers = dict(sorted(kmer_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    sns.barplot(x=list(top_kmers.keys()), y=list(top_kmers.values()))
    plt.title(f'Top {top_n} {k}-mers (χ² p-value: {p_value:.2e})')
    plt.xticks(rotation=45)
    plt.ylabel('Frequency')
    
    return {'counts': kmer_counts, 'chi2_stat': chi2_stat, 'p_value': p_value}
```

**Example Interpretation:**
- Over-represented k-mers: Potential binding sites
- Under-represented k-mers: Possible selective constraints
- Chi-square p-value: Sequence randomness measure
- Pattern distribution: Functional element indicators

## 2. Advanced Statistical Analyses

### A. Sequence Complexity Analysis
```python
def calculate_sequence_complexity(sequence, window_size=50):
    import numpy as np
    from collections import Counter
    
    def entropy(seq):
        counts = Counter(seq)
        frequencies = [count/len(seq) for count in counts.values()]
        return -sum(f * np.log2(f) for f in frequencies)
    
    def linguistic_complexity(seq):
        observed_subseqs = set()
        max_possible = 0
        for k in range(1, len(seq) + 1):
            max_possible += min(4**k, len(seq) - k + 1)
            for i in range(len(seq) - k + 1):
                observed_subseqs.add(seq[i:i+k])
        return len(observed_subseqs) / max_possible
    
    # Calculate metrics in sliding windows
    entropies = []
    complexities = []
    positions = []
    
    for i in range(0, len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        entropies.append(entropy(window))
        complexities.append(linguistic_complexity(window))
        positions.append(i)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(positions, entropies, 'b-')
    ax1.set_title('Sequence Entropy')
    ax1.set_ylabel('Entropy (bits)')
    
    ax2.plot(positions, complexities, 'g-')
    ax2.set_title('Linguistic Complexity')
    ax2.set_ylabel('Complexity Score')
    ax2.set_xlabel('Position')
    
    plt.tight_layout()
    
    return {
        'entropy': np.mean(entropies),
        'complexity': np.mean(complexities),
        'entropy_std': np.std(entropies),
        'complexity_std': np.std(complexities)
    }
```

### B. Comparative Sequence Analysis
```python
def compare_sequences(sequences, names=None):
    import numpy as np
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist
    
    if names is None:
        names = [f'Seq_{i+1}' for i in range(len(sequences))]
    
    def sequence_to_vector(seq):
        """Convert sequence to frequency vector"""
        bases = ['A', 'T', 'C', 'G']
        return [seq.count(base)/len(seq) for base in bases]
    
    # Calculate feature vectors
    vectors = np.array([sequence_to_vector(seq) for seq in sequences])
    
    # Calculate distance matrix
    distances = pdist(vectors, metric='euclidean')
    
    # Hierarchical clustering
    plt.figure(figsize=(10, 8))
    hierarchy.dendrogram(hierarchy.linkage(distances),
                        labels=names,
                        leaf_rotation=90)
    plt.title('Sequence Similarity Dendrogram')
    
    # Pairwise comparison matrix
    plt.figure(figsize=(10, 8))
    similarity_matrix = 1 - pdist(vectors, metric='euclidean').reshape(len(sequences), -1)
    sns.heatmap(similarity_matrix,
                xticklabels=names,
                yticklabels=names,
                annot=True,
                cmap='YlOrRd')
    plt.title('Sequence Similarity Matrix')
    
    return {'distances': distances, 'similarity_matrix': similarity_matrix}
```

### C. Statistical Significance Testing
```python
def sequence_statistical_analysis(generated_sequences, reference_sequences):
    """Comprehensive statistical analysis of generated vs reference sequences"""
    import scipy.stats as stats
    
    def calculate_metrics(seq):
        return {
            'gc_content': (seq.count('G') + seq.count('C')) / len(seq),
            'length': len(seq),
            'complexity': calculate_sequence_complexity(seq)['complexity'],
            'entropy': calculate_sequence_complexity(seq)['entropy']
        }
    
    # Calculate metrics for both sets
    gen_metrics = [calculate_metrics(seq) for seq in generated_sequences]
    ref_metrics = [calculate_metrics(seq) for seq in reference_sequences]
    
    # Statistical tests
    results = {}
    for metric in ['gc_content', 'length', 'complexity', 'entropy']:
        gen_values = [m[metric] for m in gen_metrics]
        ref_values = [m[metric] for m in ref_metrics]
        
        # T-test
        t_stat, p_value = stats.ttest_ind(gen_values, ref_values)
        
        # KS test for distribution comparison
        ks_stat, ks_p = stats.ks_2samp(gen_values, ref_values)
        
        results[metric] = {
            't_statistic': t_stat,
            't_p_value': p_value,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p
        }
    
    return results
```

## 3. Interpretation Guidelines

### Statistical Metrics Interpretation:

1. GC Content Analysis:
- Mean within 40-60%: Biologically plausible
- Standard deviation < 0.1: Stable composition
- Z-scores > 2: Significant local variations

2. K-mer Analysis:
- P-value < 0.05: Non-random distribution
- Over-represented motifs: Potential functional elements
- Under-represented motifs: Possible constraints

3. Sequence Complexity:
- Entropy > 1.5: Good sequence diversity
- Complexity > 0.7: Well-structured sequence
- Low variation: Stable sequence properties

4. Comparative Analysis:
- Distance < 0.1: Highly similar sequences
- Clustering patterns: Sequence families
- Matrix patterns: Sequence relationships

### Warning Signs:

1. GC Content:
- Extreme values (< 30% or > 70%)
- Rapid fluctuations
- Monotonic patterns

2. K-mer Distribution:
- Highly skewed distributions
- Missing expected motifs
- Over-abundance of simple repeats

3. Complexity Metrics:
- Very low entropy (< 1.0)
- Low complexity regions
- Sudden complexity drops

4. Statistical Tests:
- P-values << 0.05: Significant deviations
- Large effect sizes: Potential artifacts
- Inconsistent patterns: Generation issues

