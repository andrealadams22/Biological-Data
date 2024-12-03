# Sequence Analysis Visualization Guidelines

## 1. Base Composition Plots

### GC Content Distribution
```python
import matplotlib.pyplot as plt
import numpy as np

def plot_gc_content(sequence, window_size=100):
    # Calculate GC content in sliding windows
    gc_content = []
    for i in range(0, len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        gc = (window.count('G') + window.count('C')) / window_size
        gc_content.append(gc)
    
    plt.figure(figsize=(12, 6))
    plt.plot(gc_content)
    plt.axhline(y=0.5, color='r', linestyle='--', label='50% GC')
    plt.title(f'GC Content Distribution (Window Size: {window_size}bp)')
    plt.xlabel('Sequence Position')
    plt.ylabel('GC Content')
    plt.legend()
```

Expected Patterns:
- Normal range: 35-65% GC content
- Smooth transitions between regions
- Potential CpG islands: >60% GC
- Promoter regions often show distinct patterns

## 2. Sequence Logo Visualization

### Motif Analysis
```python
from weblogo import *

def create_sequence_logo(sequences, output_file):
    # Create sequence logo for aligned sequences
    sequences = [Seq(seq) for seq in sequences]
    data = LogoData.from_seqs(sequences)
    options = LogoOptions()
    options.title = "Sequence Motif Logo"
    format = LogoFormat(data, options)
    logo = Logo(data, options)
    logo.save(output_file)
```

What to Look For:
- Conservation levels at each position
- Base preferences
- Motif boundaries
- Position-specific variations

## 3. Dot Plot Analysis

### Self-Similarity Detection
```python
def create_dot_plot(sequence, window_size=7):
    n = len(sequence)
    matrix = np.zeros((n, n))
    
    for i in range(n - window_size + 1):
        for j in range(n - window_size + 1):
            if sequence[i:i+window_size] == sequence[j:j+window_size]:
                matrix[i, j] = 1
    
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, cmap='binary')
    plt.title('Sequence Dot Plot')
    plt.xlabel('Sequence Position')
    plt.ylabel('Sequence Position')
```

Pattern Analysis:
- Diagonal lines: Direct repeats
- Anti-diagonal lines: Inverted repeats
- Blocks: Repeated regions
- Complexity assessment

## 4. Feature Distribution Plots

### Regulatory Element Mapping
```python
def plot_feature_distribution(sequence, features):
    plt.figure(figsize=(15, 5))
    y_positions = np.arange(len(features))
    
    for i, (feature, pattern) in enumerate(features.items()):
        positions = [m.start() for m in re.finditer(pattern, sequence)]
        plt.scatter(positions, [i] * len(positions), label=feature)
    
    plt.yticks(y_positions, features.keys())
    plt.title('Feature Distribution Along Sequence')
    plt.xlabel('Sequence Position')
    plt.legend(bbox_to_anchor=(1.05, 1))
```

Key Elements to Map:
- Promoter elements
- Regulatory motifs
- Start/Stop codons
- Splice sites
- Enhancer elements

## 5. Quality Score Visualization

### Sequence Quality Metrics
```python
def plot_quality_metrics(sequence, window_size=50):
    metrics = calculate_quality_metrics(sequence, window_size)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Complexity score
    ax1.plot(metrics['complexity'])
    ax1.set_title('Sequence Complexity')
    
    # Entropy
    ax2.plot(metrics['entropy'])
    ax2.set_title('Sequence Entropy')
    
    # Repeat content
    ax3.plot(metrics['repeat_content'])
    ax3.set_title('Repeat Content')
    
    plt.tight_layout()
```

Quality Indicators:
- Complexity score: >0.7 desired
- Entropy: Higher is better
- Repeat content: <30% preferred

## 6. Comparative Analysis

### Multiple Sequence Comparison
```python
def plot_sequence_comparison(sequences, window_size=50):
    plt.figure(figsize=(15, 8))
    
    for i, seq in enumerate(sequences):
        gc_content = calculate_gc_content(seq, window_size)
        plt.plot(gc_content, label=f'Sequence {i+1}')
    
    plt.title('Comparative GC Content Analysis')
    plt.xlabel('Window Position')
    plt.ylabel('GC Content')
    plt.legend()
```

Comparison Points:
- Base composition
- Motif distribution
- Feature locations
- Quality metrics

## 7. Statistical Analysis Plots

### Nucleotide Statistics
```python
def plot_nucleotide_stats(sequence):
    # Single nucleotide frequencies
    singles = calculate_nucleotide_freq(sequence)
    
    # Dinucleotide frequencies
    dinucs = calculate_dinucleotide_freq(sequence)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot single nucleotide frequencies
    ax1.bar(singles.keys(), singles.values())
    ax1.set_title('Nucleotide Frequencies')
    
    # Plot dinucleotide frequencies
    ax2.bar(dinucs.keys(), dinucs.values())
    ax2.set_title('Dinucleotide Frequencies')
    plt.xticks(rotation=45)
```

Statistical Measures:
- Base frequencies
- Dinucleotide bias
- k-mer distributions
- Positional bias

## Best Practices for Visualization

1. Color Usage:
- Use colorblind-friendly palettes
- Consistent color schemes
- Clear contrast between elements

2. Layout:
- Proper figure sizing
- Clear axes labels
- Informative titles
- Appropriate legends

3. Data Processing:
- Appropriate window sizes
- Normalized values
- Statistical significance

4. Export Settings:
- High resolution (300 DPI minimum)
- Vector formats when possible
- Consistent sizing

## Workflow Integration

1. Automated Analysis Pipeline:
```python
def analyze_sequence(sequence, output_dir):
    """Complete analysis pipeline"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    plot_gc_content(sequence)
    plt.savefig(f'{output_dir}/gc_content.png')
    
    plot_feature_distribution(sequence, features)
    plt.savefig(f'{output_dir}/features.png')
    
    create_dot_plot(sequence)
    plt.savefig(f'{output_dir}/dot_plot.png')
    
    plot_quality_metrics(sequence)
    plt.savefig(f'{output_dir}/quality.png')
    
    # Generate report
    generate_analysis_report(sequence, output_dir)
```

2. Regular Monitoring:
- Track metrics over time
- Compare against benchmarks
- Document anomalies
- Update visualization parameters

## Interpretation Guidelines

1. Quality Assessment:
- Compare against known sequences
- Check for biological plausibility
- Identify potential artifacts
- Validate against databases

2. Pattern Recognition:
- Identify common motifs
- Detect unusual patterns
- Compare with expected features
- Flag potential issues

3. Documentation:
- Record analysis parameters
- Save visualization settings
- Document unusual findings
- Maintain version control

