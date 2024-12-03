# Life Sciences AI Testing Guide

## Test Cases and Expected Outcomes

### 1. Basic Promoter Tests

```python
seed = "TATA"  # TATA box promoter
generated = generate_sequence(model, tokenizer, seed, max_length=100, temperature=0.8)
```

Expected Outcomes:
- Sequence should maintain TATA box characteristics
- Look for downstream promoter elements
- Check for proper spacing (25-35 bp) from transcription start
- Verify GC-rich regions around TATA box
- Common patterns: TATAAA, TATAAAA, or TATATAA

### 2. Start Codon Tests

```python
seed = "ATG"  # Start codon
generated = generate_sequence(model, tokenizer, seed, max_length=1000, temperature=0.7)
```

Expected Outcomes:
- Should begin with ATG
- Look for Kozak sequence context (GCCACC upstream of ATG)
- Check for in-frame stop codons (TAA, TAG, TGA)
- Verify codon usage matches expected frequencies
- Look for open reading frame maintenance

### 3. Terminator Sequence Tests

```python
seed = "AATAAA"  # Polyadenylation signal
gene