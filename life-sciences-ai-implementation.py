# sequence_generator.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import random
from collections import defaultdict

class SequenceTokenizer:
    """Handles tokenization of biological sequences"""
    
    def __init__(self, sequence_type: str):
        self.sequence_type = sequence_type.upper()
        self.token_to_idx = {}
        self.idx_to_token = {}
        
        if sequence_type.upper() == 'DNA':
            tokens = ['<PAD>', '<START>', '<END>', 'A', 'T', 'C', 'G', 'N']
        elif sequence_type.upper() == 'RNA':
            tokens = ['<PAD>', '<START>', '<END>', 'A', 'U', 'C', 'G', 'N']
        elif sequence_type.upper() == 'PROTEIN':
            tokens = ['<PAD>', '<START>', '<END>'] + list('ACDEFGHIKLMNPQRSTVWY')
        
        for idx, token in enumerate(tokens):
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
        
        self.vocab_size = len(tokens)
    
    def encode(self, sequence: str) -> torch.Tensor:
        """Convert sequence to tensor of indices"""
        tokens = ['<START>'] + list(sequence) + ['<END>']
        return torch.tensor([self.token_to_idx[t] for t in tokens])
    
    def decode(self, indices: torch.Tensor) -> str:
        """Convert tensor of indices back to sequence"""
        tokens = [self.idx_to_token[idx.item()] for idx in indices]
        # Remove special tokens
        tokens = [t for t in tokens if t not in ['<PAD>', '<START>', '<END>']]
        return ''.join(tokens)

class BiologicalDataGenerator:
    """Generates synthetic biological sequences for training"""
    
    def __init__(self, sequence_type: str):
        self.sequence_type = sequence_type.upper()
        if sequence_type == 'DNA':
        