import numpy as np
import pandas as pd
import re
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from typing import List

# Global constants
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"  # Standard 20 amino acids

# Hydrophobicity & Charge Tables
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8,
    'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0, 'G': 0, 'H': 1, 'I': 0,
    'K': 1, 'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1, 'S': 0,
    'T': 0, 'V': 0, 'W': 0, 'Y': 0
}

# Precompile motif regex patterns for efficiency
MOTIF_PATTERNS = {
    "mito_signal": re.compile(r"M.{5,10}R"),  # Example mitochondrial motif
    "nuclear_NLS": re.compile(r"K.{2,4}K"),     # Nuclear Localization Signal
    "ER_KDEL": re.compile(r"KDEL$")             # ER retention signal
}

def clean_sequence(sequence: str) -> str:
    """
    Remove invalid characters from the protein sequence.
    
    Args:
        sequence (str): Input protein sequence.
        
    Returns:
        str: Cleaned protein sequence containing only valid amino acids.
    """
    sequence = sequence.upper()
    valid_amino_acids = set(AA_LIST)
    return "".join(aa for aa in sequence if aa in valid_amino_acids)

def compute_aa_composition(sequence: str) -> np.ndarray:
    """
    Compute amino acid composition as percentages.
    
    Args:
        sequence (str): Protein sequence.
        
    Returns:
        np.ndarray: Array of composition percentages for each amino acid in AA_LIST.
    """
    if not sequence:
        return np.zeros(len(AA_LIST))
    count = Counter(sequence)
    return np.array([count.get(aa, 0) / len(sequence) for aa in AA_LIST])

def compute_local_aa_composition(sequence: str, window_size: int = 50) -> np.ndarray:
    """
    Compute amino acid composition for the first and last window_size residues.
    
    Args:
        sequence (str): Protein sequence.
        window_size (int): Size of the segment to consider from start and end.
        
    Returns:
        np.ndarray: Concatenated amino acid composition for start and end segments.
    """
    first_segment = sequence[:window_size] if len(sequence) >= window_size else sequence
    last_segment = sequence[-window_size:] if len(sequence) >= window_size else sequence
    first_comp = compute_aa_composition(first_segment)
    last_comp = compute_aa_composition(last_segment)
    return np.concatenate([first_comp, last_comp])

def compute_molecular_properties(sequence: str) -> np.ndarray:
    """
    Compute molecular properties: isoelectric point and molecular weight.
    Assumes the sequence is already cleaned.
    
    Args:
        sequence (str): Protein sequence.
        
    Returns:
        np.ndarray: Array containing isoelectric point and molecular weight.
    """
    if not sequence:
        return np.array([np.nan, np.nan])
    analysis = ProteinAnalysis(sequence)
    return np.array([analysis.isoelectric_point(), analysis.molecular_weight()])

def contains_signal_motif(sequence: str) -> np.ndarray:
    """
    Detect the presence of known localization motifs.
    
    Args:
        sequence (str): Protein sequence.
        
    Returns:
        np.ndarray: Binary array indicating presence (1) or absence (0) for each motif.
    """
    return np.array([1 if pattern.search(sequence) else 0 for pattern in MOTIF_PATTERNS.values()])

def compute_hydrophobicity_charge(sequence: str) -> np.ndarray:
    """
    Compute mean hydrophobicity and total charge of the sequence.
    
    Args:
        sequence (str): Protein sequence.
        
    Returns:
        np.ndarray: Array with mean hydrophobicity and total charge.
    """
    hydro_scores = [HYDROPHOBICITY.get(aa, 0) for aa in sequence]
    charge_scores = [CHARGE.get(aa, 0) for aa in sequence]
    return np.array([np.mean(hydro_scores) if hydro_scores else 0, np.sum(charge_scores)])

def compute_physicochemical_features(sequence: str) -> np.ndarray:
    """
    Compute aromaticity, GRAVY (hydrophobicity index), and aliphatic index.
    
    Args:
        sequence (str): Protein sequence.
        
    Returns:
        np.ndarray: Array of computed physicochemical features.
    """
    if not sequence:
        return np.zeros(3)
    analysis = ProteinAnalysis(sequence)
    aromaticity = analysis.aromaticity()
    gravy = analysis.gravy()
    aliphatic_index = (sequence.count('A') + 2 * sequence.count('V') +
                       2.9 * (sequence.count('I') + sequence.count('L'))) / len(sequence)
    return np.array([aromaticity, gravy, aliphatic_index])

def extract_features(sequence: str) -> np.ndarray:
    """
    Extract a comprehensive set of features from a protein sequence.
    
    Features include:
      - Sequence length
      - Global amino acid composition (20 features)
      - Local amino acid composition (first and last segments; 40 features)
      - Molecular properties (isoelectric point, molecular weight; 2 features)
      - Presence of signal motifs (3 features)
      - Hydrophobicity and charge (2 features)
      - Additional physicochemical features (aromaticity, GRAVY, aliphatic index; 3 features)
    
    Total feature dimension: 71.
    
    Args:
        sequence (str): Input protein sequence.
        
    Returns:
        np.ndarray: Concatenated feature vector.
    """
    sequence = clean_sequence(sequence)

    if not sequence:
        # Return default values for empty sequences: dimension = 71
        return np.concatenate([
            np.array([0]),           # Sequence length
            np.zeros(20),            # Global amino acid composition
            np.zeros(40),            # Local amino acid composition (start + end)
            np.array([np.nan, np.nan]),  # Molecular properties
            np.zeros(3),             # Signal motifs
            np.array([0, 0]),        # Hydrophobicity and charge
            np.zeros(3)              # Physicochemical features
        ])

    seq_len = np.array([len(sequence)])
    global_comp = compute_aa_composition(sequence)
    local_comp = compute_local_aa_composition(sequence)
    mol_props = compute_molecular_properties(sequence)
    motifs = contains_signal_motif(sequence)
    hydro_charge = compute_hydrophobicity_charge(sequence)
    physicochem = compute_physicochemical_features(sequence)

    return np.concatenate([seq_len, global_comp, local_comp, mol_props, motifs, hydro_charge, physicochem])

def process_sequences(sequences: List[str]) -> np.ndarray:
    """
    Process a list of protein sequences to extract feature vectors.
    
    Args:
        sequences (List[str]): List of protein sequences.
        
    Returns:
        np.ndarray: 2D array of extracted features (num_sequences x feature_dimension).
    """
    features = []
    for i, seq in enumerate(sequences):
        try:
            feat = extract_features(seq)
            features.append(feat)
        except Exception as e:
            print(f"Error processing sequence at index {i} (first 100 chars: {seq[:100]}): {e}")
            raise e
    return np.array(features)
