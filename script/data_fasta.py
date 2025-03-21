# data_fasta.py
import os
from .config import data_path, test_name, classes

def _parse_fasta(data, lines, label):
    """
    Parse FASTA-formatted lines and append sequence information to the data dictionary.
    
    Args:
        data (dict): Dictionary with keys 'info', 'seq', 'class'.
        lines (list): Lines read from a FASTA file.
        label: Class label for the sequences (None for test data).
    
    Returns:
        int: Maximum sequence length encountered.
    """
    lines_iter = iter(lines)
    line = next(lines_iter, None)
    max_len = 0
    while line:
        if not line.startswith('>'):
            raise ValueError("FASTA format error: Expected a header line starting with '>'")
        info = line.strip()
        line = next(lines_iter, None)
        seq = ''
        while line and not line.startswith('>'):
            seq += line.strip()
            line = next(lines_iter, None)
        max_len = max(max_len, len(seq))
        data['info'].append(info)
        data['seq'].append(seq)
        data['class'].append(label)
    return max_len

def _load_data(data, name, label):
    """
    Load and parse a FASTA file.
    
    Args:
        data (dict): Dictionary to store parsed data.
        name (str): Name of the FASTA file (without extension).
        label: Class label for the sequences.
    
    Returns:
        int: Maximum sequence length found in the file.
    """
    file_path = os.path.join(data_path, f"{name}.fasta")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return _parse_fasta(data, lines, label)

def _get_data():
    """
    Load training and test data from FASTA files.
    
    Returns:
        tuple: (train_data, test_data, max_length)
    """
    train = {'info': [], 'seq': [], 'class': []}
    test = {'info': [], 'seq': [], 'class': []}
    
    max_len = _load_data(test, test_name, None)
    print(f"Max sequence length (test): {max_len}")
    
    for cls in classes:
        seq_len = _load_data(train, cls, cls)
        max_len = max(max_len, seq_len)

    print(f"Max sequence length (overall): {max_len}")
    print(f"Train sequences: {len(train['info'])}")
    print(f"Test sequences: {len(test['info'])}")
    return train, test, max_len