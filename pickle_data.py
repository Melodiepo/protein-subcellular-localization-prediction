# pickle_data.py
from .data_fasta import _get_data # raw data
from config import pkl_path
import pickle 
import os

def ensure_dir(directory):
    """Ensure that the directory exists; create it if not."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_pkl():
    """
    Load sequences from FASTA, process them, and store as pickled files.
    This creates separate pickle files for training sequences, training labels, and test sequences.
    """
    train_data, test_data, _ = _get_data()  # Load raw data from FASTA
    
    # Extract sequences and labels
    train_seqs = train_data['seq']
    train_labels = train_data['class']
    test_seqs = test_data['seq']
    
    ensure_dir(pkl_path)
    
    with open(os.path.join(pkl_path, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_seqs, f)
    with open(os.path.join(pkl_path, 'train_labels.pkl'), 'wb') as f:
        pickle.dump(train_labels, f)
    with open(os.path.join(pkl_path, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_seqs, f)
def get_data(train=True):
    """
    Load pickled sequence data. If the pickle files do not exist, generate them from FASTA.
    
    Args:
        train (bool): Flag indicating whether to load training data.
        
    Returns:
        tuple: (data, labels) where labels is None for test data.
    """
    prefix = 'train' if train else 'test'
    data_file = os.path.join(pkl_path, f'{prefix}_data.pkl')
    label_file = os.path.join(pkl_path, f'{prefix}_labels.pkl') if train else None

    if not os.path.exists(data_file):
        print("Pickle files not found, generating from FASTA...")
        generate_pkl()  

    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    labels = None
    if train:
        with open(label_file, 'rb') as f:
            labels = pickle.load(f)
    
    max_len = max(len(seq) for seq in data)
    print(f"Max sequence length ({'train' if train else 'test'}): {max_len}")
    print(f"Total {'train' if train else 'test'} sequences: {len(data)}")

    return data, labels