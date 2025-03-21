# preprocessing.py
import numpy as np
import pandas as pd

def build_dictionary(sequences, vocab=None, max_sent_len_=None):
    """
    Build a vocabulary dictionary from protein sequences and encode them into integer tokens.
    
    Args:
        sequences (list of str): List of protein sequences.
        vocab (dict, optional): Predefined vocabulary mapping tokens to integer IDs.
                                If None, a new vocabulary is created with default tokens.
        max_sent_len_ (int, optional): Maximum sequence length for encoding. Overrides computed length if provided.
    
    Returns:
        tuple: (vocab, reverse_vocab, sequence_lengths, encoding_length, padded_encoded_sequences)
    """
    if vocab is None:
        vocab = {'<PAD>': 0, '<OOV>': 1}
        external_vocab = False
    else:
        external_vocab = True

    encoded_sequences = []
    max_seq_len = 0

    for seq in sequences:
        encoded_seq = []
        for char in seq:
            if not external_vocab and char not in vocab:
                vocab[char] = len(vocab)
            token_id = vocab.get(char, vocab['<OOV>'])
            encoded_seq.append(token_id)
        max_seq_len = max(max_seq_len, len(encoded_seq))
        encoded_sequences.append(encoded_seq)

    if max_sent_len_ is not None:
        max_seq_len = max_sent_len_

    num_sequences = len(encoded_sequences)
    padded_sequences = np.full((num_sequences, max_seq_len), vocab['<PAD>'], dtype=np.int32)

    sequence_lengths = []
    for i, seq in enumerate(encoded_sequences):
        padded_sequences[i, :len(seq)] = seq
        sequence_lengths.append(len(seq))
    sequence_lengths = np.array(sequence_lengths, dtype=np.int32)

    # Create reverse mapping from token IDs to tokens
    reverse_vocab = {idx: token for token, idx in vocab.items()}

    return vocab, reverse_vocab, sequence_lengths, max_seq_len + 1, padded_sequences

def summary_stats(lengths, labels, dataset_name):
    """
    Generate summary statistics and a frequency table for sequence lengths grouped by class.
    
    Args:
        lengths (array-like): Sequence lengths.
        labels (array-like): Corresponding class labels.
        dataset_name (str): Name of the dataset (e.g., 'train' or 'test').
    
    Returns:
        tuple: (DataFrame of raw data, crosstab summary table)
    """
    bins = [0, 100, 500, 1000, 1500, 2000, 2499]
    class_names = ['cyto', 'secreted', 'mito', 'nucleus']

    df = pd.DataFrame({'length': lengths, 'label': labels})
    bin_indices = np.digitize(df['length'], bins)
    table = pd.crosstab(bin_indices, df['label'])
    table.index = pd.Index(
        ['[0,100)', '[100,500)', '[500,1000)', '[1000,1500)', '[1500,2000)', '[2000,2500)', '[2500,inf]'],
        name="Bin"
    )
    table.columns = pd.Index(class_names, name="Class")

    # Add row and column totals
    table['Total'] = table.sum(axis=1)
    total_row = pd.DataFrame(table.sum()).T
    total_row.index = ['Total']
    summary_table = pd.concat([table, total_row])
    
    print(f"\n~~~~~~~ Summary stats for {dataset_name} set ~~~~~~~")
    print("\nCount of sequence lengths by class")
    print(summary_table)
    print("\nDescriptive statistics")
    print(df.describe())
    
    return df, summary_table