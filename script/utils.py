# utils.py
import torch as T
import numpy as np

def get_cuda(tensor):
    """
    Move a tensor or module to GPU if available.
    """
    if T.cuda.is_available():
        return tensor.cuda()
    return tensor

def gradient_clamping(model, clip_value):
    """
    Clamp gradients of all model parameters by a specified clip_value.
    """
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def data_shuffle(enc_seq, enc_label, enc_length, random_seed=None):
    """
    Shuffle data, labels, lengths using same permutation.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    perm_rnd = np.random.permutation(len(enc_seq))
    perm_seq = enc_seq[perm_rnd]
    perm_label = enc_label[perm_rnd]
    perm_length = enc_length[perm_rnd]

    return perm_seq, perm_label, perm_length


def truncate(enc_sentences, sentence_lengths, max_enc_length=None):
    """
    If max_enc_length is None, do NOT truncate. Just return original arrays.
    """
    if max_enc_length is None:
        return enc_sentences, sentence_lengths
    
    # Otherwise do real truncation
    import copy
    original_length = copy.copy(sentence_lengths)
    enc_truncated = enc_sentences[:, :max_enc_length]
    for i in range(len(sentence_lengths)):
        if sentence_lengths[i] > max_enc_length:
            enc_truncated[i] = np.concatenate(
                (enc_sentences[i, : (max_enc_length - 100)], enc_sentences[i, -100:]),
                axis=0
            )
            sentence_lengths[i] = max_enc_length
    
    return enc_truncated, sentence_lengths

