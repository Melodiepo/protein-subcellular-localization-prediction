# experiments.py
"""
Defines multiple "experiment_*" functions for different model architectures
(e.g., LSTM, BiLSTM, CNN+LSTM, etc.), each controlling:
1) Data shuffle/truncation
2) Model initialization & optional checkpoint loading
3) K-fold training & result collection
4) Confusion matrix plot
"""
import numpy as np
import torch as T
import matplotlib.pyplot as plt
import os
from .utils import data_shuffle, truncate
from .trainer import run

def experiment_lstm_no_attention(
    enc_sentences, sentence_lengths, Labels, vocab, reverse_dictionary,
    batch_size=32, output_size=4, hidden_size=128, embedding_size=64,
    num_epoch=32, lr=0.001, kernel_size=None, bidirection=False, if_attention=False,
    gra_clip=True, k_fold=5, fold_idx=1, save_model_path=None, random_seed=42, load_epoch=None
):
    """
    1) LSTM without Attention
    Demonstrates how you can replicate the 'LSTM' experiment.
    """
    # Step 1: Shuffle data
    perm_seq, perm_label, perm_length = data_shuffle(
        enc_sentences, np.array(Labels), sentence_lengths, random_seed
    )
    enc_truncated, enc_length = perm_seq, perm_length

    # Step 2: Initialize run() object
    LSTM_1 = run(
        batch_size=batch_size,
        output_size=output_size,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        embedding_size=embedding_size,
        num_epoch=num_epoch,
        lr=lr,
        gradient_clip=gra_clip,
        kernel_size=[] if kernel_size is None else kernel_size,
        vocab=vocab
    )
    
    # Load model if checkpoint exists
    LSTM_1.setup_model(bidirection, if_attention, model_name='lstm')
    
    if load_epoch is not None:
        checkpoint_path = os.path.join(save_model_path, f"{load_epoch:05d}.tar")
        if os.path.exists(checkpoint_path):
            print(f"🔄 Loading model from checkpoint: {checkpoint_path}")
            checkpoint = T.load(checkpoint_path, map_location=T.device("cpu"))
            
            LSTM_1.model.load_state_dict(checkpoint["model_dict"])  # Load model weights
            print("✅ Model successfully loaded. Skipping training.")
            
            LSTM_1.train_loss = checkpoint.get("train_loss", [])
            LSTM_1.valid_loss = checkpoint.get("valid_loss", [])
            LSTM_1.train_acc = checkpoint.get("train_acc", [])
            LSTM_1.valid_acc = checkpoint.get("valid_acc", [])
            
            return LSTM_1, None  # Return the loaded model, skip training
        else:
            print(f"⚠️ Checkpoint {checkpoint_path} not found! Running full training...")
            
    # Step 3: Perform k-fold
    TL1, Tacc1, VL1, Vacc1, count_matrix, y_true, y_pred, y_prob = LSTM_1.lets_go(
        enc_length, enc_truncated, perm_label, k_fold, fold_idx,
        bidirection, if_attention, model_name='lstm', save_model_path=save_model_path
    )

    # Step 4: Optional matrix plot
    LSTM_1.matrix_plot()
    plt.show()

    return LSTM_1, (TL1, Tacc1, VL1, Vacc1, count_matrix, y_true, y_pred, y_prob)

def experiment_bilstm_no_attention(
    enc_sentences, sentence_lengths, Labels, vocab, reverse_dictionary,
    batch_size=32, output_size=4, hidden_size=128, embedding_size=64,
    num_epoch=32, lr=0.001, kernel_size=None, bidirection=True, if_attention=False,
    gra_clip=True, k_fold=5, fold_idx=1, save_model_path=None, random_seed=12, load_epoch=None
):
    """
    2) Bi-LSTM without Attention
    """
    perm_seq, perm_label, perm_length = data_shuffle(
        enc_sentences, np.array(Labels), sentence_lengths, random_seed
    )
    enc_truncated, enc_length = perm_seq, perm_length

    BLSTM = run(
        batch_size=batch_size,
        output_size=output_size,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        embedding_size=embedding_size,
        num_epoch=num_epoch,
        lr=lr,
        gradient_clip=gra_clip,
        kernel_size=[] if kernel_size is None else kernel_size,
        vocab=vocab
    )
    
    # Load model if checkpoint exists
    BLSTM.setup_model(bidirection, if_attention, model_name='lstm')
    
    if load_epoch is not None:
        checkpoint_path = os.path.join(save_model_path, f"{load_epoch:05d}.tar")
        if os.path.exists(checkpoint_path):
            print(f"🔄 Loading model from checkpoint: {checkpoint_path}")
            checkpoint = T.load(checkpoint_path, map_location=T.device("cpu"))
            
            BLSTM.model.load_state_dict(checkpoint["model_dict"])  # Load model weights
            print("✅ Model successfully loaded. Skipping training.")
            
            BLSTM.train_loss = checkpoint.get("train_loss", [])
            BLSTM.valid_loss = checkpoint.get("valid_loss", [])
            BLSTM.train_acc = checkpoint.get("train_acc", [])
            BLSTM.valid_acc = checkpoint.get("valid_acc", [])
            
            return BLSTM, None  # Return the loaded model, skip training
        else:
            print(f"⚠️ Checkpoint {checkpoint_path} not found! Running full training...")
    
    
    TL2, Tacc2, VL2, Vacc2, count_matrix2, y_true, y_pred, y_prob = BLSTM.lets_go(
        enc_length, enc_truncated, perm_label, k_fold, fold_idx,
        bidirection, if_attention, model_name='lstm', save_model_path=save_model_path
    )

    BLSTM.matrix_plot()
    plt.show()

    return BLSTM, (TL2, Tacc2, VL2, Vacc2, count_matrix2, y_true, y_pred, y_prob)


def experiment_lstm_attention(
    enc_sentences, sentence_lengths, Labels, vocab, reverse_dictionary,
    batch_size=32, output_size=4, hidden_size=128, embedding_size=64,
    num_epoch=22, lr=0.001, kernel_size=None, bidirection=False, if_attention=True,
    gra_clip=True, k_fold=5, fold_idx=1, save_model_path=None, random_seed=12, load_epoch=None
):
    """
    3) LSTM + Attention
    """
    perm_seq, perm_label, perm_length = data_shuffle(
        enc_sentences, np.array(Labels), sentence_lengths, random_seed
    )
    enc_truncated, enc_length = perm_seq, perm_length

    LSTM_att = run(
        batch_size=batch_size,
        output_size=output_size,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        embedding_size=embedding_size,
        num_epoch=num_epoch,
        lr=lr,
        gradient_clip=gra_clip,
        kernel_size=[] if kernel_size is None else kernel_size,
        vocab=vocab
    )
    
    # Load model if checkpoint exists
    LSTM_att.setup_model(bidirection, if_attention, model_name='lstm')
    
    if load_epoch is not None:
        checkpoint_path = os.path.join(save_model_path, f"{load_epoch:05d}.tar")
        if os.path.exists(checkpoint_path):
            print(f"🔄 Loading model from checkpoint: {checkpoint_path}")
            checkpoint = T.load(checkpoint_path, map_location=T.device("cpu"))
            
            LSTM_att.model.load_state_dict(checkpoint["model_dict"])  # Load model weights
            print("✅ Model successfully loaded. Skipping training.")
            
            LSTM_att.train_loss = checkpoint.get("train_loss", [])
            LSTM_att.valid_loss = checkpoint.get("valid_loss", [])
            LSTM_att.train_acc = checkpoint.get("train_acc", [])
            LSTM_att.valid_acc = checkpoint.get("valid_acc", [])
            
            return LSTM_att, None  # Return the loaded model, skip training
        else:
            print(f"⚠️ Checkpoint {checkpoint_path} not found! Running full training...")
            
    
    TL_att, Tacc_att, VL_att, Vacc_att, count_matrix_att, y_true, y_pred, y_prob = LSTM_att.lets_go(
        enc_length, enc_truncated, perm_label, k_fold, fold_idx,
        bidirection, if_attention, model_name='lstm', save_model_path=save_model_path
    )

    LSTM_att.matrix_plot()
    plt.show()
    
    return LSTM_att, (TL_att, Tacc_att, VL_att, Vacc_att, count_matrix_att, y_true, y_pred, y_prob)


def attention_analysis(LSTM_att, reverse_dictionary, threshold=0.05, example_idxs=None):
    """
    Show how to replicate your post-training attention analysis:
        - Extract attention
        - Print peptides
        - Plot some annotated attention positions
        - Possibly build a heatmap
    """
    # 1. Get attention dict from the trainer
    at_dict, data_at_dict = LSTM_att.attention()
    # Let's say we focus on class '1' (secreted)
    at_se = T.cat(at_dict['1']).cpu().numpy()
    data_se = T.cat(data_at_dict['1']).cpu().numpy()

    # 2. Build seq_se
    seq_se = []
    for i in range(data_se.shape[0]):
        temp = ''
        for j in range(data_se.shape[1]):
            if data_se[i,j] != 0:
                temp += reverse_dictionary[data_se[i,j]]
        seq_se.append(temp)

    # 3. Mark positions above threshold
    peptide = []
    for i in range(at_se.shape[0]):
        tmp = ''
        for j in range(at_se.shape[1]):
            if at_se[i,j] == 0:
                break
            if at_se[i,j] > threshold:
                tmp += seq_se[i][j]
            else:
                tmp += '_'
        peptide.append(tmp)

    # 4. Optionally plot annotated attention for a few sequences
    if example_idxs is None:
        example_idxs = [3, 6, 8, 203]  # from your code
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    for idx, subplot_idx in zip(example_idxs, [(0,0), (0,1), (1,0), (1,1)]):
        row, col = subplot_idx
        ax[row, col].plot(at_se[idx][:len(seq_se[idx])], label="Secreted protein")
        # Annotate
        for i, txt in enumerate(seq_se[idx]):
            ax[row, col].annotate(txt, (i, at_se[idx][i]), fontsize='xx-large')
        ax[row, col].set_xlabel('Position')
        ax[row, col].set_ylabel('Attention weight')
        ax[row, col].legend()

    plt.show()

    # 5. Return anything else you might want for further analysis
    return at_se, seq_se, peptide


def experiment_cnn_lstm_no_attention(
    enc_sentences, sentence_lengths, Labels, vocab,
    batch_size=32, output_size=4, hidden_size=128, embedding_size=64,
    kernel_size=[3, 5,9],
    num_epoch=25, lr=0.001, bidirection=False, if_attention=False,
    gra_clip=True, k_fold=5, fold_idx=1, save_model_path=None, random_seed=42, load_epoch=None
):
    """
    4) CNN + LSTM (no attention)
    """
    perm_seq, perm_label, perm_length = data_shuffle(
        enc_sentences, np.array(Labels), sentence_lengths, random_seed
    )
    enc_truncated, enc_length = perm_seq, perm_length

    LSTM_cnn = run(
        batch_size=batch_size,
        output_size=output_size,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        embedding_size=embedding_size,
        num_epoch=num_epoch,
        lr=lr,
        gradient_clip=gra_clip,
        kernel_size=kernel_size,
        vocab=vocab
    )
    
    # Load model if checkpoint exists
    LSTM_cnn.setup_model(bidirection, if_attention, model_name='CNN-lstm')
    
    if load_epoch is not None:
        checkpoint_path = os.path.join(save_model_path, f"{load_epoch:05d}.tar")
        if os.path.exists(checkpoint_path):
            print(f"🔄 Loading model from checkpoint: {checkpoint_path}")
            checkpoint = T.load(checkpoint_path, map_location=T.device("cpu"))
            
            LSTM_cnn.model.load_state_dict(checkpoint["model_dict"])  # Load model weights
            print("✅ Model successfully loaded. Skipping training.")
            
            LSTM_cnn.train_loss = checkpoint.get("train_loss", [])
            LSTM_cnn.valid_loss = checkpoint.get("valid_loss", [])
            LSTM_cnn.train_acc = checkpoint.get("train_acc", [])
            LSTM_cnn.valid_acc = checkpoint.get("valid_acc", [])
            
            return LSTM_cnn, None  # Return the loaded model, skip training
        else:
            print(f"⚠️ Checkpoint {checkpoint_path} not found! Running full training...")
            
    TL4, Tacc4, VL4, Vacc4, count_matrix4, y_true, y_pred, y_prob = LSTM_cnn.lets_go(
        enc_length, enc_truncated, perm_label, k_fold, fold_idx,
        bidirection, if_attention, model_name='CNN-lstm', save_model_path=save_model_path
    )
    LSTM_cnn.matrix_plot()
    plt.show()

    return LSTM_cnn, (TL4, Tacc4, VL4, Vacc4, count_matrix4, y_true, y_pred, y_prob)

def experiment_cnn_bilstm_attention(
    enc_sentences, sentence_lengths, Labels, vocab,
    batch_size=32, output_size=4, hidden_size=128, embedding_size=64,
    kernel_size=[3,5,9],
    num_epoch=25, lr=0.001, if_attention=True,
    gra_clip=True, k_fold=5, fold_idx=1, save_model_path=None, random_seed=42, load_epoch=None
):
    """
    CNN + BiLSTM + Attention
    """
    from .trainer import run
    from .utils import data_shuffle

    # 1) Shuffle data
    perm_seq, perm_label, perm_length = data_shuffle(
        enc_sentences, np.array(Labels), sentence_lengths, random_seed
    )
    enc_truncated, enc_length = perm_seq, perm_length

    # 2) Initialize run
    model_runner = run(
        batch_size=batch_size,
        output_size=output_size,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        embedding_size=embedding_size,
        num_epoch=num_epoch,
        lr=lr,
        gradient_clip=gra_clip,
        kernel_size=kernel_size,
        vocab=vocab
    )

    # 3) Load checkpoint if provided...
    #    (similar to your other experiment functions)
    #    e.g., check if load_epoch is not None, attempt to load

    # 4) K-fold cross validation or single pass
    TL, Tacc, VL, Vacc, cm, y_true, y_pred, y_prob = model_runner.lets_go(
        enc_length, enc_truncated, perm_label, k_fold, fold_idx,
        if_bidirection=True,   # BiLSTM
        if_attention=if_attention,
        model_name='cnn_bilstm',   # NOTE: must match "elif model_name == 'cnn_bilstm'"
        save_model_path=save_model_path
    )

    # 5) Optionally plot matrix
    model_runner.matrix_plot()
    plt.show()

    return model_runner, (TL, Tacc, VL, Vacc, cm, y_true, y_pred, y_prob)


def train_on_full_data_and_test(
    enc_sentences, sentence_lengths, Labels, test_dict,
    vocab, reverse_dictionary,
    batch_size=32, output_size=4, hidden_size=128, embedding_size=64,
    kernel_size=[3,5,9],
    num_epoch=20, lr=0.001, bidirection=False, if_attention=True,
    gra_clip=True, save_model_path=None
):
    """
    6) Train on full data, then test
    """
    # Shuffle/truncate as needed
    perm_seq, perm_label, perm_length = data_shuffle(
        enc_sentences, np.array(Labels), sentence_lengths, random_seed=42
    )
    enc_truncated, enc_length = perm_seq, perm_length

    full_run = run(
        batch_size=batch_size,
        output_size=output_size,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        embedding_size=embedding_size,
        num_epoch=num_epoch,
        lr=lr,
        gradient_clip=gra_clip,
        kernel_size=kernel_size,
        vocab=vocab, 
        test=test_dict     # pass in test so we can do print_frame
    )
    # Train on full data
    TL_full, Tacc_full = full_run.train_full_data(
        enc_length, enc_truncated, perm_label,
        bidirection, if_attention, model_name='lstm', save_model_path=save_model_path
    )

    # 7) Testing
    # Assume test data is in test_dict['enc_test'] or something
    if 'enc_test_sentences' in test_dict and 'test_sentence_lengths' in test_dict:
        enc_test = test_dict['enc_test_sentences']
        enc_test_length = test_dict['test_sentence_lengths']
        att_test = full_run.testing(enc_test, enc_test_length)
        df = full_run.print_frame()
        return full_run, (TL_full, Tacc_full), df, att_test
    else:
        return full_run, (TL_full, Tacc_full), None, None
