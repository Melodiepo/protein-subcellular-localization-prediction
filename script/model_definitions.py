# model_definitions.py
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from .utils import get_cuda
from .config import max_enc_length 

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class attentionLSTM(nn.Module):
    """
    LSTM model with optional attention.
    """
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_size, 
                 bidirection, if_attention):
        super(attentionLSTM, self).__init__()
        self.batch_size = batch_size # number of samples per batch
        self.output_size = output_size # number of classes
        self.hidden_size = hidden_size # number of LSTM units per layer
        self.vocab_size = vocab_size # number of unique tokens in the vocab
        self.embedding_size = embedding_size # size of the word embeddings
        self.bidirection = bidirection # whether to use Bi-LSTM
        self.if_attention = if_attention # whether to use attention

        # 1. Word Embedding Layer
        # Convert integer token indices to dense vector representations of size embedding_szie
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        # Learnable initial hidden state & cell state
        # Normally, initial states are zero vectors, but here trainable
        self.h_0 = nn.Parameter(
            T.rand(2 if self.bidirection else 1, self.batch_size, self.hidden_size, device=device),
            requires_grad=True
        )
        self.c_0 = nn.Parameter(
            T.rand(2 if self.bidirection else 1, self.batch_size, self.hidden_size, device=device),
            requires_grad=True
        )

        # 2. LSTM Layer
        # Takes embedded sequence as input and outputs hidden and cell states
        self.lstm = nn.LSTM(
            self.embedding_size, self.hidden_size,
            batch_first=False, bidirectional=self.bidirection
        )
        
        # 3. Fully Connected Layer
        # Linear 1: maps from hidden_size to hidden_size 
        # Linear 2: maps from hidden_size to output_size
        factor = 2 if self.bidirection else 1
        self.linear1 = nn.Linear(self.hidden_size * factor, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)

        # Attention
        # Project LSTM hidden states to a common space
        self.att_wh = nn.Linear(self.hidden_size * factor, self.hidden_size * factor, bias=False)
        # Project the final hidden state h_T
        self.att_ws = nn.Linear(self.hidden_size * factor, self.hidden_size * factor)
        # Learns importance scores for each position
        self.att_v = nn.Linear(self.hidden_size * factor, 1, bias=False)

    def attention_net(self, output_lstm, last_hidden, enc_padding_mask):
        """
        Compute attention weights over the sequence of hidden states.
        
        last_hidden: [num_direction, batch_size, hidden_size]
        output_lstm: [batch_size, seq_length, num_direction * hidden_size]
        enc_padding_mask: [batch_size, seq_length]
        
        """
        et = self.att_wh(output_lstm)  
        hidden = last_hidden.view(last_hidden.shape[1], -1)  
        final_fea = self.att_ws(hidden).unsqueeze(1)        
        et = et + final_fea
        # use tanh activation to normalize scores
        et = T.tanh(et)
        # et represents how much each LSTM hidden state h_i aligns with the final hidden state h_T
        et = self.att_v(et).squeeze(2)                    

        # Softmax over seq_length
        et1 = F.softmax(et, dim=1) # convert scores to probabilities
        at = et1 * enc_padding_mask # zero out padded positions
        norm_factor = at.sum(1, keepdim=True) 
        at = at / norm_factor      # re-normalize

        at = at.unsqueeze(1)                  
        ct_e = T.bmm(at, output_lstm).squeeze(1) 
        at = at.squeeze(1)
        return ct_e, at # this ct_e is a weighted sum of LSTM hidden states

    def forward(self, input_sequence, enc_padding_mask):
        # 1. Compute Word Embeddings
        # input_sequence: [batch_size, seq_length]
        embedded = self.embedding(input_sequence)    
        # We swap batch_size and seq_length for LSTM
        embedded = embedded.permute(1, 0, 2)
        
        
        # 2. Run LSTM
        # Take the embedded sequence as input and output hidden and cell states
        output, (final_hidden_state, _) = self.lstm(embedded, (self.h_0, self.c_0))
        output = output.permute(1, 0, 2)           

        if self.if_attention:
            attention_output, attention_weight = self.attention_net(output, final_hidden_state, enc_padding_mask)
            # Apply tanh and linear layers
            output = T.tanh(self.linear1(attention_output)) 
            logits = F.softmax(self.linear2(output), dim=1)
        else:
            # No attention (vanilla LSTM): we do max-pooling across sequence dimension
            # max-pooling finds the most important feature across all timesteps
            output = output.permute(0, 2, 1)  
            output = F.max_pool1d(output, output.shape[2]).squeeze(2) 
            # Apply tanh and linear layers
            output = T.tanh(self.linear1(output))
            # Apply softmax to get class probabilities from logits
            logits = F.softmax(self.linear2(output), dim=1)
            attention_weight = 0

        return logits, attention_weight

class CNN_lstm(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_size,
                 kernel_size, bidirection, if_attention):
        super(CNN_lstm, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        # new parameter kernel size: defines CNN filter sizes
        self.kernel_size = kernel_size
        self.num_kernel = len(kernel_size)
        self.bidirection = bidirection
        self.if_attention = if_attention

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        # Conv layers: each kernel size -> one conv
        # Using get_cuda so that the Conv2d is immediately on GPU if available
        KK = []
        for k in kernel_size:
            KK.append(k + 1 if k % 2 == 0 else k)
        
        # nn.Conv2d inputs a sequence of word embeddings
        # CNN treats it like an image, extracting features across local regions
        # 1 channel represents a single text input
        self.conv = nn.ModuleList([
            nn.Conv2d(1, self.embedding_size, (k_, self.embedding_size), stride=1, padding=(k_ // 2, 0))
            for k_ in KK
        ])

        # Learnable initial states
        factor = 2 if self.bidirection else 1
        self.h_0 = nn.Parameter(
            T.rand(factor, self.batch_size, self.hidden_size, device=device),
            requires_grad=True
        )
        self.c_0 = nn.Parameter(
            T.rand(factor, self.batch_size, self.hidden_size, device=device),
            requires_grad=True
        )

        self.lstm = nn.LSTM(
            self.embedding_size * self.num_kernel,
            self.hidden_size,
            batch_first=False,
            bidirectional=self.bidirection
        )
        self.linear1 = nn.Linear(self.hidden_size * factor, (self.hidden_size * factor)//2)
        self.linear2 = nn.Linear((self.hidden_size * factor)//2, self.output_size)
        self.relu = nn.ReLU()

        # Attention
        self.att_wh = nn.Linear(self.hidden_size * factor, self.hidden_size * factor, bias=False)
        self.att_ws = nn.Linear(self.hidden_size * factor, self.hidden_size * factor)
        self.att_v = nn.Linear(self.hidden_size * factor, 1, bias=False)

    def attention_net(self, output_lstm, last_hidden, enc_padding_mask):
        et = self.att_wh(output_lstm)
        hidden = last_hidden.view(last_hidden.shape[1], -1)
        final_fea = self.att_ws(hidden).unsqueeze(1)
        et = et + final_fea
        et = T.tanh(et)
        et = self.att_v(et).squeeze(2)

        et1 = F.softmax(et, dim=1)
        at = et1 * enc_padding_mask
        norm_factor = at.sum(1, keepdim=True)
        at = at / norm_factor

        at = at.unsqueeze(1)
        ct_e = T.bmm(at, output_lstm).squeeze(1)
        at = at.squeeze(1)
        return ct_e, at

    def forward(self, input_sequence, enc_padding_mask):
        
        embedded = self.embedding(input_sequence)          
        embedded = embedded.unsqueeze(1)                   
        
        # Perform 2D conv with multiple kernel sizes
        conved = [self.relu(conv(embedded)).squeeze(-1) for conv in self.conv]

        # concatenate CNN features
        conved = T.cat(conved, 1)                          
        conved = conved.permute(2, 0, 1)                  

        # CNN-reduced sequence is fed into LSTM
        # LSTM captures global dependencies from CNN-extracted features
        output, (final_hidden_state, _) = self.lstm(conved, (self.h_0, self.c_0))
        output = output.permute(1, 0, 2)                   

        if self.if_attention:
            attention_output, attention_weight = self.attention_net(output, final_hidden_state, enc_padding_mask)
            logits = F.softmax(self.linear1(attention_output), dim=1)
        else:
            # No attention: max-pool over seq_length
            output = output.permute(0, 2, 1)
            output = F.max_pool1d(output, output.shape[2]).squeeze(2)
            output = T.tanh(output)
            output = T.tanh(self.linear1(output))
            logits = F.softmax(self.linear2(output), dim=1)
            attention_weight = 0

        return logits, attention_weight


class CNN_BiLSTM_Attn(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_size,
                 kernel_size, if_attention=True):
        super().__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.num_kernel = len(kernel_size)
        self.if_attention = if_attention

        # 1) Embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        # 2) CNN Feature Extractors with multiple kernel sizes
        #    Each kernel size -> 1 Conv2d layer
        #    We store them in an nn.ModuleList so PyTorch can track them.
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=self.embedding_size,
                kernel_size=(k, self.embedding_size),
                stride=1,
                padding=(k // 2, 0)
            )
            for k in kernel_size
        ])

        # Because we want a BiLSTM, factor=2 means LSTM output dimension is hidden_size*2
        factor = 2

        # 3) BiLSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_size * self.num_kernel,  # after CNN cat
            hidden_size=self.hidden_size,
            batch_first=True,  
            bidirectional=True  # BiLSTM
        )

        # 4) Attention Layers
        #    We'll do the same approach as your attentionLSTM or CNN_lstm
        self.att_wh = nn.Linear(self.hidden_size * factor, self.hidden_size * factor, bias=False)
        self.att_ws = nn.Linear(self.hidden_size * factor, self.hidden_size * factor)
        self.att_v = nn.Linear(self.hidden_size * factor, 1, bias=False)

        # 5) Final FC Layers for classification
        self.linear1 = nn.Linear(self.hidden_size * factor, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)

        # Optional: Activation
        self.relu = nn.ReLU()

    def attention_net(self, output_lstm, final_hidden, enc_padding_mask):
        """
        output_lstm: [batch_size, seq_len, hidden_dim*2]
        final_hidden: [2, batch_size, hidden_dim] (because bidirectional)
        enc_padding_mask: [batch_size, seq_len]
        """
        # 1) Project LSTM outputs
        et = self.att_wh(output_lstm)  

        # 2) final_hidden
        hidden_concat = final_hidden.permute(1, 0, 2).reshape(output_lstm.size(0), -1)

        # 3) Apply the second projection
        final_fea = self.att_ws(hidden_concat).unsqueeze(1)  
        et = et + final_fea
        et = T.tanh(et)  # activation

        # 4) Compute alignment scores
        et = self.att_v(et).squeeze(2)

        # 5) Softmax across seq_len, but mask out padded tokens
        alpha = F.softmax(et, dim=1) * enc_padding_mask  # zero out pads
        norm = alpha.sum(dim=1, keepdim=True) + 1e-8
        alpha = alpha / norm

        # 6) Weighted sum of LSTM outputs
        alpha = alpha.unsqueeze(1)
        context = T.bmm(alpha, output_lstm).squeeze(1)

        return context, alpha.squeeze(1)

    def forward(self, input_sequence, enc_padding_mask):
        # 1) Embedding
        embedded = self.embedding(input_sequence)

        # 2) CNN 
        x = embedded.unsqueeze(1)  
        
        # Perform 2D conv for each kernel, then ReLU, then squeeze
        conved = []
        for conv in self.convs:
            c = conv(x)                        
            c = self.relu(c).squeeze(-1)    
            conved.append(c)

        # Concatenate along the "embedding_size * num_kernel" dimension
        out_cnn = T.cat(conved, dim=1) 

        # 3) BiLSTM 
        out_cnn = out_cnn.permute(0, 2, 1)
        # out_cnn 

        output_lstm, (h_n, c_n) = self.lstm(out_cnn) 

        if self.if_attention:
            # 4) Attention => get a context vector
            context, att_weights = self.attention_net(output_lstm, h_n, enc_padding_mask)
            # Then pass context
            out = T.tanh(self.linear1(context)) 
            logits = F.softmax(self.linear2(out), dim=1)
            return logits, att_weights
        else:
            # 4b) No attention => do max pooling across seq_len dimension
            out = output_lstm.permute(0, 2, 1) 
            out = F.max_pool1d(out, kernel_size=out.shape[2]).squeeze(2)
            out = T.tanh(self.linear1(out))
            logits = F.softmax(self.linear2(out), dim=1)
            return logits, None

