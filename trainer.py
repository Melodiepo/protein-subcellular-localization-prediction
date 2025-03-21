# trainer.py
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd

from script.utils import get_cuda, gradient_clamping, data_shuffle, truncate
from script.model_definitions import attentionLSTM, CNN_lstm, TransformerModel, CNN_BiLSTM_Attn
from script.config import classes, max_enc_length  
SEED = 42

class run():
    def __init__(self, batch_size, output_size, hidden_size, vocab_size,
                 embedding_size, num_epoch, lr, gradient_clip, kernel_size=0, vocab=None, test=None):
        """
        test: pass in your 'test' dictionary if you need to print test info
        """
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.num_epoch = num_epoch
        self.lr = lr
        self.gradient_clip = gradient_clip

        self.train_loss = []
        self.train_acc = []
        self.valid_loss = []
        self.valid_acc = []
        self.count_matrix = np.zeros((4, 4))

        self.attention_dict = {}
        self.data_att_dict = {}

        # testing
        self.prediction_score = None
        self.att_test = None
        self.vocab = vocab
        self.test = test

    def setup_model(self, bidirection, if_attention, model_name):
        if model_name == 'CNN-lstm':
            self.model = CNN_lstm(
                self.batch_size, self.output_size, self.hidden_size,
                self.vocab_size, self.embedding_size, self.kernel_size,
                bidirection, if_attention
            )
        elif model_name == 'lstm':
            self.model = attentionLSTM(
                self.batch_size, self.output_size, self.hidden_size,
                self.vocab_size, self.embedding_size,
                bidirection, if_attention
            )
        elif model_name == 'transformer':
            self.model = TransformerModel(
                batch_size=self.batch_size,
                output_size=self.output_size,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                embedding_size=self.embedding_size,
                n_heads=4,
                n_layers=2,
                max_seq_len=max_enc_length,
                dropout=0.1
            )
        elif model_name == 'cnn_bilstm':
            self.model = CNN_BiLSTM_Attn(
                batch_size=self.batch_size,
                output_size=self.output_size,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                embedding_size=self.embedding_size,
                kernel_size=self.kernel_size, 
                if_attention=if_attention
            )
        self.model = get_cuda(self.model)
        print(f"Model is on: {next(self.model.parameters()).device}")
        # use Adam optimiser for training
        self.optimiser = T.optim.Adam(self.model.parameters(), lr=self.lr)

    def save_model(self, save_model_path, epoch):
        save_path = f"{save_model_path}/{epoch+1:05d}.tar"
        T.save({
            'epoch': epoch + 1,
            'model_dict': self.model.state_dict(),
            'optimiser_dict': self.optimiser.state_dict(),
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'valid_loss': self.valid_loss,
            'valid_acc': self.valid_acc
        }, save_path)

    def one_batcher(self, seq, label, length, batch_index):
        """
        Return one batch of data.
        """
        batch_data = seq[batch_index * self.batch_size : (batch_index+1)*self.batch_size]
        batch_label = label[batch_index * self.batch_size : (batch_index+1)*self.batch_size]
        batch_length = length[batch_index * self.batch_size : (batch_index+1)*self.batch_size]

        enc_padding_mask = np.zeros(batch_data.shape, dtype=np.float32)
        for row in range(batch_data.shape[0]):
            enc_padding_mask[row, :batch_length[row]] = 1.0

        return batch_data, batch_label, batch_length, enc_padding_mask

    def train(self, enc_train, enc_train_label, enc_train_length, epoch):
        counter = 0
        total_epoch_loss = 0
        total_epoch_acc = 0

        # Random shuffle each epoch
        train_perm, train_label_perm, train_length_perm = data_shuffle(
            enc_train, enc_train_label, enc_train_length,
            random_seed=SEED
        )

        N = enc_train.shape[0]
        num_batches = N // self.batch_size
        for n in range(num_batches):
            batch_data, batch_label, _, enc_padding_mask = self.one_batcher(
                train_perm, train_label_perm, train_length_perm, batch_index=n
            )
            input_data = get_cuda(T.LongTensor(batch_data))
            input_label = get_cuda(T.LongTensor(batch_label))
            input_mask = get_cuda(T.FloatTensor(enc_padding_mask))

            self.optimiser.zero_grad() # zero the gradients
            prediction_score, _ = self.model(input_data, enc_padding_mask=input_mask) # forward pass through LSTM

            loss = F.cross_entropy(prediction_score, input_label) # compute cross-entropy loss
            # training accuracy
            num_correct = (T.max(prediction_score, 1)[1].view(input_label.size()).data == input_label.data).sum()
            acc = 100.0 * num_correct / len(batch_data)

            loss.backward() # backpropagation
            if self.gradient_clip:
                gradient_clamping(self.model, 1e-1)
            self.optimiser.step() # update weights

            counter += 1
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

        return total_epoch_loss / counter, total_epoch_acc / counter

    def validation(self, enc_valid, enc_valid_label, enc_valid_length, epoch, if_attention):
        total_epoch_loss = 0
        total_epoch_acc = 0
        
        y_true_list = []
        y_pred_list = []
        y_prob_list = []

        # Re-init confusion matrix every 10 epochs
        if epoch % 10 == 0:
            self.count_matrix = np.zeros((4, 4))

        attention_dict = {'0': [], '1': [], '2': [], '3': []}
        data_att_dict = {'0': [], '1': [], '2': [], '3': []}

        enc_valid, enc_valid_label, enc_valid_length = data_shuffle(
            enc_valid, enc_valid_label, enc_valid_length, random_seed=epoch+10
        )

        with T.no_grad():
            N = enc_valid.shape[0]
            num_batches = N // self.batch_size
            for n in range(num_batches):
                batch_data, batch_label, _, enc_padding_mask = self.one_batcher(
                    enc_valid, enc_valid_label, enc_valid_length, batch_index=n
                )
                input_data = get_cuda(T.LongTensor(batch_data))
                input_label = get_cuda(T.LongTensor(batch_label))
                input_mask = get_cuda(T.FloatTensor(enc_padding_mask))

                prediction_valid_score, attention_weight = self.model(input_data, enc_padding_mask=input_mask)
                loss_valid = F.cross_entropy(prediction_valid_score, input_label)
                # validation accuracy
                num_correct_valid = (T.max(prediction_valid_score, 1)[1].view(input_label.size()) == input_label).sum()
                acc_valid = 100.0 * num_correct_valid / len(batch_data)

                if if_attention and (epoch % 10 == 0):
                    pred_label = T.max(prediction_valid_score, 1)[1]
                    for i in range(len(pred_label)):
                        if int(pred_label[i]) == int(input_label[i]):
                            # correct prediction
                            attention_dict[str(int(pred_label[i]))].append(attention_weight[i].unsqueeze(0))
                            data_att_dict[str(int(pred_label[i]))].append(input_data[i, :].unsqueeze(0))

                    self.attention_dict = attention_dict
                    self.data_att_dict = data_att_dict

                # update confusion matrix
                if epoch % 10 == 0:
                    pred_temp = T.max(prediction_valid_score, 1)[1].view(input_label.size()).data
                    for i in range(input_label.shape[0]):
                        self.count_matrix[int(pred_temp[i]), int(input_label[i])] += 1

                y_true_list.append(input_label.cpu().numpy())  # Move to CPU & store
                y_pred_list.append(T.argmax(prediction_valid_score, dim=1).cpu().numpy())  # Predicted labels
                y_prob_list.append(F.softmax(prediction_valid_score, dim=1).cpu().numpy())  # Softmax probabilities
                
                total_epoch_loss += loss_valid
                total_epoch_acc += acc_valid
        
        self.y_true = np.concatenate(y_true_list, axis=0)
        self.y_pred = np.concatenate(y_pred_list, axis=0)
        self.y_prob = np.concatenate(y_prob_list, axis=0)

        return total_epoch_loss / num_batches, total_epoch_acc / num_batches

    def lets_go(self, sentence_lengths, enc_truncated, enc_labels, k_fold, k,
                if_bidirection, if_attention, model_name, save_model_path=None):
        """
        Perform k-fold cross validation using the k-th fold as validation.
        """
        self.setup_model(if_bidirection, if_attention, model_name)

        val_size = len(sentence_lengths) // k_fold
        val_mask = np.arange(val_size * (k-1), val_size * k)  # k-th fold

        enc_valid = enc_truncated[val_mask]
        enc_valid_label = enc_labels[val_mask]
        enc_valid_length = sentence_lengths[val_mask]

        enc_train = np.delete(enc_truncated, val_mask, axis=0)
        enc_train_label = np.delete(enc_labels, val_mask, axis=0)
        enc_train_length = np.delete(sentence_lengths, val_mask, axis=0)

        for epoch in range(self.num_epoch):
            epoch_train_loss, epoch_train_acc = self.train(enc_train, enc_train_label, enc_train_length, epoch)
            epoch_valid_loss, epoch_valid_acc = self.validation(enc_valid, enc_valid_label, enc_valid_length, epoch, if_attention)

            self.train_loss.append(epoch_train_loss)
            self.train_acc.append(epoch_train_acc)
            self.valid_loss.append(epoch_valid_loss)
            self.valid_acc.append(epoch_valid_acc)

            print(f"Epoch:{epoch+1:02}, "
                  f"Train loss: {epoch_train_loss:.3f}, Train acc: {epoch_train_acc:.2f}%, "
                  f"Val loss: {epoch_valid_loss:.3f}, Val acc: {epoch_valid_acc:.2f}%")

            if save_model_path is not None and (epoch+1) % 5 == 0:
                 self.save_model(save_model_path, epoch)

        return self.train_loss, self.train_acc, self.valid_loss, self.valid_acc, self.count_matrix, self.y_true, self.y_pred, self.y_prob

    def train_full_data(self, sentence_lengths, enc_truncated, enc_labels, if_bidirection,
                        if_attention, model_name, save_model_path=None):
        """
        Train on the full dataset (no validation split).
        """
        self.setup_model(if_bidirection, if_attention, model_name)

        enc_train = enc_truncated
        enc_train_label = enc_labels
        enc_train_length = sentence_lengths

        for epoch in range(self.num_epoch):
            epoch_train_loss, epoch_train_acc = self.train(enc_train, enc_train_label, enc_train_length, epoch)
            self.train_loss.append(epoch_train_loss)
            self.train_acc.append(epoch_train_acc)

            print(f"Epoch:{epoch+1:02}, Train loss: {epoch_train_loss:.3f}, Train acc: {epoch_train_acc:.2f}%.")

        return self.train_loss, self.train_acc

    def testing(self, testing_enc_seq, testing_length):
        """
        Perform testing on a small dataset (fewer samples than batch_size).
        """
        N = testing_enc_seq.shape[0]
        batch_test, test_padding_mask = self.test_padding(testing_enc_seq, testing_length)
        assert batch_test.shape == test_padding_mask.shape == (self.batch_size, max_enc_length)

        input_test = get_cuda(T.LongTensor(batch_test))
        input_test_padding_mask = get_cuda(T.FloatTensor(test_padding_mask))

        pred, att_w = self.model(input_test, enc_padding_mask=input_test_padding_mask)
        self.prediction_score = pred[:N]  # discard padded portion
        self.att_test = att_w[:N]
        return self.att_test

    def test_padding(self, input_seq, input_length):
        """
        Pad the testing data up to batch_size if it has fewer samples.
        """
        padded = np.full((self.batch_size, input_seq.shape[1]), self.vocab['<PAD>'], dtype=np.float32)
        padded[: input_seq.shape[0], :] = input_seq

        enc_padding_mask = np.zeros(padded.shape, dtype=np.float32)
        for row in range(len(input_length)):
            enc_padding_mask[row, : input_length[row]] = 1.0
        # For leftover rows, let's just fill a dummy mask of 1.0 or 0.0:
        for row in range(len(input_length), self.batch_size):
            enc_padding_mask[row, 0] = 1.0

        return padded, enc_padding_mask

    def print_frame(self):
        """
        Print the dataframe of predictive probability of each of four classes (for testing).
        """
        if self.prediction_score is None:
            return None

        prob = self.prediction_score.detach().cpu().numpy()
        prob_max = T.max(self.prediction_score, axis=1)[1].cpu().numpy()
        pred_class = np.array([classes[i] for i in prob_max])

        if self.test is not None and 'info' in self.test:
            test_label_frame = [info[1:-1] for info in self.test['info']]
        else:
            test_label_frame = [f"seq_{i}" for i in range(len(prob))]

        df = pd.DataFrame(prob, index=test_label_frame, columns=classes)
        df = df.round(3)
        df['Prediction'] = pd.Series(pred_class, index=df.index)
        return df

    def plot(self, mode='loss'):
        if mode == 'loss':
            plt.plot(self.train_loss, label='training loss')
            plt.plot(self.valid_loss, label='validation loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss plot')
            plt.legend()
            plt.show()
        elif mode == 'acc':
            plt.plot(self.train_acc, label='training acc')
            plt.plot(self.valid_acc, label='validation acc')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy plot')
            plt.legend()
            plt.show()

    def matrix_plot(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        xlabel_list = [' ', 'cyto', ' ', 'secreted', ' ', 'mito', ' ', 'nucleus']
        ylabel_list = [' ', 'cyto', ' ', 'secreted', ' ', 'mito', ' ', 'nucleus']

        img = ax[0].imshow(self.count_matrix, aspect=1, cmap='Blues')
        ax[0].set_xticklabels(xlabel_list)
        ax[0].set_yticklabels(ylabel_list)
        ax[0].set_xlabel('Prediction')
        ax[0].set_ylabel('Ground truth')
        ax[0].set_title('Confusion Matrix')
        for i in range(4):
            for j in range(4):
                c = self.count_matrix[j, i]
                ax[0].text(i, j, str(int(c)), va='center', ha='center')

        row_sum = self.count_matrix.sum(axis=0, keepdims=True)
        error_rate = self.count_matrix / row_sum
        img2 = ax[1].imshow(error_rate, aspect=1, cmap='Oranges')
        ax[1].set_xticklabels(xlabel_list)
        ax[1].set_yticklabels(ylabel_list)
        ax[1].set_xlabel('Prediction')
        ax[1].set_ylabel('Ground truth')
        ax[1].set_title('Error During Validation')

        for i in range(4):
            for j in range(4):
                c = error_rate[j, i]
                ax[1].text(i, j, '{:.2f}%'.format(c * 100), va='center', ha='center')

        fig.colorbar(img2, orientation='horizontal')
        plt.tight_layout()

    def attention(self):
        return self.attention_dict, self.data_att_dict      