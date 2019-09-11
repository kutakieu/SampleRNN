from my_model import SampleRNN

# from optim import gradient_clipping
# from nn import sequence_nll_loss_bits

# from dataset import FolderDataset, DataLoader
from audio_dataset import AudioFileDataset
from audio_dataset_test_model import AudioDataset2test_model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import sys
import re
import argparse

import librosa
import numpy as np

import utils

from os import listdir
from os.path import isfile, join

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

default_params = {
    # model parameters
    'n_rnn': 1,
    'dim': 1024,
    'learn_h0': True,
    'q_levels': 256,
    'seq_len': 1024,
    'weight_norm': True,
    'batch_size': 1,
    'val_frac': 0.1,
    'test_frac': 0.1,

    # training parameters
    'keep_old_checkpoints': False,
    'datasets_path': 'datasets',
    'results_path': 'results',
    'epoch_limit': 1000,
    'resume': True,
    'sample_rate': 16000,
    'n_samples': 1,
    'sample_length': 80000,
    'loss_smoothing': 0.99,
    'cuda': False,
    'comet_key': None
}

tag_params = [
    'exp', 'frame_sizes', 'n_rnn', 'dim', 'learn_h0', 'q_levels', 'seq_len',
    'batch_size', 'dataset', 'val_frac', 'test_frac'
]


def main(args):


    frame_sizes = [16,4,4]
    model = None
    sub_batch_length = 512
    finish_flag = False

    model = SampleRNN(frame_sizes = frame_sizes, bs = args.batch_size)
    model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    # dataset = AudioFileDataset("./datasets/debug/")
    dataset = AudioDataset2test_model("./datasets/debug/")

    dataloader_training = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


    """Load wav file here"""
    # y, sr = librosa.load("./data/test.wav")
    # (seq, _) = librosa.load("./data/test.wav", sr=None, mono=True)
    # print(len(seq))

    for epoch in range(args.epochs):
        # loss_history_training.append(0.0)
        # accuracy_history_training.append(0)
        # scheduler.step()
        for i, batch in enumerate(dataloader_training):
            # print(batch.shape)
            # print(type(batch))
            first_zero_array = torch.LongTensor(args.batch_size, model.first_tire.frame_size).fill_(utils.q_zero(model.q_levels))
            quantized_input = torch.LongTensor(utils.quantize_data(batch, model.q_levels))
            input_sequences =  torch.cat((first_zero_array, quantized_input), dim=1)
            input_sequences = input_sequences[:batch.shape[1]-model.first_tire.frame_size]
            target_sequences = input_sequences[:, model.first_tire.frame_size:]
            # print("input_sequences : ", input_sequences.shape)
            # print("target_sequences : ", target_sequences.shape)
            print("batch ", i)
            for j in range(batch.shape[1] // sub_batch_length):
                input_sub_batch = input_sequences[:, sub_batch_length*j : sub_batch_length*(j+1)]
                target_sub_batch = target_sequences[:, sub_batch_length*j : sub_batch_length*(j+1)-model.first_tire.frame_size]
                # print("input_sub_batch : ", input_sub_batch.shape)
                # print("target_sub_batch : ", target_sub_batch.shape)

                logit = model(input_sub_batch).view(-1, model.q_levels)
                prediction = F.log_softmax(logit.view(args.batch_size, -1, model.q_levels), dim=1)

                # print("logit shape : ", logit.shape)
                # print("prediction shape : ", prediction.shape)
                target_sub_batch = target_sub_batch.contiguous().view(-1)
                # print("target_sequences shape : ", target_sub_batch.shape)

                loss = loss_function(logit, target_sub_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(loss)

                if loss.detach().numpy() < 0.39:
                    finish_flag = True
                if finish_flag:
                    break
            if finish_flag:
                break
        if finish_flag:
            break
    j = 12
    generate(input_sequences[:, sub_batch_length*j : sub_batch_length*(j+1)][:1, -frame_sizes[0]*2:], model)


def generate(input_sequence, model):
    print(input_sequence.shape)
    samples = np.zeros((model.first_tire.frame_size * 1000))
    for i in range(1000):
        logit = model(input_sequence).view(-1, model.q_levels)

        # print(logit.shape)

        prediction = np.argmax(logit.view(1, -1, model.q_levels).detach().numpy(), axis=2)
        # print(prediction.shape)
        samples[i*model.first_tire.frame_size : (i+1)*model.first_tire.frame_size] = utils.mu_law_decoding(prediction)[0]
        # print(prediction)
        # print("shapes")
        # print(prediction.shape)
        # print(input_sequence.shape)
        input_sequence = input_sequence[:, -model.first_tire.frame_size:]
        # print(input_sequence.shape)
        # print(model.first_tire.frame_size)

        input_sequence =  torch.cat((input_sequence[-model.first_tire.frame_size:], torch.LongTensor(prediction.reshape(1,-1))), dim=1)
        # print(input_sequence.shape)
        # print("here")

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    print(samples.shape)
    print(np.arange(16000).shape)
    ax.plot(np.arange(16000), samples, color='tab:blue')
    plt.show()

    from scipy.io.wavfile import write
    scaled = np.int16(samples/np.max(np.abs(samples)) * 32767)
    write('test.wav', 16000, scaled)



def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")

    # Data
    parser.add_argument('--batch_size', type=int, default=1, help='Number of instances per batch during training')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--debug', action="store_true", help='if it is in debug mode')

    return parser.parse_args(args)

if __name__ == '__main__':
    main(get_options())

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )

    def parse_bool(arg):
        arg = arg.lower()
        if 'true'.startswith(arg):
            return True
        elif 'false'.startswith(arg):
            return False
        else:
            raise ValueError()

    parser.add_argument('--exp', required=True, help='experiment name')
    parser.add_argument(
        '--frame_sizes', nargs='+', type=int, required=False,
        help='frame sizes in terms of the number of lower tier frames, \
              starting from the lowest RNN tier'
    )
    parser.add_argument(
        '--dataset', required=True,
        help='dataset name - name of a directory in the datasets path \
              (settable by --datasets_path)'
    )
    parser.add_argument(
        '--n_rnn', type=int, help='number of RNN layers in each tier'
    )
    parser.add_argument(
        '--dim', type=int, help='number of neurons in every RNN and MLP layer'
    )
    parser.add_argument(
        '--learn_h0', type=parse_bool,
        help='whether to learn the initial states of RNNs'
    )
    parser.add_argument(
        '--q_levels', type=int,
        help='number of bins in quantization of audio samples'
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='how many samples to include in each truncated BPTT pass'
    )
    parser.add_argument(
        '--weight_norm', type=parse_bool,
        help='whether to use weight normalization'
    )
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument(
        '--val_frac', type=float,
        help='fraction of data to go into the validation set'
    )
    parser.add_argument(
        '--test_frac', type=float,
        help='fraction of data to go into the test set'
    )
    parser.add_argument(
        '--keep_old_checkpoints', type=parse_bool,
        help='whether to keep checkpoints from past epochs'
    )
    parser.add_argument(
        '--datasets_path', help='path to the directory containing datasets'
    )
    parser.add_argument(
        '--results_path', help='path to the directory to save the results to'
    )
    parser.add_argument('--epoch_limit', help='how many epochs to run')
    parser.add_argument(
        '--resume', type=parse_bool, default=True,
        help='whether to resume training from the last checkpoint'
    )
    parser.add_argument(
        '--sample_rate', type=int,
        help='sample rate of the training data and generated sound'
    )
    parser.add_argument(
        '--n_samples', type=int,
        help='number of samples to generate in each epoch'
    )
    parser.add_argument(
        '--sample_length', type=int,
        help='length of each generated sample (in samples)'
    )
    parser.add_argument(
        '--loss_smoothing', type=float,
        help='smoothing parameter of the exponential moving average over \
              training loss, used in the log and in the loss plot'
    )
    parser.add_argument(
        '--cuda', type=parse_bool,
        help='whether to use CUDA'
    )
    parser.add_argument(
        '--comet_key', help='comet.ml API key'
    )

    parser.set_defaults(**default_params)

    main(**vars(parser.parse_args()))
"""
