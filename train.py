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

import utils

from os import listdir
from os.path import isfile, join


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

def make_data_loader(overlap_len, params):
    path = os.path.join(params['datasets_path'], params['dataset'])
    def data_loader(split_from, split_to, eval):
        dataset = FolderDataset(
            path, overlap_len, params['q_levels'], split_from, split_to
        )
        return DataLoader(
            dataset,
            batch_size=params['batch_size'],
            seq_len=params['seq_len'],
            overlap_len=overlap_len,
            shuffle=(not eval),
            drop_last=(not eval)
        )
    return data_loader

def prepare_data(data_dir):
    filenames = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    # X =
    for i, filename in enumerate(filenames):
        (seq, _) = librosa.load(data_dir + filename, sr=None, mono=True)
        print(len(seq))
        input_sequences =  torch.cat([
            torch.LongTensor(model.lookback) \
                 .fill_(utils.q_zero(model.q_levels)),
            utils.linear_quantize(
                torch.from_numpy(seq), model.q_levels
            )
        ])


def main(args):
    frame_sizes = [16,4,4]
    model = SampleRNN(
        frame_sizes = frame_sizes, bs = args.batch_size
    )
    sub_batch_length = 512
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    # dataset = AudioFileDataset("./datasets/debug/")
    dataset = AudioDataset2test_model("./datasets/debug/")

    dataloader_training = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


    """Load wav file here"""
    # y, sr = librosa.load("./data/test.wav")

    (seq, _) = librosa.load("./data/test.wav", sr=None, mono=True)
    print(len(seq))

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

                # exit()

    # upper_tier_conditioning = None
    # hidden_states = {rnn: None for rnn in model.frame_level_rnns}
    #
    # for rnn in model.frame_level_rnns:
    #     print(rnn.frame_size)
    #     print(rnn.n_frame_samples)
    #
    # for i, rnn in enumerate(reversed(model.frame_level_rnns)):
    #
    #     print(i)
    #     print(rnn.frame_size)
    #     print(rnn.n_frame_samples)
    #     from_index = model.lookback - rnn.n_frame_samples
    #     to_index = -rnn.n_frame_samples + 1
    #     prev_samples = 2 * utils.linear_dequantize(
    #         input_sequences[:, from_index : to_index],
    #         model.q_levels
    #     )
    #     print(model.q_levels)
    #     print(prev_samples.size())
    #     prev_samples = prev_samples.contiguous().view(
    #         batch_size, -1, rnn.n_frame_samples
    #     )
    #     print(prev_samples.size())
    #
    #     def run_rnn(rnn, prev_samples, upper_tier_conditioning):
    #         (output, new_hidden) = rnn(
    #             prev_samples, upper_tier_conditioning, hidden_states[rnn]
    #         )
    #         hidden_states[rnn] = new_hidden.detach()
    #         return output
    #
    #     upper_tier_conditioning = run_rnn(
    #         rnn, prev_samples, upper_tier_conditioning
    #     )
    #
    # bottom_frame_size = model.frame_level_rnns[0].frame_size
    # mlp_input_sequences = input_sequences \
    #     [:, model.lookback - bottom_frame_size :]
    #
    # result = model.sample_level_mlp(
    #     mlp_input_sequences, upper_tier_conditioning
    # )
    #
    # print("finish")
    # exit()
    #
    #
    #
    # predictor = Predictor(model)
    # if params['cuda']:
    #     model = model.cuda()
    #     predictor = predictor.cuda()
    #
    # optimizer = gradient_clipping(torch.optim.Adam(predictor.parameters()))
    #
    # data_loader = make_data_loader(model.lookback, params)
    # test_split = 1 - params['test_frac']
    # val_split = test_split - params['val_frac']


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")

    # Data
    parser.add_argument('--batch_size', type=int, default=2, help='Number of instances per batch during training')
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
