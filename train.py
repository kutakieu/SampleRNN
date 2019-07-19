from my_model import SampleRNN

from optim import gradient_clipping
from nn import sequence_nll_loss_bits

from dataset import FolderDataset, DataLoader

import torch

import os
import sys
import re
import argparse

import librosa

import nn
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
    'batch_size': 128,
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


def main(exp, frame_sizes, dataset, **params):
    params = dict(
        default_params,
        exp=exp, frame_sizes=frame_sizes, dataset=dataset,
        **params
    )


    model = SampleRNN(
        frame_sizes=[16,4,4]
    )

    # def reset_hidden_states(model):
    #     hidden_states = {rnn: None for rnn in model.frame_level_rnns}



    """Load wav file here"""
    # y, sr = librosa.load("./data/test.wav")

    (seq, _) = librosa.load("./data/test.wav", sr=None, mono=True)
    print(len(seq))
    input_sequences =  torch.cat([torch.LongTensor(model.first_tire.frame_size).fill_(utils.q_zero(model.q_levels)),
                                    utils.linear_quantize(torch.from_numpy(seq), model.q_levels)
                                ])
    print("input_sequences shape = " +  str(input_sequences.size()))
    input_sequences = input_sequences.view(1,-1)
    print("input_sequences shape = " +  str(input_sequences.size()))
    input_sequences = input_sequences[:,:-1]
    print("input_sequences shape = " +  str(input_sequences.size()))
    input_sequences = input_sequences[:,:48]
    # input_sequences = torch.from_numpy(y.reshape(1,-1))
    print("input_sequences shape = " +  str(input_sequences.size()))

    print(model(input_sequences).size())

    exit()


    batch_size = 1

    upper_tier_conditioning = None
    hidden_states = {rnn: None for rnn in model.frame_level_rnns}

    for rnn in model.frame_level_rnns:
        print(rnn.frame_size)
        print(rnn.n_frame_samples)

    for i, rnn in enumerate(reversed(model.frame_level_rnns)):

        print(i)
        print(rnn.frame_size)
        print(rnn.n_frame_samples)
        from_index = model.lookback - rnn.n_frame_samples
        to_index = -rnn.n_frame_samples + 1
        prev_samples = 2 * utils.linear_dequantize(
            input_sequences[:, from_index : to_index],
            model.q_levels
        )
        print(model.q_levels)
        print(prev_samples.size())
        prev_samples = prev_samples.contiguous().view(
            batch_size, -1, rnn.n_frame_samples
        )
        print(prev_samples.size())

        def run_rnn(rnn, prev_samples, upper_tier_conditioning):
            (output, new_hidden) = rnn(
                prev_samples, upper_tier_conditioning, hidden_states[rnn]
            )
            hidden_states[rnn] = new_hidden.detach()
            return output

        upper_tier_conditioning = run_rnn(
            rnn, prev_samples, upper_tier_conditioning
        )

    bottom_frame_size = model.frame_level_rnns[0].frame_size
    mlp_input_sequences = input_sequences \
        [:, model.lookback - bottom_frame_size :]

    result = model.sample_level_mlp(
        mlp_input_sequences, upper_tier_conditioning
    )

    print("finish")
    exit()



    predictor = Predictor(model)
    if params['cuda']:
        model = model.cuda()
        predictor = predictor.cuda()

    optimizer = gradient_clipping(torch.optim.Adam(predictor.parameters()))

    data_loader = make_data_loader(model.lookback, params)
    test_split = 1 - params['test_frac']
    val_split = test_split - params['val_frac']




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
        '--frame_sizes', nargs='+', type=int, required=True,
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
