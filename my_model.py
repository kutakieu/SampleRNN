from torch import nn
import utils

import torch
from torch.nn import functional as F
from torch.nn import init

import numpy as np

DIM = 512

class SampleRNN(torch.nn.Module):

    def __init__(self, frame_sizes, n_rnn=1, dim=DIM, q_levels=256, bs=1):
        super().__init__()

        self.dim = dim
        self.q_levels = q_levels
        self.batch_size = bs

        self.first_tire = FirstTire(frame_size=frame_sizes[0], frame_size4upsampling=frame_sizes[1])

        self.middle_tire = MiddleTire(frame_size=frame_sizes[1], frame_size4upsampling=frame_sizes[2])

        self.last_tire = LastTire()


    def forward(self, input_sequences):

        """First Tire"""
        to_index = -self.first_tire.frame_size
        prev_samples = 2 * utils.linear_dequantize(
            input_sequences[:,  : to_index],
            self.q_levels
        )
        print("prev_samples.shape : {}".format(prev_samples.shape))
        input4first_tire = prev_samples.contiguous().view(
            self.batch_size, -1, self.first_tire.frame_size
        )
        print("input4first_tire.shape : {}".format(input4first_tire.shape))
        first_tire_output = self.first_tire(input4first_tire)
        print("first_tire_output.shape : {}".format(first_tire_output.shape))

        """Middle Tire"""
        from_index = self.first_tire.frame_size - self.middle_tire.frame_size
        to_index = -self.middle_tire.frame_size
        prev_samples = 2 * utils.linear_dequantize(
            input_sequences[:, from_index : to_index],
            self.q_levels
        )
        input4middle_tire = prev_samples.contiguous().view(
            self.batch_size, -1, self.middle_tire.frame_size
        )
        middle_tire_output = self.middle_tire(input4middle_tire, first_tire_output)

        """Last Tire"""
        # input4last_tire = prev_samples.contiguous().view(
        #     self.batch_size, -1, self.last_tire.frame_size
        # )

        return self.last_tire(input_sequences[:, from_index : to_index], middle_tire_output)

    @property
    def lookback(self):
        return self.frame_level_rnns[-1].n_frame_samples

class FirstTire(torch.nn.Module):
    def __init__(self, frame_size, frame_size4upsampling, n_rnn=1, dim=DIM):
        super().__init__()

        self.frame_size = frame_size
        self.frame_size4upsampling = frame_size4upsampling
        self.rnn_hidden_dim = dim

        self.input_expand = torch.nn.Conv1d(
            in_channels=frame_size,
            out_channels=dim,
            kernel_size=1
        )

        self.rnn = torch.nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=n_rnn,
            batch_first=True
        )
        self.rnn_hideen_state = None

        self.upsampling = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=frame_size4upsampling,
            stride=frame_size4upsampling,
            bias=False
        )


    def forward(self, input):
        x = self.input_expand(input.permute(0, 2, 1)).permute(0, 2, 1)
        print("x.shape : {}".format(x.shape))

        rnn_output, new_hidden_state = self.rnn(x)
        print("rnn_output.shape : {}".format(rnn_output.shape))

        # update the rnn's hidden state
        self.rnn_hideen_state = new_hidden_state

        return self.upsampling(rnn_output.permute(0, 2, 1)).permute(0, 2, 1)




class MiddleTire(torch.nn.Module):
    def __init__(self, frame_size, frame_size4upsampling, n_rnn=1, dim=DIM):
        super().__init__()

        self.frame_size = frame_size
        self.frame_size4upsampling = frame_size4upsampling
        self.rnn_hidden_dim = dim

        self.input_expand = torch.nn.Conv1d(
            in_channels=frame_size,
            out_channels=dim,
            kernel_size=1
        )

        self.rnn = torch.nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=n_rnn,
            batch_first=True
        )

        self.hidden_state = None

        self.upsampling = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=frame_size4upsampling,
            stride=frame_size4upsampling,
            bias=False
        )

    def forward(self, input, input_from_1st_tire):
        input = self.input_expand(input.permute(0, 2, 1)).permute(0, 2, 1)
        print(input.size())
        print(input_from_1st_tire.size())
        x = input + input_from_1st_tire

        rnn_output, new_hidden_state = self.rnn(x)

        # update the rnn's hidden state
        self.rnn_hideen_state = new_hidden_state

        return self.upsampling(rnn_output.permute(0, 2, 1)).permute(0, 2, 1)

class LastTire(torch.nn.Module):
    def __init__(self, frame_size=None, q_levels=256, bs=1, dim=DIM):
        super().__init__()

        self.q_levels = q_levels

        self.embedding = torch.nn.Embedding(
            self.q_levels,
            self.q_levels
        )

        self.MLP_1 = torch.nn.Conv1d(
            in_channels=q_levels,
            out_channels=dim,
            kernel_size=1
        )
        self.MLP_2 = torch.nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1
        )
        self.MLP_3 = torch.nn.Conv1d(
            in_channels=dim,
            out_channels=q_levels,
            kernel_size=1
        )
        self.batch_size = bs


    def forward(self, input, input_from_middle_tire):

        input = self.embedding(
            input.contiguous().view(-1)
        ).view(
            self.batch_size, -1, self.q_levels
        )

        input = input.permute(0, 2, 1)
        input_from_middle_tire = input_from_middle_tire.permute(0, 2, 1)

        x = F.relu(self.MLP_1(input) + input_from_middle_tire)
        x = F.relu(self.MLP_2(x))
        x = self.MLP_3(x).permute(0, 2, 1).contiguous()
        print("x.view(-1, self.q_levels) : {}".format(x.view(-1, self.q_levels).shape))

        return F.log_softmax(x.view(-1, self.q_levels), dim=1).view(self.batch_size, -1, self.q_levels)
