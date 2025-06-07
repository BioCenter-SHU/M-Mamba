import torch
from mamba_ssm import Mamba
from RevIN.RevIN import RevIN
from torch.nn.functional import softmax
from torch import nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channels, reduction=5):
        super(SELayer, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, seq_len = x.size()
        y = x.mean(dim=2)  # (batch_size, channels)
        y = self.fc1(y)    # (batch_size, channels // reduction)
        y = self.relu(y)
        y = self.fc2(y)    # (batch_size, channels)
        y = self.sigmoid(y).unsqueeze(2)  # (batch_size, channels, 1)
        return x * y  # (batch_size, channels, seq_len)

class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.resolutions = self.configs.resolutions
        self.threshold = 0.01

        if self.configs.revin == 1:
            self.revin_layer = RevIN(self.configs.enc_in)

        self.lin1 = torch.nn.Linear(configs.seq_len, configs.n1)
        self.dropout1 = torch.nn.Dropout(self.configs.dropout)

        self.lin2 = torch.nn.Linear(configs.n1, configs.pred_len)
        self.dropout2 = torch.nn.Dropout(self.configs.dropout)

        self.raw_weights = nn.Parameter(torch.zeros(len(self.resolutions)))

        self.device = self.configs.gpu
        self.weights_log = {}

        self.linear_layers = []
        for b in self.resolutions:
            num_segments = (self.configs.seq_len - 1) // b
            last_point_index = num_segments * b
            if last_point_index == (self.configs.seq_len - 1):
                linear_layer = torch.nn.Linear(num_segments + 1, self.configs.n1).to(self.device)
            else:
                linear_layer = torch.nn.Linear(num_segments + 2, self.configs.n1).to(self.device)
            self.linear_layers.append(linear_layer)

        self.mamba_list = []
        for i, b in enumerate(self.resolutions):
            Mmamba = Mamba(d_model=self.configs.n1, d_state=self.configs.d_state, d_conv=self.configs.dconv,
                           expand=self.configs.e_fact).to(self.device)
            mamba_name = f"Mmamba_{i}"
            setattr(self, mamba_name, Mmamba)
            self.mamba_list.append(getattr(self, mamba_name))

        self.se_layers = nn.ModuleList([SELayer(channels=self.configs.enc_in) for _ in self.resolutions])

    def channel_dropout(self, x, dropout_rate=0.2):
        if dropout_rate <= 0.0:
            return x
        mask = (torch.rand(x.size(1), device=x.device) > dropout_rate).float()
        mask = mask[None, :, None]

        return x * mask

    def calculate_dropout_rate(self, resolution, weight, k=0.0001, epsilon=1e-6, r_min=0.0, r_max=0.4):
        dropout_rate = k * resolution / (weight + epsilon)
        dropout_rate = max(r_min, min(dropout_rate, r_max))
        # print(f"Resolution: {resolution}, Weight: {weight:.4f}, Dropout Rate: {dropout_rate:.4f}")

        return dropout_rate



    def forward(self, x):
        batch_size = next(iter(x.values())).size(0)
        weighted_forecast_sum = torch.zeros(batch_size, self.configs.pred_len, self.configs.enc_in).to(self.device)

        temp = 0.01
        weights = softmax(self.raw_weights / temp, dim=0)

        for resolution, padded_sequence in x.items():
            forecast = torch.zeros(batch_size, self.configs.pred_len, self.configs.enc_in).to(self.device)

            resolution_index = self.resolutions.index(resolution)

            if self.configs.revin == 1:
                padded_sequence = self.revin_layer(padded_sequence, 'norm')
            else:
                means = padded_sequence.mean(1, keepdim=True).detach()
                padded_sequence = padded_sequence - means
                stdev = torch.sqrt(torch.var(padded_sequence, dim=1, keepdim=True, unbiased=False) + 1e-5)
                padded_sequence /= stdev

            padded_sequence = torch.permute(padded_sequence, (0, 2, 1))

            linear_layer = self.linear_layers[resolution_index]
            padded_sequence = linear_layer(padded_sequence)

            padded_sequence = self.dropout1(padded_sequence)

            Mmamba = self.mamba_list[resolution_index]
            padded_sequence = Mmamba(padded_sequence)

            dropout_rate = self.calculate_dropout_rate(resolution, weights[resolution_index])
            # dropout_rate = 0.4
            padded_sequence = self.channel_dropout(padded_sequence, dropout_rate)

            se_layer = self.se_layers[resolution_index]
            padded_sequence = se_layer(padded_sequence)

            padded_sequence = self.lin2(padded_sequence)
            padded_sequence = self.dropout2(padded_sequence)


            padded_sequence = torch.permute(padded_sequence, (0, 2, 1))
            forecast += padded_sequence

            x = forecast
            if self.configs.revin == 1:
                x = self.revin_layer(x, 'denorm')
            else:
                x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
                x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))

            weight = weights[resolution_index]
            if weight < self.threshold:
                continue

            if resolution not in self.weights_log:
                self.weights_log[resolution] = []
            self.weights_log[resolution].append(weight.item())

            weighted_forecast_sum += weight * x


        return weighted_forecast_sum