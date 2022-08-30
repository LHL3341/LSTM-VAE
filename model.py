import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

def gauss_sampling(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class LSTM_VAE(nn.modules.Module):
    def __init__(self, input_size, hidden_size, latent_size,dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.output_size = input_size
        self.dropout = dropout

        self.rnn1 = nn.LSTM(input_size, hidden_size, batch_first=True,num_layers=1)
        self.rnn2 = nn.LSTM(latent_size, hidden_size, batch_first=True,num_layers=1)

        self.fc1 = nn.Linear(hidden_size, latent_size)
        self.fc2 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(hidden_size, latent_size)
        self.fc4 = nn.Linear(hidden_size, latent_size)
        self.fc = nn.Linear(latent_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, input):
        output, _ = self.rnn1(input.to(torch.float32))
        output = self.dropout(output)
        output = self.norm(output)
        mean1 = self.fc1(output)
        logvar1 = self.fc2(output)
        z = gauss_sampling(mean1, logvar1)
        output, _ = self.rnn2(z)
        output = self.dropout(output)
        mean2 = self.fc3(output)
        tanh = nn.Tanh()
        logvar2 = tanh(self.fc4(output))
        z = gauss_sampling(mean2, logvar2)
        output = self.fc(z)
        return mean1, logvar1, mean2, logvar2, output