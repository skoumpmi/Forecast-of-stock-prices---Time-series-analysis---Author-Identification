import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0.).to(device)

    def forward(self, x):

        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x

class CNN2d(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 multi_channel, dropout, pad_idx):
        super().__init__()

        self.__conf = {
            "vocab_size":vocab_size,
            "embedding_dim":embedding_dim,
            "n_filters":n_filters,
            "filter_sizes": filter_sizes,
            "output_dim":output_dim,
            "multi_channel": multi_channel,
            "dropout": dropout,
            "pad_idx":pad_idx
        }

        self.max_filter_size = max(filter_sizes)
        self.multi_channel = multi_channel
        self.noise = GaussianNoise(sigma=0.1)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding.weight.requires_grad = True

        if self.multi_channel:
            self.word_emb_multi = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.word_emb_multi.weight.requires_grad = False
            self.in_channels = 2
        else:
            self.in_channels = 1

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        # p: probability of an element to be zeroed
        self.dropout = nn.Dropout(p=dropout)


    def get_config(self):
        return self.__conf

    def config_json_file(self, file_path):
        with open(file_path, 'w') as json_file:
            json.dump(self.get_config(), json_file)

    def patch_sentence(self, embedded):
        max_len = embedded.size(1)
        if self.max_filter_size > max_len:
            tokens_zeros = Variable(embedded.data.new(embedded.size(0),
                                                        self.max_filter_size - max_len,
                                                        embedded.size(2)))
            embedded = torch.cat([embedded, tokens_zeros], 1)

        return embedded

    def forward(self, text):
        # text = [sent len, batch size]

        text = text.permute(1, 0)

        # text = [batch size, sent len]

        embedded = self.embedding(text)

        #embedded = self.noise(embedded)

        embedded = self.patch_sentence(embedded)

        #embedded = self.dropout(embedded)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        if self.multi_channel:
            embedded_multi = self.word_emb_multi(text)
            embedded_multi = self.patch_sentence(embedded_multi)
            embedded_multi = embedded_multi.unsqueeze(1)
            embedded = torch.cat((embedded, embedded_multi), 1)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

