
#from pytorch_transformers  import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from .CNN_txt import CNN2d
from torch import nn
import torch

from transformers.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertEmbeddings

class cnnClassifier(BertPreTrainedModel):

    def __init__(self, config, n_filters, filter_sizes, output_dim,
                 dropout=0.5, multi_channel=False, fine_tunning=False):
        super().__init__(config)
        self.config = config

        self.max_filter_size = max(filter_sizes)

        self.multi_channel = multi_channel

        #self.noise = GaussianNoise(sigma=0.1)

        self.embeddings = BertEmbeddings(config)

        self.cnn_vocab_size  = config.vocab_size
        self.cnn_embedding_dim = config.hidden_size
        self.cnn_output_dim = config.num_labels
        self.cnn_dropout = 0.5

        #self.embedding = nn.Embedding(self.cnn_vocab_size, self.cnn_embedding_dim, padding_idx=pad_idx)
        #self.embeddings.weight.requires_grad = True

        if self.multi_channel:
            self.word_emb_multi = nn.Embedding(self.cnn_vocab_size, self.cnn_embedding_dim, padding_idx=pad_idx)
            self.word_emb_multi.weight.requires_grad = False
            self.in_channels = 2
        else:
            self.in_channels = 1

        if not fine_tunning:
            self.freeze_bert_encoder()


        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels   =   self.in_channels,
                      out_channels  =   n_filters,
                      kernel_size   =   (fs, self.cnn_embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, self.cnn_output_dim)

        # p: probability of an element to be zeroed
        self.dropout = nn.Dropout(p=dropout)

    def patch_sentence(self, embedded):
        max_len = embedded.size(1)
        if self.max_filter_size > max_len:
            tokens_zeros = Variable(embedded.data.new(embedded.size(0),
                                                        self.max_filter_size - max_len,
                                                        embedded.size(2)))
            embedded = torch.cat([embedded, tokens_zeros], 1)

        return embedded

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        # embedded = [batch size, sent len, emb dim]
        embedded = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        # embedded = [batch size, 1, sent len, emb dim]
        embedded = embedded.unsqueeze(1)

        if self.multi_channel:
            embedded_multi = self.word_emb_multi(text)
            embedded_multi = self.patch_sentence(embedded_multi)
            embedded_multi = embedded_multi.unsqueeze(1)
            embedded = torch.cat((embedded, embedded_multi), 1)

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # pooled_n = [batch size, n_filters]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # cat = [batch size, n_filters * len(filter_sizes)]
        cat = self.dropout(torch.cat(pooled, dim=1))

        logits = self.fc(cat)
        output = (logits,)
        return output

    def freeze_bert_encoder(self):
        for param in self.embeddings.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.embeddings.parameters():
            param.requires_grad = True

class linearClassifier(BertForSequenceClassification):

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

