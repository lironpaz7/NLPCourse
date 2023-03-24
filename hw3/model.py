import torch
from torch import nn


class DepParserLA(nn.Module):
    def __init__(self, pos_embedding_dim, hidden_dim, word_vocab_size, tag_vocab_size, word_embedding_dim=None,
                 glove_embedding=None, lstm_layers=5):
        super(DepParserLA, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if glove_embedding is None:
            self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(torch.from_numpy(glove_embedding), freeze=False)
            word_embedding_dim = glove_embedding.shape[1]
        # ------------------------------------------------------------------------------------------------------
        self.embeds_POS = nn.Embedding(tag_vocab_size, pos_embedding_dim)
        lstm_input_size = word_embedding_dim + pos_embedding_dim
        self.encoder = nn.LSTM(input_size=lstm_input_size,
                               hidden_size=hidden_dim,
                               num_layers=lstm_layers,
                               bidirectional=True,
                               batch_first=False)
        linear_input = hidden_dim * 2
        self.head_1 = nn.Linear(linear_input, 150)
        self.mod1 = nn.Linear(linear_input, 150)
        self.activation = nn.Tanh()
        self.fc = nn.Linear(150, 1)

    def forward(self, word_idx_tensor, pos_idx_tensor, length):
        embeds_pos = self.embeds_POS(pos_idx_tensor.view(-1).to(self.device))
        embeds_word = self.word_embedding(word_idx_tensor.view(-1).to(self.device))

        vec = torch.cat((embeds_word, embeds_pos), dim=1)
        lstm, _ = self.encoder(vec.view(vec.shape[0], 1, -1))

        modifiers = self.mod1(lstm.view(length, -1))
        headers = self.head_1(lstm.view(length, -1))

        mlp_output = self.activation(headers.unsqueeze(1) + modifiers)

        return self.fc(mlp_output).reshape((length, length))
