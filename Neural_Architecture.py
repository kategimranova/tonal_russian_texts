import torch.nn as nn

class LSTM_architecture(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, number_of_layers, drop=0.5):
        super(LSTM_architecture, self).__init__()
        self.output_size = output_size
        self.number_of_layers = number_of_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, number_of_layers,dropout=drop, batch_first=True)
        self.dropout = nn.Dropout(0.45)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden_state):
        lstm_out, hidden_state = self.lstm(self.embedding(x.long()), hidden_state)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sigmoid(out)
        batch_size = x.size(0)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        return sig_out, hidden_state

    def init_hidden_state(self, batch_size):
        weight = next(self.parameters()).data
        hidden_state = (weight.new(self.number_of_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.number_of_layers, batch_size, self.hidden_dim).zero_())
        return hidden_state