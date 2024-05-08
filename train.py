import torch
import torch.nn as nn


class xLSTMCell(nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super(xLSTMCell, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.i2h = nn.Linear(input_dim, 4 * hidden_dim)
    self.h2h = nn.Linear(hidden_dim, 4 * hidden_dim)

  def forward(self, x, hidden):
    h, c = hidden
    gates = self.i2h(x) + self.h2h(hidden)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    c_new = (forgetgate * c) + (ingate * cellgate)
    h_new = outgate * torch.tanh(c_new)

    return h_new, (h_new, c_new)
  

class xLSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers):
    super(xLSTM, self).__init__()
    self.layers = nn.ModuleList([xLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = [(torch.zeros(x.size(0), self.layers[0].hidden_dim, device=x.device),
                       torch.zeros(x.size(0), self.layers[0].hidden_dim, device=x.device)) for _ in self.layers]
        
        h, c = zip(*hidden)
        for i, layer in enumerate(self.layers):
            x, new_hc = layer(x, (h[i], c[i]))
            h[i], c[i] = new_hc
        
        return x, list(zip(h, c))
    

class GPT2WithxLSTM(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, num_layers):
        super(GPT2WithxLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, input_dim)
        self.xlstm = xLSTM(input_dim, hidden_dim, num_layers)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        x, hidden = self.xlstm(x, hidden)
        logits = self.output_layer(x)
        return logits, hidden

