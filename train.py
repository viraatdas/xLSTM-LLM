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
  
