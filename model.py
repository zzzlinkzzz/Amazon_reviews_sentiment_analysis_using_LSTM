import torch
import torch.nn as nn
from torch.autograd import Variable

from math import sqrt

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.view(-1, x.size(1))
        gates = self.x2h(x) + self.h2h(hx)
        # gates = gates.squeeze()
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        
        hy = torch.mul(outgate, torch.tanh(cy))
        
        return (hy, cy)


class LSTMnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(LSTMnet, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim # Number of hidden layers
        self.lstm = LSTMCell(input_dim, hidden_dim, layer_dim)  
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.device = device
     
    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)
        outs = []
        cn = c0[0,:,:]
        hn = h0[0,:,:]
        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[:,seq,:], (hn,cn)) 
            outs.append(hn)
        out = outs[-1].squeeze()
        out = self.fc(out) 
        return out
 
# unit test
if __name__ == "__main__":
    input_dim = 300
    hidden_dim = 250
    layer_dim = 1
    output_dim = 2
    batch_size = 100

    # torch.manual_seed(0)

    dummy_img = torch.rand( batch_size, hidden_dim, input_dim).to('cuda')
    print(dummy_img.shape)

    net = LSTMnet(input_dim, hidden_dim, layer_dim, output_dim,'cuda').to('cuda')
    outputs = net(dummy_img)
    print('outputs: ', outputs.data)

    probs = torch.softmax(outputs, dim=1)
    print('probs: ', probs.data)

    predict = torch.argmax(probs,dim=1)
    print('Predicted label: ', predict.cpu().numpy())