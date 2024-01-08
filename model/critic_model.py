import torch
import torch.nn as nn
import schnetpack as spk
from schnetpack.nn.activations import shifted_softplus

class Critic_Model(nn.Module):
    def __init__(self, args):
        super(Critic_Model, self).__init__()
        self.n_layers = 8
        self.outnet = spk.nn.build_mlp(
            n_in=args.basis,
            n_out=1,
            n_layers=self.n_layers,
            activation=shifted_softplus,
        ).to(torch.float64)

    def forward(self, repr, mol_size):
        o0 = self.outnet(repr)
        o0 = torch.squeeze(o0)
        o0 = o0.reshape(int(float(o0.shape[0])/mol_size), mol_size)
        o1 = torch.sum(o0, dim=-1)
        return o1
