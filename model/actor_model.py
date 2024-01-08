import torch
import torch.nn as nn
from schnetpack.atomistic import PairwiseDistances
import numpy as np
import schnetpack.transform as trn
import schnetpack as spk
from schnetpack.nn.activations import shifted_softplus
from schnetpack.representation import PaiNN

class Actor_Model(nn.Module):
    def __init__(self,
        n_atom_basis: int,
        batch_size: int,
        device: str,
        mol_size: int,
    ):
        super(Actor_Model, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.n_atom_basis = n_atom_basis
        self.n_layers = 6
        self.mol_size = mol_size

        self.theta_com = spk.nn.build_mlp(
            n_in=n_atom_basis,
            n_out=2,
            n_layers=self.n_layers,
            activation=shifted_softplus,
        ).to(torch.float64)

        self.phi_com = spk.nn.build_mlp(
            n_in=n_atom_basis,
            n_out=2,
            n_layers=self.n_layers,
            activation=shifted_softplus,
        ).to(torch.float64)


    def forward(self, repr):

        """
        Forward inputs through output modules and representation.
        """
        shape = repr.size()
        repr = repr.reshape(int(shape[0]/self.mol_size), self.mol_size, shape[-1])

        theta = self.theta_com(repr)
        theta_mu, theta_std = torch.split(theta, split_size_or_sections=1, dim=-1)

        phi = self.phi_com(repr)
        phi_mu, phi_std = torch.split(phi, split_size_or_sections=1, dim=-1)

        return theta_mu, torch.abs(theta_std), phi_mu, torch.abs(phi_std)
