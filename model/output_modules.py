import torch
import torch.nn as nn
import torch.nn.functional as F


class CartesianCoordinateMu(nn.Module):
    def __init__(self, batch_size, n_images, n_basis):
        super(CartesianCoordinateMu, self).__init__()
        self.n_layers = 6
        self.n_out = 3
        self.batch_size = batch_size
        self.n_images = n_images
        self.n_basis = n_basis
        self.initializer = nn.init.xavier_uniform_
        self.n_neurons = list(range(self.n_basis, self.n_out, int((self.n_out - self.n_basis) / self.n_layers)))
        self.torsion_out_net = nn.ModuleDict()

        for i in range(self.n_layers):
            self.torsion_out_net[str(i)] = nn.Linear(self.n_neurons[i], self.n_neurons[i])

        def forward(self, x):
            for i in range(self.n_layers):
                x = self.torsion_out_net[str(i)](x)
            return x
        
class CartesianCoordinateStd(nn.Module):
    def __init__(self, batch_size, n_images, n_basis):
        super(CartesianCoordinateMu, self).__init__()
        self.n_layers = 6
        self.n_out = 3
        self.batch_size = batch_size
        self.n_images = n_images
        self.n_basis = n_basis
        self.initializer = nn.init.xavier_uniform_
        self.n_neurons = list(range(self.n_basis, self.n_out, int((self.n_out - self.n_basis) / self.n_layers)))
        self.torsion_out_net = nn.ModuleDict()

        for i in range(self.n_layers):
            self.torsion_out_net[str(i)] = nn.Linear(self.n_neurons[i], self.n_neurons[i])

        def forward(self, x):
            for i in range(self.n_layers):
                x = self.torsion_out_net[str(i)](x)
            return x



class BondTorsion(nn.Module):
    def __init__(self, batch_size, n_images, n_basis):
        super(BondTorsion, self).__init__()
        self.n_layers = 6
        self.n_out = 2
        self.batch_size = batch_size
        self.n_images = n_images
        self.n_basis = n_basis
        self.initializer = nn.init.xavier_uniform_
        self.n_neurons = list(range(self.n_basis, self.n_out, int((self.n_out - self.n_basis) / self.n_layers)))

        self.torsion_dense1 = nn.Linear(3*self.n_basis, self.n_basis)
        self.torsion_dense2 = nn.Linear(2*self.n_basis, self.n_basis)
        self.torsion_dense3 = nn.Linear(self.n_basis, self.n_basis)
        self.torsion_out_net = nn.ModuleDict()

        for i in range(self.n_layers):
            self.torsion_out_net[str(i)] = nn.Linear(self.n_neurons[i], self.n_neurons[i])

    def forward(self, x, bond_torsion_ind):
        batch = []
        for i in range(self.batch_size):
            bond_torsion_atoms_i = x[i].index_select(1, bond_torsion_ind[i])
            bond_torsion_atoms_i = bond_torsion_atoms_i.view(self.n_images, -1, self.n_basis * 4)
            batch.append(bond_torsion_atoms_i.unsqueeze(0))
        bond_torsion_atoms = torch.cat(batch, 0)

        t1 = F.softplus(self.torsion_dense1(bond_torsion_atoms))
        t1 = F.softplus(self.torsion_dense2(t1))
        t1 = F.softplus(self.torsion_dense3(t1))

        t2 = t1
        for i in range(self.n_layers):
            t2 = F.softplus(self.torsion_out_net[str(i)](t2))

        t3 = []
        for i in range(self.batch_size):
            g1 = torch.zeros_like(t2[i][0])
            g1 = g1.unsqueeze(0)
            g2 = torch.cat([g1, t2[i][1:-1]], dim=0)
            g3 = torch.zeros_like(t2[i][0])
            g3 = g3.unsqueeze(0)
            g4 = torch.cat([g2, g3], dim=0)
            t3.append(g4.unsqueeze(0))

        t4 = torch.cat(t3, dim=0)

        return t4


class BondAngles(nn.Module):
    def __init__(self, batch_size, n_images, n_basis):
        super(BondAngles, self).__init__()
        self.n_layers = 6
        self.n_out = 2
        self.batch_size = batch_size
        self.n_images = n_images
        self.n_basis = n_basis
        self.initializer = nn.init.xavier_uniform_
        self.n_neurons = list(range(self.n_basis, self.n_out, int((self.n_out - self.n_basis) / self.n_layers)))

        self.angle_dense1 = nn.Linear(3*self.n_basis, self.n_basis)
        self.angle_dense2 = nn.Linear(2*self.n_basis, self.n_basis)
        self.angle_dense3 = nn.Linear(self.n_basis, self.n_basis)
        self.angle_out_net = nn.ModuleDict()

        for i in range(self.n_layers):
            self.angle_out_net[str(i)] = nn.Linear(self.n_neurons[i], self.n_neurons[i])

    def forward(self, x, bond_angle_ind):
        batch = []
        for i in range(self.batch_size):
            bond_angle_atoms_i = x[i].index_select(1, bond_angle_ind[i])
            bond_angle_atoms_i = bond_angle_atoms_i.view(self.n_images, -1, self.n_basis * 3)
            batch.append(bond_angle_atoms_i.unsqueeze(0))
        bond_angle_atoms = torch.cat(batch, 0)

        a1 = F.softplus(self.angle_dense1(bond_angle_atoms))
        a1 = F.softplus(self.angle_dense2(a1))
        a1 = F.softplus(self.angle_dense3(a1))

        a2 = a1
        for i in range(self.n_layers):
            a2 = F.softplus(self.angle_out_net[str(i)](a2))

        a3 = []
        for i in range(self.batch_size):
            g1 = torch.zeros_like(a2[i][0])
            g1 = g1.unsqueeze(0)
            g2 = torch.cat([g1, a2[i][1:-1]], dim=0)
            g3 = torch.zeros_like(a2[i][0])
            g3 = g3.unsqueeze(0)
            g4 = torch.cat([g2, g3], dim=0)
            a3.append(g4.unsqueeze(0))

        a4 = torch.cat(a3, dim=0)

        return a4


class BondLengths(nn.Module):
    def __init__(self, batch_size, n_images, n_basis):
        super(BondLengths, self).__init__()
        self.n_layers = 6
        self.n_out = 2
        self.batch_size = batch_size
        self.n_images = n_images
        self.n_basis = n_basis
        self.initializer = nn.init.xavier_uniform_
        self.n_neurons = list(range(self.n_basis, self.n_out, int((self.n_out - self.n_basis) / self.n_layers)))

        self.length_dense1 = nn.Linear(2*self.n_basis, self.n_basis)
        self.length_dense2 = nn.Linear(2*self.n_basis, self.n_basis)
        self.length_dense3 = nn.Linear(self.n_basis, self.n_basis)
        self.length_out_net = nn.ModuleDict()

        for i in range(self.n_layers):
            self.length_out_net[str(i)] = nn.Linear(self.n_neurons[i], self.n_neurons[i])

    def forward(self, x, bond_len_ind):
        batch = []
        for i in range(self.batch_size):
            bond_length_atoms_i = x[i].index_select(1, bond_len_ind[i])
            bond_length_atoms_i = bond_length_atoms_i.view(self.n_images, -1, self.n_basis * 2)
            batch.append(bond_length_atoms_i.unsqueeze(0))
        bond_length_atoms = torch.cat(batch, 0)

        l1 = F.softplus(self.length_dense1(bond_length_atoms))
        l1 = F.softplus(self.length_dense2(l1))
        l1 = F.softplus(self.length_dense3(l1))

        l2 = l1
        for i in range(self.n_layers):
            l2 = F.softplus(self.length_out_net[str(i)](l2))

        l3 = []
        for i in range(self.batch_size):
            g1 = torch.zeros_like(l2[i][0])
            g1 = g1.unsqueeze(0)
            g2 = torch.cat([g1, l2[i][1:-1]], dim=0)
            g3 = torch.zeros_like(l2[i][0])
            g3 = g3.unsqueeze(0)
            g4 = torch.cat([g2, g3], dim=0)
            l3.append(g4.unsqueeze(0))

        l4 = torch.cat(l3, dim=0)

        return l4
