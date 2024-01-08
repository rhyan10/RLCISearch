from typing import Callable, Dict, Optional, Sequence, Union

import schnetpack as spk
import schnetpack.nn as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from schnetpack import properties
from .spainn import SPAINN

__all__ = ["Atomwise", "Nacs", "Forces", "Dipoles", "SOCs"]


class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    If `aggregation_mode` is None, only the per-atom predictions will be returned.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        aggregation_mode: str = "sum",
        output_key: str = "y",
        per_atom_output_key: Optional[str] = None,
    ):
        super().__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key
        self.n_out = n_out

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.outnet = spk.nn.build_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        self.aggregation_mode = aggregation_mode

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """predict atomwise contributions"""
        y = self.outnet(inputs["scalar_representation"])

        # aggregate
        if self.aggregation_mode is not None:
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            y = snn.scatter_add(y, idx_m, dim_size=maxm)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                for i in range(len(y[0])):
                    y[:, i] = y[:, i] / inputs[properties.n_atoms]

        inputs[self.output_key] = y
        return inputs


class SOCs(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    If `aggregation_mode` is None, only the per-atom predictions will be returned.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        socs_key: str = SPAINN.socs,
    ):
        super().__init__()
        self.socs_key = socs_key
        self.model_outputs = [socs_key]
        self.n_out = n_out

        self.outnet = spk.nn.build_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """predict atomwise contributions"""
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1
        y = self.outnet(inputs["scalar_representation"])
        y = snn.scatter_add(y, idx_m, dim_size=maxm)
        inputs[self.socs_key] = y
        return inputs


class Nacs(nn.Module):
    """
    Predicts NACs
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        nac_key: str = SPAINN.nacs,
        use_vector_repr: bool = False,
    ):
        """
        Args:
            n_in:            input dimension of representation
            n_out:           output dimension (n_out*3)
            n_hidden:        size of hidden layers
            n_layers:        number of layers
            activation:      activation function
            nac_key:         key under which nacs are stored
            use_vector_repr: use equivariant representation
        """
        super().__init__()
        self.nac_key = nac_key
        self.model_outputs = [nac_key]
        self.use_vector_repr = use_vector_repr

        if self.use_vector_repr:
            self.outnet = spk.nn.build_gated_equivariant_mlp(
                n_in=n_in,
                n_out=n_out,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
                sactivation=activation,
            )
        else:
            self.outnet = spk.nn.build_mlp(
                n_in=n_in,
                n_out=n_out,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict NACs
        """
        l0 = inputs["scalar_representation"]

        if self.use_vector_repr:
            l1 = inputs["vector_representation"]
            nacss, nacsv = self.outnet((l0, l1))

            # Multiply scalar part with positions and add vector part
            # Seems to be the best way to include the scalar part and improve predictions
            nacs_painn = torch.einsum('ij,ik->ijk', inputs[properties.R], nacss) + nacsv
            inputs[self.nac_key] = torch.transpose(nacs_painn, 2, 1)

        else:
            virt = self.outnet(l0)
            nacs = spk.nn.derivative_from_molecular(
                virt, inputs["_positions"], True, True
            )[:]
            inputs[self.nac_key] = nacs

        return inputs


class Forces(nn.Module):
    """
    Predicts forces and stress as response of the energy prediction
    w.r.t. the atom positions and strain.

    """

    def __init__(
        self,
        calc_forces: bool = True,
        calc_stress: bool = False,
        energy_key: str = properties.energy,
        force_key: str = properties.forces,
        n_states: int = 1,
    ):
        """
        Args:
            calc_forces: If True, calculate atomic forces.
            calc_stress: If True, calculate the stress tensor.
            energy_key:  Key of the energy in results.
            force_key:   Key of the forces in results.
        """
        super().__init__()
        self.calc_forces = calc_forces
        self.calc_stress = calc_stress
        self.energy_key = energy_key
        self.force_key = force_key
        self.model_outputs = [force_key]
        self.n_states = n_states

        self.required_derivatives = []
        if self.calc_forces:
            self.required_derivatives.append(properties.R)
        if self.calc_stress:
            self.required_derivatives.append(properties.strain)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict forces"""
        E_pred = inputs[self.energy_key]
        grads = spk.nn.derivative_from_molecular(
            E_pred, inputs[properties.R], self.training, True
        )

        inputs[self.force_key] = -grads
        return inputs


class Dipoles(nn.Module):
    """
    Predicts Dipoles
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        dipole_key: str = SPAINN.dipoles,
        return_charges: bool = True,
        charges_key: str = properties.partial_charges,
        use_vector_repr: bool = False,
    ):
        """
        Args:
            n_in:            input dimension of representation
            n_out:           output dimension (n_out*3)
            n_hidden:        size of hidden layers
            n_layers:        number of layers
            activation:      activation function
            dipole_key:      key under which dipoles are stored
            return_charges:  return partial charges
            charges_key:     key under which partial charges are stored
            use_vector_repr: use equivariant representation
        """
        super().__init__()
        self.dipole_key = dipole_key
        self.charges_key = charges_key
        self.return_charges = return_charges
        self.model_outputs = [dipole_key]
        self.use_vector_repr = use_vector_repr
        if self.return_charges:
            self.model_outputs.append(self.charges_key)

        if self.use_vector_repr:
            self.outnet = spk.nn.build_gated_equivariant_mlp(
                n_in=n_in,
                n_out=n_out,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
                sactivation=activation,
            )
        else:
            self.outnet = spk.nn.build_mlp(
                n_in=n_in,
                n_out=n_out,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict Dipoles
        """
        positions = inputs[spk.properties.R]
        l0 = inputs["scalar_representation"]
        natoms = inputs[properties.n_atoms]
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1

        if self.use_vector_repr:
            l1 = inputs["vector_representation"]
            charges, atomic_dipoles = self.outnet((l0, l1))
        else:
            charges = self.outnet(l0)
            atomic_dipoles = 0.0

        if properties.total_charge in inputs:
            total_charge = inputs[properties.total_charge]
            sum_charge = snn.scatter_add(charges, idx_m, dim_size=maxm)
            charge_correction = (total_charge[:, None] - sum_charge) / natoms.unsqueeze(
                -1
            )
            charge_correction = charge_correction[idx_m]
            charges = charges + charge_correction

        # y = (positions * charges) + atomic_dipoles
        y = torch.einsum("ij,ik->ijk", positions, charges) + atomic_dipoles
        y = snn.scatter_add(y, idx_m, dim_size=maxm)
        y = torch.transpose(y, 2, 1)
        inputs[self.dipole_key] = y.reshape(-1, 3)

        if self.return_charges:
            inputs[self.charges_key] = charges

        return inputs
