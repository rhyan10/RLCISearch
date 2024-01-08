from typing import Tuple
from schnetpack.data import AtomsDataModule
import torch
from .multidatamodule import calculate_multistats

__all__ = ["SPAINN"]


class SPAINN(AtomsDataModule):
    """
    Adapted AtomsDataMosule class to calculating stats
    for multiple states
    """

    energy = "energy"
    forces = "forces"
    nacs = "nacs"
    dipoles = "dipoles"
    socs = "socs"
    smooth_nacs = "smooth_nacs"

    def __init__(self, n_nacs: int = 1, n_states: int = 1, **kwargs):
        self.n_nacs = n_nacs
        self.n_states = n_states
        super().__init__(**kwargs)

    def get_stats(
        self, property: str, divide_by_atoms: bool, remove_atomref: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (property, divide_by_atoms, remove_atomref)
        if key in self._stats:
            return self._stats[key]

        stats = calculate_multistats(
            self.train_dataloader(),
            divide_by_atoms={property: divide_by_atoms},
            atomref=self.train_dataset.atomrefs,
            n_states=self.n_states,
            n_nacs=self.n_nacs,
        )[property]
        self._stats[key] = stats
        return stats
