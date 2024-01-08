from typing import Dict, Tuple

import schnetpack.properties as structure
import torch
from schnetpack.data import AtomsLoader
from tqdm import tqdm

__all__ = [
    "calculate_multistats",
]


def calculate_multistats(
    dataloader: AtomsLoader,
    divide_by_atoms: Dict[str, bool],
    atomref: Dict[str, torch.Tensor] = None,
    n_nacs: Dict[int, torch.Tensor] = 1,
    n_states: Dict[int, torch.Tensor] = 1,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, int]]:
    property_names = list(divide_by_atoms.keys())
    norm_mask = torch.tensor(
        [float(divide_by_atoms[p]) for p in property_names], dtype=torch.float64
    )
    if property_names[0] == "nacs" or property_names[0] == "smooth_nacs":
        statesvector = torch.ones(n_nacs, 3)
        mean = torch.zeros_like(norm_mask) * statesvector[None, :, :]
        M2 = torch.zeros_like(norm_mask) * statesvector[None, :, :]
    else:
        statesvector = torch.ones(n_states)
        mean = torch.zeros_like(norm_mask) * statesvector[None, :]
        M2 = torch.zeros_like(norm_mask) * statesvector[None, :]

    count = 0

    for props in tqdm(dataloader):
        sample_values = []
        for p in property_names:
            val = props[p][None, :]
            if atomref and p in atomref.keys():
                ar = atomref[p]
                ar = ar[props[structure.Z]]
                idx_m = props[structure.idx_m]
                tmp = torch.zeros((idx_m[-1] + 1,), dtype=ar.dtype, device=ar.device)
                v0 = tmp.index_add(0, idx_m, ar)
                val -= v0

            sample_values.append(val)
        sample_values = torch.cat(sample_values, dim=0)

        batch_size = sample_values.shape[1]
        new_count = count + batch_size

        if property_names[0] == "nacs" or property_names[0] == "smooth_nacs":
            norm = norm_mask[None, :] + (1 - norm_mask[None, :])
            norm = norm[:, :, None, None] * statesvector[None, :, :]

        else:
            norm = norm_mask[:, None] * props[structure.n_atoms] + (
                1 - norm_mask[:, None]
            )
            norm = norm[:, :, None] * statesvector[None, None, :]

        sample_values /= norm
        sample_mean = torch.mean(sample_values, dim=1)

        sample_m2 = torch.sum((sample_values - sample_mean[:, None]) ** 2, dim=1)
        delta = sample_mean - mean
        mean += delta * batch_size / new_count
        corr = batch_size * count / new_count
        M2 += sample_m2 + delta**2 * corr
        count = new_count

    stddev = torch.sqrt(M2 / count)
    # This is now a separate mean and standard deviation for each state.
    stats = {pn: (mu, std) for pn, mu, std in zip(property_names, mean, stddev)}
    print(stats)
    return stats
