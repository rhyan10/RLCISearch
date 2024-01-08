import os
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import schnetpack.transform as trn
from ase import Atoms
from matplotlib.ticker import ScalarFormatter
from schnetpack import properties
from schnetpack.data import AtomsDataFormat, load_dataset

from .interface import NacCalculator

__all__ = ["PlotMAE"]


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.2f"


class PlotMAE:
    """
    A class to generate scatter plots for trained models
    """

    dot_size = 1.25
    plot_size = 5.0

    def __init__(
        self,
        database: str = None,
        split_file: str = os.path.join(os.getcwd(), "split.npz"),
        model_file: str = os.path.join(os.getcwd(), "best_model"),
        cutoff: float = 10.0,
        properties2plot: List[str] = None,
        subset2plot: List[str] = None,
    ):
        """
        Args:
            database: path to ASE database
            split_file: Path to file containing splitting information of dataset
            model_file: path best inference model from training to get predictions
            cutoff: cutoff distance for radial basis
            properties2plot: Keys for properties to plot
            subset2plot: List of strings for splitted subsets, i.e., 'train', 'test' or 'val', to plot
        """

        self.prop2plot = (
            properties2plot if isinstance(properties2plot, List) else ["energy"]
        )
        self.subset2plot = subset2plot if isinstance(subset2plot, List) else ["train"]
        self.cutoff = cutoff

        if not database:
            raise ValueError("Please specify a path to a database for plotting.")

        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"Model file {model_file} does not exist!")

        # Generate Pytorch dataset for atomistic data from ASE database (type: Dict)
        self.data_module = load_dataset(database, AtomsDataFormat.ASE)
        (
            self.nstates,
            self.nsinglets,
            self.nduplets,
            self.ntriplets,
        ) = self._get_nstates()
        self.coupling_names = self._get_coupling_label()
        self.train_idx, self.val_idx, self.test_idx = self._get_splits(split_file)
        self.calculator = NacCalculator(
            model_file=model_file,
            neighbor_list=trn.MatScipyNeighborList(cutoff=cutoff),
        )

        _2plot_name = ["train", "val", "test"]
        _2plot_idx = [self.train_idx, self.val_idx, self.test_idx]

        self.data_sets = [
            (name, idx)
            for name, idx in zip(_2plot_name, _2plot_idx)
            if name in subset2plot
        ]

    def _get_nstates(self) -> Tuple[int, int, int, int]:
        singlets = self.data_module.metadata.get("n_singlets", 0)
        duplets = self.data_module.metadata.get("n_duplets", 0)
        triplets = self.data_module.metadata.get("n_triplets", 0)

        states = sum([singlets, duplets, triplets])
        assert states > 0, "No states in databse metadata!"

        return states, singlets, duplets, triplets

    def _get_splits(self, split_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train, val and test splits from split file
        """
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Slit file {split_file} does not exist!")
        data = np.load(split_file)

        return data["train_idx"], data["val_idx"], data["test_idx"]

    def mae(
        self,
        pred: Union[List[float], np.ndarray],
        target: Union[List[float], np.ndarray],
    ) -> np.ndarray:
        return np.mean(
            np.abs(np.asarray(pred).flatten() - np.asarray(target).flatten())
        )

    def _get_data4set(
        self, set_idx: np.ndarray, propname: List[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        structure = [self.data_module[int(x)] for x in set_idx]
        atoms = (
            Atoms(
                numbers=self.data_module[0][properties.Z],
                positions=struc[properties.R],
                calculator=self.calculator,
            )
            for struc in structure
        )

        # load all properties
        props = [at.get_properties(self.prop2plot) for at in atoms]

        pred = {
            key: np.stack((val[key] for val in props)).squeeze() for key in props[0]
        }
        ref = {}
        for prop in propname:
            ref[prop] = np.stack(
                (
                    struc[prop if prop != "smooth_nacs" else "nacs"].numpy()
                    for struc in structure
                )
            ).squeeze()

            if pred[prop].shape != ref[prop].shape:
                raise ValueError("Target and Prediction have incompatible shape!")

        return ref, pred

    def _get_num2plot(self, ref, pred, propname) -> int:
        assert (
            ref[propname].shape == pred[propname].shape
        ), "Reference and predicted data have different shape."

        if len(ref[propname].shape) == 2:
            return ref[propname].shape[1]
        if len(ref[propname].shape) == 4:
            return ref[propname].shape[2]

        raise ValueError("Invalid shape for property.")

    def _get_label(self, index: int, electronic_state: bool = True) -> str:
        """
        Get the label for a given state index, based on the metadata in the given data module.
        If the data module does not have any state metadata, the label is generated as 'S' + the state index.

        Args:
            index (int): The index of the state to get the label for.
            electronic_state: If True property is for single electronic states, else between electronic states.

        Returns:
            str: The label for the given state index.
        """

        state_metadata = self.data_module.metadata.get("states", None)

        if electronic_state:
            if state_metadata:
                labels = state_metadata.split()
                real_idx = labels.index(labels[index])
                if labels[index] != "S":
                    real_idx += 1
                label = str(labels[index]) + "$_" + str(index - real_idx) + "$"
            else:
                label = "S$_" + str(index) + "$"

        else:
            label = str(self.coupling_names[index])

        return label

    def _get_coupling_label(self) -> List[str]:
        labels = []

        for nstates, statelabel in zip(
            [self.nsinglets, self.nduplets, self.ntriplets], ["S", "D", "T"]
        ):
            for i in range(nstates):
                for j in range(i + 1, nstates):
                    labels.append(str(statelabel) + "$_{" + str(i) + str(j) + "}$")
        return labels

    def plot(self) -> None:
        num_rows = len(self.prop2plot)
        num_cols = len(self.subset2plot)

        units = self.data_module.units

        plt.rcParams["figure.figsize"] = [
            PlotMAE.plot_size * num_cols,
            PlotMAE.plot_size * num_rows,
        ]
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["font.size"] = 14

        # Create the grid of subplots using the calculated number of rows and columns
        if num_rows == 1 and num_cols == 1:
            ax = plt.gca()
        else:
            _, axs = plt.subplots(num_rows, num_cols)

        # Iterate over each row and column of the grid to plot your data
        for c, (_, set_idx) in enumerate(self.data_sets):
            ref, pred = self._get_data4set(set_idx=set_idx, propname=self.prop2plot)

            for r, propname in enumerate(self.prop2plot):
                # Select the current subplot to plot on
                if num_cols == 1 and num_rows > 1:
                    ax = axs[r]
                elif num_rows == 1 and num_cols > 1:
                    ax = axs[c]
                elif num_rows > 1 and num_cols > 1:
                    ax = axs[r, c]

                num2plot = self._get_num2plot(ref=ref, pred=pred, propname=propname)

                if propname == "energy":
                    unit = units["energy"] if "energy" in units.keys() else "Ha"
                    for state in range(num2plot):
                        label = self._get_label(index=state, electronic_state=True)
                        ax.scatter(
                            ref[propname][:, state],
                            pred[propname][:, state],
                            label=label,
                            s=PlotMAE.dot_size,
                        )

                    yScalarFormatter = ScalarFormatterClass(useMathText=True)
                    yScalarFormatter.set_powerlimits((0, 3))
                    ax.yaxis.set_major_formatter(yScalarFormatter)
                    ax.xaxis.set_major_formatter(yScalarFormatter)

                    ax.set_xlabel("rel. E$_{ref}$ / " + str(unit))
                    if c == 0:
                        ax.set_ylabel("rel. E$_{pred}$ / " + str(unit))

                elif propname in ("forces", "nacs", "smooth_nacs", "dipoles"):
                    bunit = "Ha/Bohr" if propname == "forces" else "1/Bohr"
                    unit = units[propname] if propname in units.keys() else bunit

                    # Funktioniert nicht für NACs weil states != couplings
                    for state in range(num2plot):
                        if propname == "forces":
                            label = self._get_label(index=state, electronic_state=True)
                        else:
                            label = self._get_label(index=state, electronic_state=False)

                        ax.scatter(
                            ref[propname][:, :, state],
                            pred[propname][:, :, state],
                            label=label,
                            s=PlotMAE.dot_size,
                        )

                    yScalarFormatter = ScalarFormatterClass(useMathText=True)
                    yScalarFormatter.set_powerlimits((0, 3))
                    ax.yaxis.set_major_formatter(yScalarFormatter)
                    ax.xaxis.set_major_formatter(yScalarFormatter)

                    ax.set_xlabel("rel. " + str(propname) + "$_{ref}$ / " + str(unit))
                    if c == 0:
                        ax.set_ylabel(
                            "rel. " + str(propname) + "$_{pred}$ / " + str(unit)
                        )

                ax.set_title(
                    "Property: "
                    + str(propname)
                    + ",\nDataset: "
                    + str(self.subset2plot[c])
                )
                ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
                ax.legend()

        plt.show()
