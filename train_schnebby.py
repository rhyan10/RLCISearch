import ase.io
import argparse
import logging
import dill as pickle
import warnings
import torch
from tqdm import tqdm
from ase.visualize import view
import numpy as np
from ase import Atoms
import torch
from ase import units
from agent import Agent
import chemcoord as cc
import random
import ase.io
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_model_dir',
                        help='Output directory for model and training log.',
                        default="./weights")
    parser.add_argument('--batch_size', type=int, help='Batch size',
                        default=20)
    parser.add_argument('--cutoff', type=float, help='Distance cutoff',
                        default=20)
    parser.add_argument('--basis', type=int, help='Atomwise dense layer size',
                        default=256)
    parser.add_argument('--max_epochs', type=int, help='Number of steps',
                        default=10000)
    parser.add_argument('--lr', type=float, help='Actor learning rate',
                        default=2e-4)
    parser.add_argument('--device', help='Add device to run model on CPU/GPU',
                        default='cuda')
    parser.add_argument('--mol_file', help='Location of reactant file',
                         default='./data/CH2NH2.db')
    parser.add_argument('--episode_length', help='Length of episode',
                         default=30)
    parser.add_argument('--spainn_model_location', help='Location of PaiNN model',
                         default="./spainn_model/best_inference_model")
    args = parser.parse_args()

    db = ase.io.read(args.mol_file, ":")

    agent = Agent(args)
    logging.info("Training Model")
    agent.run(db)


if __name__ == "__main__":
    main()
