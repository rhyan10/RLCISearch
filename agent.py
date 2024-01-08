import torch
import schnetpack.transform as trn
import torch.nn.functional as F
import argparse
import numpy as np
import random
from model import Critic_Model, Actor_Model
import dill
import schnetpack.transform as trn
import schnetpack as spk
import numpy as np
import random
from schnetpack.interfaces import AtomsConverter as SchNetConverter
from torch.optim import Adam
import logging
from tqdm import tqdm
from env import Env
import torch
import torch.nn.functional as F
from spainn.interface import NacCalculator
import schnetpack.transform as trn
import ase.io
import copy
from utils.gen_start import gen_start
from ase.visualize import view
import time

# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
# logger = logging.getLogger('logger')
# file_handler = logging.FileHandler('./logging/rewards.txt')
# file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
# logger.addHandler(file_handler)

def std_function(x):
    if 0 <= x <= 0.01:
        return 1
    else:
        return 1 / (1 + (x - 0.01) ** 2)


class Actor:
    def __init__(self, args, mol_size):
        self.batch_size = args.batch_size
        self.entropy_beta = 1
        self.max_epochs = args.max_epochs
        self.device = args.device
        self.mol_size = mol_size
        self.model = Actor_Model(args.basis, args.batch_size, args.device, mol_size)
        self.model.to(args.device)
        self.optimizer = Adam(self.model.parameters(), args.lr)
        self.calculators = []
        for i in range(0,4):
            calculator = NacCalculator("spainn_model/ciopt"+str(i+1)+"/best_model", trn.ASENeighborList(cutoff=args.cutoff))
            self.calculators.append(calculator)
        self.converter = SchNetConverter(trn.ASENeighborList(cutoff=10), device=args.device, dtype=torch.float64)
        self.random_model = NacCalculator("spainn_model/ciopt1/best_model", trn.ASENeighborList(cutoff=args.cutoff))

    def get_action(self, inputs):
        results = np.random.choice([0, 1], size = self.batch_size, p=[0, 1])
        num_samples = np.count_nonzero(results == 1)
        energies = []
        for i in range(0,4):
            input_cp = copy.deepcopy(inputs)
            results, input_cp = self.calculators[i].calculate(input_cp)
            energy = results['energy']
            energies.append(energy)

        re_energies = []
        for i in range(len(inputs)):
            mol_energies = []
            for j in range(0,4):
                mol_energies.append(energies[j][i])
            mol_energies = torch.stack(mol_energies).T
            re_energies.append(mol_energies)
        re_energies = torch.stack(re_energies)
        energies = torch.mean(re_energies, axis=-1)
        std = torch.std(re_energies, axis = -1)
        std_sum = torch.sum(std, axis=-1)

        input_cp = copy.deepcopy(inputs)
        results, input_cp = self.random_model.calculate(input_cp)
        theta_mu, theta_std, phi_mu, phi_std = self.model(input_cp['scalar_representation'][:num_samples*self.mol_size])
        theta_act = torch.normal(mean=theta_mu, std=theta_std)
        phi_act = torch.normal(mean=phi_mu, std=phi_std)

        actions = {"theta": theta_act, "phi": phi_act}
        mus = {"theta": theta_mu, "phi": phi_mu}
        stds = {"theta": theta_std, "phi": phi_std}

        s1_energy = torch.roll(energies, 1, 1)
        d1 = torch.abs(s1_energy - energy)
        scaler = d1[:,1]
        return actions, mus, stds, scaler, std_sum, re_energies

    def log_pdf(self, action, mu, std):
        theta = []
        phi = []
        for act in action:
            theta.append(act["theta"])
            phi.append(act['phi'])
        theta = torch.stack(theta).to(self.device)
        phi = torch.stack(phi).to(self.device)

        mu_theta = []
        mu_phi = []
        for m in mu:
            mu_theta.append(m["theta"])
            mu_phi.append(m['phi'])
        mu_theta = torch.squeeze(torch.stack(mu_theta))
        mu_phi = torch.squeeze(torch.stack(mu_phi))

        std_theta = []
        std_phi = []
        for s in std:
            std_theta.append(s["theta"])
            std_phi.append(s["phi"])
        std_theta = torch.squeeze(torch.stack(std_theta))
        std_phi = torch.squeeze(torch.stack(std_phi))

        var_theta = std_theta ** 2
        log_policy_pdf_theta = -0.5 * (theta - mu_theta.unsqueeze(-1)) ** 2 / var_theta.unsqueeze(-1) - 0.5 * torch.log(var_theta.unsqueeze(-1) * 2 * np.pi)

        var_phi = std_phi ** 2
        log_policy_pdf_phi = -0.5 * (phi - mu_phi.unsqueeze(-1)) ** 2 / var_phi.unsqueeze(-1) - 0.5 * torch.log(var_phi.unsqueeze(-1) * 2 * np.pi)

        sum_log_policy = torch.sum(log_policy_pdf_theta, dim=2, keepdim=True) + torch.sum(log_policy_pdf_phi, dim=2, keepdim=True)

        return sum_log_policy

    def loss(self, mu, std, actions, advantages):
        log_policy_pdf = self.log_pdf(actions, mu, std)
        loss_policy = torch.sum(log_policy_pdf, axis=-1) * advantages
        loss = torch.sum(loss_policy)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Critic:
    def __init__(self, args):
        self.model = Critic_Model(args)
        self.model.to(args.device)
        self.optimizer = Adam(self.model.parameters(), args.lr)

    def compute_loss(self, v_pred, td_targets):
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(td_targets, v_pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def loss(self, v_values, td_targets):
        loss = self.compute_loss(v_values, td_targets)
        return loss

class Agent:
    def __init__(self, args):
        self.mol_size = 6
        self.actor = Actor(args, self.mol_size)
        self.critic = Critic(args)
        self.args = args
        self.batch_size = args.batch_size
        self.cutoff = args.cutoff
        self.basis = args.basis
        self.gamma = 1.01
        self.update_interval = 5
        self.episode_length = args.episode_length
        self.max_epochs = args.max_epochs
        self.critic_calculator = NacCalculator("spainn_model/ciopt1/best_model", trn.ASENeighborList(cutoff=args.cutoff))
        self.env = Env(self.batch_size, args.device)

    def n_step_td_target(self, rewards, next_v_value, done, td_targets_shape):
        td_targets = torch.zeros(td_targets_shape).to(torch.float64)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = self.gamma * cumulative + rewards[k].cuda()
            td_targets[k] = cumulative

        return td_targets

    def advantage(self, td_targets, baselines):
        return (td_targets - baselines)

    def tuple_list_to_batch(self, tuples):
        batch = list(tuples[0])
        for elem in tuples[1:]:
            for i, value in enumerate(elem):
                batch[i] = np.append(batch[i], value, axis=0)
        return tuple(batch)

    def list_to_batch(self, list_):
        batch = []
        for elem in list_:
            split = torch.chunk(elem, self.batch_size)
            batch = batch + list(split)
        return batch

    def run(self, db):
        min_dbs = []
        min_rewards = []
        for epoch in tqdm(range(self.max_epochs)):
            reward_batch = []
            v_values = []
            actions = []
            mus = []
            stds = []
            saves_mols = []
            done = False
            counter = 0
            random_numbers = random.sample(range(len(db)), self.batch_size)
            o_mols = gen_start(db, random_numbers)
            mols = copy.deepcopy(o_mols)
            for counter in tqdm(range(self.episode_length)):
                action, mu, std, scaler, std_e, energies = self.actor.get_action(mols)
                _, inputs = self.critic_calculator.calculate(mols)

                if counter == self.episode_length - 1:
                    done = True
                    with open("./logging/actor_model.pkl", "wb") as f:
                        dill.dump(self.actor.model, f)
                    with open("./logging/critic_model.pkl", "wb") as f:
                        dill.dump(self.critic.model, f)
                    with open("./logging/reward"+str(epoch)+".pkl", "wb") as f:
                        dill.dump(reward, f)
                    detached_rew = min_rew.detach().cpu().numpy()
#                    print(energies)
                    print(np.mean(reward.detach().cpu().numpy()))
                    min_rewards.append(detached_rew)
                    ase.io.write("./best_mols/mols"+str(epoch)+".xyz", mols)
                    ase.io.write("logging/save_mols.db", saves_mols)

                v_value = self.critic.model(inputs['scalar_representation'], self.mol_size)
                reward, c_mols = self.env.step(action, scaler, mols, copy.deepcopy(o_mols), std_e)
                reward_batch.append(reward)
                v_values.append(v_value)
                mus.append(mu)
                stds.append(std)
                actions.append(action)
                saves_mols.append(copy.deepcopy(mols[0]))
                min_rew = torch.min(reward)
                min_idx = torch.argmin(reward)

                if len(reward_batch) >= self.update_interval or done == True:

                    with open("./logging/min_rewards.pkl", "wb") as f:
                        dill.dump(min_rewards, f)
#                    ase.io.write("best_mols.db", min_dbs)

                    td_targets_shape = torch.stack(reward_batch, dim=0).shape

                    v_values_batch = torch.stack(v_values, dim=0)

                    td_targets = self.n_step_td_target(
                        reward_batch, v_value, done, td_targets_shape).cuda()

                    advantages = (td_targets - v_values_batch).detach()
                    advantages = advantages.unsqueeze(-1)

                    actor_loss = self.actor.loss(mus, stds, actions, advantages)
                    critic_loss = self.critic.loss(v_values_batch, td_targets.detach())

                    # if done == True:
                    #     with open('./logging/rewards.txt', 'a') as f:
                    #         f.write(str(np.sum(reward.cpu().detach().numpy())) + '\n')

                    reward_batch = []
                    v_values = []
                    mus = []
                    stds = []
                    actions = []
                counter = counter + 1
                mols = c_mols
