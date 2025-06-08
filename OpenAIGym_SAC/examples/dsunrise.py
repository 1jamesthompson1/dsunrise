import sys
from pathlib import Path
import torch

# Add the parent directory of 'examples' (i.e., OpenAIGym_SAC) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import rlkit.torch.pytorch_util as ptu

from rlkit.data_management.env_replay_buffer import DynamicEnsembleEnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger_custom, set_seed
from rlkit.samplers.data_collector import DynamicEnsembleMdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.dsunrise import DSunriseTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import DynamicTorchBatchRLAlgorithm

import gymnasium as gym
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    # architecture
    parser.add_argument('--num_layer', default=2, type=int)
    
    # train
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--save_freq', default=0, type=int)
    parser.add_argument('--computation_device', default='cpu', type=str)
    parser.add_argument('--epochs', default=1000, type=int)

    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--exp_dir', default='data', type=str)
    parser.add_argument('--exp_name', default='experiment', type=str)
    parser.add_argument('--max_cpu', default=4, type=int)

    # env
    parser.add_argument('--env', default="Ant-v5", type=str)
    
    # ensemble
    parser.add_argument('--num_ensemble', default=10, type=int)
    parser.add_argument('--ber_mean', default=0.5, type=float)
    
    # inference
    parser.add_argument('--inference_type', default=0.0, type=float)
    
    # corrective feedback
    parser.add_argument('--temperature', default=20.0, type=float)

    # Dynamic management
    parser.add_argument('--diversity_threshold', default=0.005, type=float)
    parser.add_argument('--diversity_critical_threshold', default=0.001, type=float)
    parser.add_argument('--performance_gamma', default=0.95, type=float)
    parser.add_argument('--window_size', default=1000, type=float)
    parser.add_argument('--noise', default=0.01, type=float)
    parser.add_argument('--retrain_steps', default=0, type=int)

    parser.add_argument('--removal_check_frequency', default=10000, type=int)
    
    args = parser.parse_args()
    return args
    


class Ensemble:
    def __init__(
            self,
            starting_size,
            obs_dim,
            action_dim,
            network_structure,
            # Hyperparameters for removal and instantiation
            diversity_threshold,
            diversity_critical_threshold,
            performance_gamma,
            window_size,
            noise,
            retrain_steps,
        ):

        self.diversity_threshold = diversity_threshold
        self.diversity_critical_threshold = diversity_critical_threshold
        self.performance_gamma = performance_gamma
        self.window_size = window_size
        self.noise = noise
        self.retrain_steps = retrain_steps

        self.L_qf1, self.L_qf2, self.L_target_qf1, self.L_target_qf2, self.L_policy, self.L_eval_policy = [], [], [], [], [], []

        for idx in range(starting_size):

            qf1 = FlattenMlp(
                input_size=obs_dim + action_dim,
                output_size=1,
                hidden_sizes=network_structure,
            )
            qf2 = FlattenMlp(
                input_size=obs_dim + action_dim,
                output_size=1,
                hidden_sizes=network_structure,
            )
            target_qf1 = FlattenMlp(
                input_size=obs_dim + action_dim,
                output_size=1,
                hidden_sizes=network_structure,
            )
            target_qf2 = FlattenMlp(
                input_size=obs_dim + action_dim,
                output_size=1,
                hidden_sizes=network_structure,
            )
            policy = TanhGaussianPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=network_structure,
                id = idx
            )
            eval_policy = MakeDeterministic(policy)
            
            self.L_qf1.append(qf1)
            self.L_qf2.append(qf2)
            self.L_target_qf1.append(target_qf1)
            self.L_target_qf2.append(target_qf2)
            self.L_policy.append(policy)
            self.L_eval_policy.append(eval_policy)

    def __len__(self):
        return len(self.L_qf1)

    def get_policies(self):
        return self.L_policy

    def get_eval_policies(self):
        return self.L_eval_policy
    
    def get_critic1s(self):
        return self.L_qf1
    
    def get_critic2s(self):
        return self.L_qf2

    def get_target_critic1s(self):
        return self.L_target_qf1
    def get_target_critic2s(self):
        return self.L_target_qf2

    def compute_performance(self, returns) -> float:
        """
        Compute the performance metric for each learner in the ensemble.
        Args:
            returns: np.arrays, of shape (num_episodes,), one per policy
        Returns:
            performance: float, the computed performance metric for the last window_size episodes
        """
        if len(returns) < self.window_size:
            print(f"Not enough returns to compute performance, expected {self.window_size}, got {len(returns)}")
            return 1.0

        weights = np.array([self.performance_gamma ** (self.window_size - 1 - i) for i in range(self.window_size)])
        weights /= weights.sum()
        performance = np.dot(returns[-self.window_size:].squeeze(), weights)
        return performance

    def compute_diversity(self, policy_actions, learner_idx):
        """
        Computes the diversity of the learner_idx-th policy in the ensemble.
        Returns a measure from 0 to 1 indicating how similar the actions of two policies are.

        Args:
            policy_actions: list of np.arrays, each of shape (num_states, action_dim), one per policy
            learner_idx: index of the learner to compute diversity for

        Returns:
            max_diversity: float, maximum diversity (1 - normalized L2 distance) to another policy
        """
        a_i = policy_actions[learner_idx]  # shape: (num_states, action_dim)

        diversities = [
            (np.mean(np.linalg.norm(a_i - a_j, axis=1)) / np.linalg.norm(np.ones_like(a_i)))  # Normalize by max possible L2 distance
            for j, a_j in enumerate(policy_actions)
            if j != learner_idx
        ]

        return np.min(diversities)

    def removal_check(self, samples, returns):
        """
        Check to see if any of the learners are too similar and could be removed.
        """

        policy_actions = [
            policy.get_actions(samples)  # Use batch processing to get actions
            for policy in self.L_eval_policy
        ]

        performances = [self.compute_performance(returns[i]) for i in range(len(self.L_eval_policy))]

        diversities = [self.compute_diversity(policy_actions, i) for i in range(len(self.L_eval_policy))]

        # Find policies with low diversity
        close_policies = [i for i, div in enumerate(diversities) if div < self.diversity_threshold and div == min(diversities)]

        worst_performaning_policy = sorted([(i, perf) for i, perf in enumerate(performances) if i in close_policies], key=lambda x: x[1], reverse=True)

        # Remove the worst performing policy
        if len(worst_performaning_policy) == 0:
            return None, {"performances": performances, "diversities": diversities}
        if len(worst_performaning_policy) > 0:
            return worst_performaning_policy[0][0], {"performances": performances, "diversities": diversities}

    def replace_policy(self, policy_index, samples, train_function, sampler):
        """
        Take a policy and replace it with a new one. If it fails a diversity check it wont be added to the ensemble and instead that sac agent will be removed from the ensemble.

        The replacement of a new policy is a as follows

        Firstly apply gaussian noise to the policy parameters.

        Secondly retrain on a set of transitions for a certain number of steps
        """

        with torch.no_grad():
            for param in self.L_policy[policy_index].parameters():
                noise = torch.randn_like(param) * self.noise
                param.add_(noise)

        for _ in range(self.retrain_steps):
            train_function(sampler(), policy_index)

        
        policy_actions = [
            policy.get_actions(samples)  # Use batch processing to get actions
            for policy in self.L_eval_policy
        ]

        div = self.compute_diversity(policy_actions, policy_index)

        if div < self.diversity_critical_threshold:
            print(f"Policy {policy_index} after mutation and retraining has a diversity of {div} which is below the threshold {self.diversity_critical_threshold}")
            self.remove_policy(policy_index)

            return policy_index, div
        else:
            return None, div

    def remove_policy(self, policy_index):
        """
        Remove the policy at the given index from the ensemble.
        Args:
            policy_index: index of the policy to remove
        """
        print(f"Removing policy {policy_index}")
        del self.L_qf1[policy_index]
        del self.L_qf2[policy_index]
        del self.L_target_qf1[policy_index]
        del self.L_target_qf2[policy_index]
        del self.L_policy[policy_index]
        del self.L_eval_policy[policy_index]

        for i, policy in enumerate(self.L_policy):
            policy.id = i


def experiment(variant):
    expl_env = NormalizedBoxEnv(gym.make(variant['env']))
    eval_env = NormalizedBoxEnv(gym.make(variant['env']))
    obs_dim = expl_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    
    M = variant['layer_size']
    num_layer = variant['num_layer']
    network_structure = [M] * num_layer
    
    ensemble = Ensemble(
        variant['num_ensemble'],
        obs_dim,
        action_dim,
        network_structure,
        variant['diversity_threshold'],
        variant['diversity_critical_threshold'],
        variant['performance_gamma'],
        variant['window_size'],
        variant['noise'],
        variant['retrain_steps']
    )

    eval_path_collector = DynamicEnsembleMdpPathCollector(
        eval_env,
        ensemble,
        eval_flag=True,
    )
    
    expl_path_collector = DynamicEnsembleMdpPathCollector(
        expl_env,
        ensemble,
        ber_mean=variant['ber_mean'],
        eval_flag=False,
        inference_type=variant['inference_type'],
        feedback_type=1,
    )
    
    replay_buffer = DynamicEnsembleEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        len(ensemble),
        log_dir=variant['log_dir'],
    )
    
    trainer = DSunriseTrainer(
        env=eval_env,
        ensemble=ensemble,
        feedback_type=1,
        temperature=variant['temperature'],
        temperature_act=0,
        expl_gamma=0,
        log_dir=variant['log_dir'],
        **variant['trainer_kwargs']
    )
    algorithm = DynamicTorchBatchRLAlgorithm(
        trainer=trainer,
        ensemble=ensemble,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    args = parse_args()
    
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=args.epochs,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=args.batch_size,
            save_frequency=args.save_freq,
            removal_check_frequency=args.removal_check_frequency,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        num_ensemble=args.num_ensemble,
        num_layer=args.num_layer,
        seed=args.seed,
        ber_mean=args.ber_mean,
        env=args.env,
        inference_type=args.inference_type,
        temperature=args.temperature,
        diversity_threshold = args.diversity_threshold,
        diversity_critical_threshold = args.diversity_critical_threshold,
        performance_gamma = args.performance_gamma,
        window_size = args.window_size,
        noise = args.noise,
        retrain_steps = args.retrain_steps,
    )

    torch.set_num_threads(args.max_cpu)

    set_seed(args.seed)
    log_dir = setup_logger_custom(args.exp_name, log_dir=args.exp_dir, variant=variant)

    variant['log_dir'] = log_dir
    if 'cuda' in args.computation_device:
        ptu.set_gpu_mode(True, gpu_id=args.computation_device[0])
    else:
        ptu.set_gpu_mode(False)
    experiment(variant)