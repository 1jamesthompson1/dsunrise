from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.networks import FlattenMlp
import torch
import numpy as np

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
                id = idx,
                init_w=1
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