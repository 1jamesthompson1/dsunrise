from gymnasium.spaces import Discrete
import torch

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer, EnsembleSimpleReplayBuffer
from rlkit.data_management.simple_replay_buffer import RandomReplayBuffer, GaussianReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np
from collections import deque


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )

class EnsembleEnvReplayBuffer(EnsembleSimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            num_ensemble,
            log_dir,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
            num_ensemble=num_ensemble,
            log_dir=log_dir,
        )

    def add_sample(self, observation, action, reward, terminal, next_observation, mask, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            mask=mask,
            **kwargs
        )

class DynamicEnsembleEnvReplayBuffer(EnsembleEnvReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            num_ensemble,
            log_dir,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            env=env,
            num_ensemble=num_ensemble,
            log_dir=log_dir,
            env_info_sizes=env_info_sizes
        )

        self.policy_rewards = [deque(maxlen=max_replay_buffer_size) for _ in range(num_ensemble)]
    
    def add_sample(self, observation, action, reward, terminal, next_observation, mask, agent_info, **kwargs):
        super().add_sample(observation, action, reward, terminal, next_observation, mask, agent_infos=agent_info, **kwargs)
        self.policy_rewards[agent_info["policy_id"]].append(reward)

    def update_mask(self, actor, mask):
        """
        Update the mask for a particular action. I.e just update a column
        """
        print(f"Mask if of shape {self._mask.shape} and actor {actor} has shape {mask.shape}")

        print(f"Selection mask is of shape: {self._mask[:, actor]}")

        self._mask[:, actor] = mask
    
    def refresh_policy_rewards(self, policy):
        """
        Refresh the rewards for a particular policy. I.e reset the deque
        """
        self.policy_rewards[policy] = deque(maxlen=self._max_replay_buffer_size)
    
    def remove_policy(self, policy_idx):
        """
        Remove a policy from the mask. I.e remove the column from the mask
        """
        self._mask = np.delete(self._mask, policy_idx, axis=1)
        del self.policy_rewards[policy_idx]

    def get_policy_historic_performance(self):
        """
        Get the historic performance of a policy
        """
        return [np.array(policy_rewards) for policy_rewards in self.policy_rewards]

    def save_buffer(self, epoch):
        super().save_buffer(epoch)

        policy_rewards_path = self.buffer_dir + '/policy_rewards_%d.pt' % (epoch)
        payload = [np.array(policy_rewards) for policy_rewards in self.policy_rewards]
        torch.save(payload, policy_rewards_path)
    
    def load_buffer(self, epochs):
        for epoch in epochs:
            try:
                policy_rewards_path = self.buffer_dir + '/policy_rewards_%d.pt' % (epoch)
                payload = torch.load(policy_rewards_path, weights_only=False)
                for i, policy_rewards in enumerate(payload):
                    self.policy_rewards[i] = deque(policy_rewards, maxlen=self._max_replay_buffer_size)
                super().load_buffer(epoch)
            except FileNotFoundError:
                print(f"Policy rewards not found for epoch {epoch}. Skipping loading policy rewards.")
                continue
            print(f"Loaded replay buffer for epoch {epoch} from {policy_rewards_path}")
            break

class RandomEnvReplayBuffer(RandomReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            single_flag,
            equal_flag,
            lower,
            upper,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
            single_flag=single_flag,
            equal_flag=equal_flag,
            lower=lower,
            upper=upper,
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )
    
    
class GaussianEnvReplayBuffer(GaussianReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            prob,
            std,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
            prob=prob,
            std=std,
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )
