import abc
from collections import OrderedDict

from typing import Iterable
from torch import nn as nn
import torch
from functools import partial

from rlkit.core.batch_rl_algorithm import BatchRLAlgorithm
from rlkit.core.batch_normalized_rl_algorithm import BatchNormalRLAlgorithm
from rlkit.core.online_rl_algorithm import OnlineRLAlgorithm
from rlkit.core.trainer import Trainer
from rlkit.torch.core import np_to_pytorch_batch


class TorchOnlineRLAlgorithm(OnlineRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)

class DynamicTorchBatchRLAlgorithm(TorchBatchRLAlgorithm):

    def __init__(self, *args, ensemble, removal_check_frequency, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble = ensemble
        self.removal_check_frequency = removal_check_frequency

    def _end_epoch(self, epoch):

        super()._end_epoch(epoch)

        # Perform removal check

        if self.replay_buffer.num_steps_can_sample() % self.removal_check_frequency == 0:
            sample = self.replay_buffer.random_batch(2000)["observations"]
            returns = self.replay_buffer.get_policy_historic_performance()
            # Get the actions form the polices for all the observations in the sample

            removal_check_results = self.ensemble.removal_check(sample, returns)

            if removal_check_results is not None:
                # Replace the worst performing policy
                removed_policy = self.ensemble.replace_policy(removal_check_results, partial(self.trainer.train_single_policy, batch = np_to_pytorch_batch(self.replay_buffer.random_batch(256*8))))

                if removed_policy is not None:
                    self.replay_buffer.remove_policy(removed_policy)
                    self.trainer.remove_policy(removed_policy)
                    print(f"Removed policy {removed_policy} from the ensemble")
                else:
                    print(f"Policy {removal_check_results} was not removed and was replaced with a mutated and retrained policy")

                    self.replay_buffer.refresh_policy_rewards(removal_check_results)

                    self.replay_buffer.update_mask(removed_policy=removed_policy, mask=torch.bernoulli(torch.tensor([0.5] * self.replay_buffer._max_replay_buffer)))


class TorchBatchNormalRLAlgorithm(BatchNormalRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0

    def train(self, np_batch):
        self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch)

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_steps),
        ])

    @abc.abstractmethod
    def train_from_torch(self, batch):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass
