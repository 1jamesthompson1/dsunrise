from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class DSunriseTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            ensemble,
            feedback_type,
            temperature,
            temperature_act,
            expl_gamma,
            log_dir,
        
            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__()
        self.env = env
        self.ensemble = ensemble
        self.soft_target_tau = soft_target_tau
        
        self.target_update_period = target_update_period
        
        self.feedback_type = feedback_type
        self.temperature = temperature
        self.temperature_act = temperature_act
        self.expl_gamma = expl_gamma
        self.model_dir = log_dir + '/model/'
        
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.alpha_optimizer, self.log_alpha = [], []
            for _ in range(len(self.ensemble)):
                log_alpha = ptu.zeros(1, requires_grad=True)
                alpha_optimizer = optimizer_class(
                    [log_alpha],
                    lr=policy_lr,
                )
                self.alpha_optimizer.append(alpha_optimizer)
                self.log_alpha.append(log_alpha)
                

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss(reduction="none")
        self.vf_criterion = nn.MSELoss(reduction="none")
        
        self.policy_optimizer, self.qf1_optimizer, self.qf2_optimizer, = [], [], []
        
        for en_index in range(len(self.ensemble)):
            policy_optimizer = optimizer_class(
                self.ensemble.get_policies()[en_index].parameters(),
                lr=policy_lr,
            )
            qf1_optimizer = optimizer_class(
                self.ensemble.get_critic1s()[en_index].parameters(),
                lr=qf_lr,
            )
            qf2_optimizer = optimizer_class(
                self.ensemble.get_critic2s()[en_index].parameters(),
                lr=qf_lr,
            )
            self.policy_optimizer.append(policy_optimizer)
            self.qf1_optimizer.append(qf1_optimizer)
            self.qf2_optimizer.append(qf2_optimizer)

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def corrective_feedback(self, obs, update_type):
        std_Q_list = []
        
        if self.feedback_type == 0 or self.feedback_type == 2:
            for en_index in range(len(self.ensemble)):
                with torch.no_grad():
                    policy_action, _, _, _, *_ = self.ensemble.get_policies()[en_index](
                        obs, reparameterize=True, return_log_prob=True,
                    )
                    if update_type == 0:
                        actor_Q1 = self.ensemble.get_critic1s()[en_index](obs, policy_action)
                        actor_Q2 = self.ensemble.get_critic2s()[en_index](obs, policy_action)
                    else:
                        actor_Q1 = self.ensemble.get_target_critic1s()[en_index](obs, policy_action)
                        actor_Q2 = self.ensemble.get_target_critic2s()[en_index](obs, policy_action)
                    mean_actor_Q = 0.5*(actor_Q1 + actor_Q2)
                    var_Q = 0.5*((actor_Q1 - mean_actor_Q)**2 + (actor_Q2 - mean_actor_Q)**2)
                std_Q_list.append(torch.sqrt(var_Q).detach())
                
        elif self.feedback_type == 1 or self.feedback_type == 3:
            mean_Q, var_Q = None, None
            L_target_Q = []
            for en_index in range(len(self.ensemble)):
                with torch.no_grad():
                    policy_action, _, _, _, *_ = self.ensemble.get_policies()[en_index](
                        obs, reparameterize=True, return_log_prob=True,
                    )
                    
                    if update_type == 0: # actor
                        target_Q1 = self.ensemble.get_critic1s()[en_index](obs, policy_action)
                        target_Q2 = self.ensemble.get_critic2s()[en_index](obs, policy_action)
                    else: # critic
                        target_Q1 = self.ensemble.get_target_critic1s()[en_index](obs, policy_action)
                        target_Q2 = self.ensemble.get_target_critic2s() [en_index](obs, policy_action)
                    L_target_Q.append(target_Q1)
                    L_target_Q.append(target_Q2)
                    if en_index == 0:
                        mean_Q = 0.5*(target_Q1 + target_Q2) / len(self.ensemble)
                    else:
                        mean_Q += 0.5*(target_Q1 + target_Q2) / len(self.ensemble)

            temp_count = 0
            for target_Q in L_target_Q:
                if temp_count == 0:
                    var_Q = (target_Q.detach() - mean_Q)**2
                else:
                    var_Q += (target_Q.detach() - mean_Q)**2
                temp_count += 1
            var_Q = var_Q / temp_count
            std_Q_list.append(torch.sqrt(var_Q).detach())

        return std_Q_list
        
    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        masks = batch['masks']
        
        # variables for logging
        tot_qf1_loss, tot_qf2_loss, tot_q1_pred, tot_q2_pred, tot_q_target = 0, 0, 0, 0, 0
        tot_log_pi, tot_policy_mean, tot_policy_log_std, tot_policy_loss = 0, 0, 0, 0
        tot_alpha, tot_alpha_loss = 0, 0
        
        std_Q_actor_list = self.corrective_feedback(obs=obs, update_type=0)
        std_Q_critic_list = self.corrective_feedback(obs=next_obs, update_type=1)
        
        torch.autograd.set_detect_anomaly(True)
        for en_index in range(len(self.ensemble)):
            mask = masks[:,en_index].reshape(-1, 1)

            """
            Policy and Alpha Loss
            """
            new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.ensemble.get_policies()[en_index](
                obs, reparameterize=True, return_log_prob=True,
            )
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha[en_index] * (log_pi + self.target_entropy).detach()) * mask
                alpha_loss = alpha_loss.sum() / (mask.sum() + 1)
                self.alpha_optimizer[en_index].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer[en_index].step()
                alpha = self.log_alpha[en_index].exp()
            else:
                alpha_loss = 0
                alpha = 1

            q_new_actions = torch.min(
                self.ensemble.get_critic1s()[en_index](obs, new_obs_actions),
                self.ensemble.get_critic2s()[en_index](obs, new_obs_actions),
            )
            
            if self.feedback_type == 0 or self.feedback_type == 2:
                std_Q = std_Q_actor_list[en_index]
            else:
                std_Q = std_Q_actor_list[0]
                
            if self.feedback_type == 1 or self.feedback_type == 0:
                weight_actor_Q = torch.sigmoid(-std_Q*self.temperature_act) + 0.5
            else:
                weight_actor_Q = 2*torch.sigmoid(-std_Q*self.temperature_act)
            policy_loss = (alpha*log_pi - q_new_actions - self.expl_gamma * std_Q) * mask * weight_actor_Q.detach()
            policy_loss = policy_loss.sum() / (mask.sum() + 1)

            """
            QF Loss
            """
            q1_pred = self.ensemble.get_critic1s()[en_index](obs, actions)
            q2_pred = self.ensemble.get_critic2s()[en_index](obs, actions)

            # Make sure policy accounts for squashing functions like tanh correctly!
            new_next_actions, _, _, new_log_pi, *_ = self.ensemble.get_policies()[en_index](
                next_obs, reparameterize=True, return_log_prob=True,
            )
            target_q_values = torch.min(
                self.ensemble.get_target_critic1s()[en_index](next_obs, new_next_actions),
                self.ensemble.get_target_critic2s()[en_index](next_obs, new_next_actions),
            ) - alpha * new_log_pi
            
            if self.feedback_type == 0 or self.feedback_type == 2:
                if self.feedback_type == 0:
                    weight_target_Q = torch.sigmoid(-std_Q_critic_list[en_index]*self.temperature) + 0.5
                else:
                    weight_target_Q = 2*torch.sigmoid(-std_Q_critic_list[en_index]*self.temperature)
            else:
                if self.feedback_type == 1:
                    weight_target_Q = torch.sigmoid(-std_Q_critic_list[0]*self.temperature) + 0.5
                else:
                    weight_target_Q = 2*torch.sigmoid(-std_Q_critic_list[0]*self.temperature)
            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
            qf1_loss = self.qf_criterion(q1_pred, q_target.detach()) * mask * (weight_target_Q.detach())
            qf2_loss = self.qf_criterion(q2_pred, q_target.detach()) * mask * (weight_target_Q.detach())
            qf1_loss = qf1_loss.sum() / (mask.sum() + 1)
            qf2_loss = qf2_loss.sum() / (mask.sum() + 1)
            
            """
            Update networks
            """
            self.policy_optimizer[en_index].zero_grad()
            policy_loss.backward()
            self.policy_optimizer[en_index].step()

            self.qf1_optimizer[en_index].zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer[en_index].step()

            self.qf2_optimizer[en_index].zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer[en_index].step()

            """
            Soft Updates
            """
            if self._n_train_steps_total % self.target_update_period == 0:
                ptu.soft_update_from_to(
                    self.ensemble.get_critic1s()[en_index], self.ensemble.get_target_critic1s()[en_index], self.soft_target_tau
                )
                ptu.soft_update_from_to(
                    self.ensemble.get_critic2s()[en_index], self.ensemble.get_target_critic2s()[en_index], self.soft_target_tau
                )

            """
            Statistics for log
            """
            tot_qf1_loss += qf1_loss * (1/len(self.ensemble))
            tot_qf2_loss += qf2_loss * (1/len(self.ensemble))
            tot_q1_pred += q1_pred * (1/len(self.ensemble))
            tot_q2_pred += q2_pred * (1/len(self.ensemble))
            tot_q_target += q_target * (1/len(self.ensemble))
            tot_log_pi += log_pi * (1/len(self.ensemble))
            tot_policy_mean += policy_mean * (1/len(self.ensemble))
            tot_policy_log_std += policy_log_std * (1/len(self.ensemble))
            tot_alpha += alpha.item() * (1/len(self.ensemble))
            tot_alpha_loss += alpha_loss.item()
            tot_policy_loss = (log_pi - q_new_actions).mean() * (1/len(self.ensemble))

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(tot_qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(tot_qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                tot_policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(tot_q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(tot_q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(tot_q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(tot_log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(tot_policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(tot_policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = tot_alpha
                self.eval_statistics['Alpha Loss'] = tot_alpha_loss
                
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def train_single_policy(self, batch, policy_idx):

        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        masks = batch['masks']
        
        mask = masks[:,policy_idx].reshape(-1, 1)
        
        std_Q_actor_list = self.corrective_feedback(obs=obs, update_type=0)

        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.ensemble.get_policies()[policy_idx](
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha[policy_idx] * (log_pi + self.target_entropy).detach())
            alpha_loss = alpha_loss.sum() / (mask.shape[0]+1)
            self.alpha_optimizer[policy_idx].zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer[policy_idx].step()
            alpha = self.log_alpha[policy_idx].exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.ensemble.get_critic1s()[policy_idx](obs, new_obs_actions),
            self.ensemble.get_critic2s()[policy_idx](obs, new_obs_actions),
        )
        
        if self.feedback_type == 0 or self.feedback_type == 2:
            std_Q = std_Q_actor_list[policy_idx]
        else:
            std_Q = std_Q_actor_list[0]
            
        if self.feedback_type == 1 or self.feedback_type == 0:
            weight_actor_Q = torch.sigmoid(-std_Q*self.temperature_act) + 0.5
        else:
            weight_actor_Q = 2*torch.sigmoid(-std_Q*self.temperature_act)
        policy_loss = (alpha*log_pi - q_new_actions - self.expl_gamma * std_Q) * weight_actor_Q.detach()
        policy_loss = policy_loss.sum() / (mask.shape[0]+1)

        self.policy_optimizer[policy_idx].zero_grad()
        policy_loss.backward()
        self.policy_optimizer[policy_idx].step()

    def remove_policy(self, removed_policy):
        """
        When a policy is removed from the ensemble the trainer needs to be updated to remove the corresponding
        elements from the replay buffer and any other relevant data structures.
        """
        
        del self.policy_optimizer[removed_policy]
        del self.qf1_optimizer[removed_policy]
        del self.qf2_optimizer[removed_policy]
        del self.log_alpha[removed_policy]
        del self.alpha_optimizer[removed_policy]
        

    @property
    def networks(self):
        output = []
        for en_index in range(len(self.ensemble)):
            output.append(self.ensemble.get_policies()[en_index])
            output.append(self.ensemble.get_critic1s()[en_index])
            output.append(self.ensemble.get_critic2s()[en_index])
            output.append(self.ensemble.get_target_critic1s()[en_index])
            output.append(self.ensemble.get_target_critic2s()[en_index])
        return output

    def get_snapshot(self):
        return dict(
            policy=self.ensemble.get_policies(),
            qf1=self.ensemble.get_critic1s(),
            qf2=self.ensemble.get_critic2s(),
            target_qf1=self.ensemble.get_target_critic1s(),
            target_qf2=self.ensemble.get_target_critic2s(),
        )
    
    def save_models(self, step):
        for en_index in range(len(self.ensemble)):
            torch.save(
                self.ensemble.get_policies()[en_index].state_dict(), '%s/%d_th_actor_%s.pt' % (self.model_dir, en_index, step)
            )
            torch.save(
                self.ensemble.get_critic1s()[en_index].state_dict(), '%s/%d_th_1st_critic_%s.pt' % (self.model_dir, en_index, step)
            )
            torch.save(
                self.ensemble.get_critic2s()[en_index].state_dict(), '%s/%d_th_2nd_critic_%s.pt' % (self.model_dir, en_index, step)
            )
            torch.save(
                self.ensemble.get_target_critic1s()[en_index].state_dict(), '%s/%d_th_1st_target_critic_%s.pt' % (self.model_dir, en_index, step)
            )
            torch.save(
                self.ensemble.get_target_critic2s()[en_index].state_dict(), '%s/%d_th_2nd_target_critic_%s.pt' % (self.model_dir, en_index, step)
            )

    def load_models(self, step):
        for en_index in range(len(self.ensemble)):
            self.ensemble.get_policies()[en_index].load_state_dict(
                torch.load('%s/%d_th_actor_%s.pt' % (self.model_dir, en_index, step))
            )
            self.ensemble.get_critic1s()[en_index].load_state_dict(
                torch.load('%s/%d_th_1st_critic_%s.pt' % (self.model_dir, en_index, step))
            )
            self.ensemble.get_critic2s()[en_index].load_state_dict(
                torch.load('%s/%d_th_2nd_critic_%s.pt' % (self.model_dir, en_index, step))
            )
            self.ensemble.get_target_critic1s()[en_index].load_state_dict(
                torch.load('%s/%d_th_1st_target_critic_%s.pt' % (self.model_dir, en_index, step))
            )
            self.ensemble.get_target_critic2s()[en_index].load_state_dict(
                torch.load('%s/%d_th_2nd_target_critic_%s.pt' % (self.model_dir, en_index, step))
            )
            
            
    def print_model(self):
        for name, param in self.ensemble.get_policies()[0].named_parameters():
            if param.requires_grad:
                print(name)
                print(param.data)
                break;