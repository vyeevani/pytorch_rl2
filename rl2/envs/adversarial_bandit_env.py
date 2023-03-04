"""
Implements the adversarial bandit environment from Laskin et al., 2022
- 'In-context reinforcement learning with Algorithmic Distillation'
"""

from typing import Tuple
from enum import Enum

from rl2.envs.bandit_env import BanditEnv

class AdversarialBanditEnv(BanditEnv):
    def __init__(self, num_actions):
        super().__init__(num_actions)
        self._train = True
    def train(self):
        self._train = True
    def test(self):
        self._train = False
    def _new_payout_probabilities(self):
        """
        Set the payout probabilities to look like:
        Reward is more likely distributed under odd arms 95% of the time during training. 
        At evaluation, the opposite happens - reward appears more often under even arms 95% of the time.

        If training then sample a p_i ~ Uniform[0, 1] * .95 for each odd arm and p_i ~ Uniform[0, 1] * .05 for each even arm
        If testing then sample a p_i ~ Uniform[0, 1] * .95 for each even arm and p_i ~ Uniform[0, 1] * .05 for each odd arm
        """
        if self._train:
            self._payout_probabilities = np.random.uniform(
                low=0.0, high=1.0, size=self._num_actions)
            self._payout_probabilities[::2] *= .95
            self._payout_probabilities[1::2] *= .05
        else:
            self._payout_probabilities = np.random.uniform(
                low=0.0, high=1.0, size=self._num_actions)
            self._payout_probabilities[::2] *= .05
            self._payout_probabilities[1::2] *= .95