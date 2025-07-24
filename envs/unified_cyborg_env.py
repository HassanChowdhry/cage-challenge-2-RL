import random
import inspect
import gym

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from ray.rllib.env.env_context import EnvContext

class UnifiedCybORGEnv(gym.Env):
    scenario_path = str(inspect.getfile(CybORG))[:-10] + "/Shared/Scenarios/Scenario2.yaml"
    max_steps = 100

    def __init__(self, config: EnvContext):
        self.red_choices = [RedMeanderAgent, B_lineAgent, SleepAgent]
        self.cyborg = CybORG(self.scenario_path, environment='sim', agents={'Red': RedMeanderAgent})
        self.env = ChallengeWrapper(env=self.cyborg, agent_name='Blue')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.steps = 0

    def reset(self):
        RedAgentClass = random.choice(self.red_choices)
        self.cyborg = CybORG(self.scenario_path, environment='sim', agents={'Red': RedAgentClass})
        self.env = ChallengeWrapper(env=self.cyborg, agent_name='Blue')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.steps = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        return obs, reward, done, info