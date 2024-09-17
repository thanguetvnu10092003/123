from utils.Make_Env import make_env
import numpy as np


class EnvBatch:
    def __init__(self, n_envs=10):
        self.envs = [make_env('rgb_array') for _ in range(n_envs)]

    def reset(self):
        _states = []
        for env in self.envs:
            _states.append(env.reset()[0])

        return np.array(_states)

    def step(self, actions):
        next_states, rewards, dones, infos, _ = map(np.array, zip(*[env.step(a) for env, a in zip(self.envs, actions)]))

        for i in range(len(self.envs)):
            if dones[i]:
                next_states[i] = self.envs[i].reset()[0]

        return next_states, rewards, dones, infos
