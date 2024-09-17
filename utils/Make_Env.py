import gymnasium as gym
from Environment.Preprocess import PreprocessAtari


def make_env(mode):
    env = gym.make('MsPacmanDeterministic-v0', render_mode=mode)
    env = PreprocessAtari(env, height=42, width=42, crop=lambda img: img, dim_order='pytorch', color=False, n_frames=20)
    return env
