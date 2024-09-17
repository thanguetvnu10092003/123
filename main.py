from utils.Training import TrainingModel
from utils.Make_Env import make_env
from Network.Agent import Agent
from utils.Render_model import render_agent_performance
from utils.Load_model import load_model
import os

if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)

    env = make_env('rgb_array')
    state_shape = env.observation_space.shape
    number_actions = env.action_space.n
    print('Observation shape: ', state_shape)
    print('Number of actions: ', number_actions)
    print('Actions name:', env.unwrapped.get_action_meanings())

    agent = Agent(number_actions)

    # training_model = TrainingModel(env, training_epochs=100001, number_of_envs=10)
    # training_model.train(agent)

    loaded_agent = load_model(agent, episode=99000)
    render_env = make_env('human')
    render_agent_performance(loaded_agent, render_env)
