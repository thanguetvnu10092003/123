from utils.Multiple_Env import EnvBatch
from utils.Evaluate_Reward import evaluate
import numpy as np
import tqdm


class TrainingModel:
    def __init__(self, env, training_epochs, number_of_envs):
        self.env = env
        self.env_batch = EnvBatch(number_of_envs)
        self.training_epochs = training_epochs

    def train(self, agent):
        batch_states = self.env_batch.reset()
        with tqdm.trange(0, self.training_epochs) as progress_bar:
            for i in progress_bar:
                batch_actions = agent.act(batch_states)
                batch_next_states, batch_rewards, batch_dones, _ = self.env_batch.step(batch_actions)
                batch_rewards *= 0.01

                agent.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
                batch_states = batch_next_states

                if i % 1000 == 0:
                    average_reward = np.mean(evaluate(agent, self.env, n_episodes=10))
                    print("Average agent reward: ", average_reward)
                    if average_reward > 400.:
                        model_path = f"models/model_episode_{i}.pth"
                        agent.save_model(model_path)
                        print(f"Model saved to {model_path}")
                    elif i % 100000 == 0 and i > 0:
                        model_path = f"models/model_episode_{i}.pth"
                        agent.save_model(model_path)
                        print(f"Last model save to {model_path}")