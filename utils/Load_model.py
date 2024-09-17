import os


def load_model(agent, models_dir="models", episode=None):
    if episode is not None:
        model_name = f"model_episode_{episode}.pth"
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            print(f"Loading model from episode {episode}")
            agent.load_model(model_path)
            return agent
        else:
            print(f"Model from episode {episode} not found.")
            return None
