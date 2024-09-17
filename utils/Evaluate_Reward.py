def evaluate(agent, env, n_episodes=1):
    episode_reward = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        while True:
            action = agent.act(state)
            state, reward, done, info, _ = env.step(action[0])
            total_reward += reward
            if done:
                break

        episode_reward.append(total_reward)

    return episode_reward
