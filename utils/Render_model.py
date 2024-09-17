def render_agent_performance(agent, env):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action[0])
        total_reward += reward

    env.close()
    print(f"Total reward: {total_reward}")
