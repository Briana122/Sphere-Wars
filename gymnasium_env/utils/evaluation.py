# here we will evaluate the agent's performance for each model

def evaluate(env, agent, episodes=100):
    # not complete - just a placeholder
    
    results = {agent.name: 0 for agent in env.agents}
    for episode in range(episodes):
        for agent in env.agents:
            reward = play_game(env, agent)
            results[agent.name] += reward
    return results