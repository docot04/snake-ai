agent=Agent()
env=SnakeEnv()

max_rounds=500
def training():
    for round in range(max_rounds):
        state=env.reset()
        done=False
        total_reward=0

        while not done:
            action=agent.choose_action(state)
            next_state, reward, done=env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            state=next_state
            total_reward+=reward

        agent.update_target()

