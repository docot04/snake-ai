import time
import sys
from agent import Agent
from env import SnakeEnv

MAX_TRAIN_EPISODES = 500
MAX_TEST_EPISODES = 10
TARGET_UPDATE_FREQUENCY=10

# training the DQN
def training(agent, env, model_path):
    for episode in range(1, MAX_TRAIN_EPISODES+1):
        state=env.reset()
        done=False
        total_reward=0

        while not done:
            action=agent.choose_action(state)
            next_state, reward, done, score=env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            state=next_state
            total_reward+=reward

        # decay epsilon once per episode
        agent.decay_epsilon()

        # update target network every 10 rounds
        if episode % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target()

        print(
            f"Episode: {episode:4d}, "
            f"Score: {score:3d}, "
            f"Reward: {total_reward:8.2f}, "
            f"Epsilon: {agent.epsilon:.3f}")    

    # save trained model
    agent.save(model_path)
    print(f"\nModel saved to: {model_path}")

# testing the DQN
def testing(agent, env, model_path):

    agent.load(model_path)

    # disable exploration to always choose best action
    agent.epsilon = 0.0
    scores = []

    try:
        for episode in range(1, MAX_TEST_EPISODES + 1):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, score = env.step(action)
                state = next_state
                total_reward += reward
                steps += 1

                # to slow down the visuals
                # time.sleep(0.1) 
            scores.append(score)

            print(
                f"Episode: {episode:4d}, "
                f"Score: {score:3d}, "
                f"Reward: {total_reward:8.2f}, "
                f"Steps: {steps:5d}")   

    except KeyboardInterrupt:
        print("\nTesting interrupted.")

    finally:
        env.close()

    if scores:
        average_score = sum(scores) / len(scores)
        print("\nTEST RESULTS")
        print(f"Episodes:       {len(scores)}")
        print(f"Average score:  {average_score:.2f}")
        print(f"Best score:     {max(scores)}")
        print(f"Worst score:    {min(scores)}")


def main():
    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "  python model.py train <model.pth>\n"
            "  python model.py test  <model.pth>"
        )
        sys.exit(1)

    mode = sys.argv[1]
    model_path = sys.argv[2]
    if mode not in ("train", "test"):
        print("Mode must be either 'train' or 'test'.")
        sys.exit(1)

    agent = Agent()
    env = SnakeEnv()
    try:
        if mode == "train":
            training(agent, env, model_path)
        elif mode == "test":
            testing(agent, env, model_path)
    finally:
        env.close()


if __name__ == "__main__":
    main()
