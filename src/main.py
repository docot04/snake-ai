import sys
from agent import Agent
from env import SnakeEnv
from model import training, testing

def main():
    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            "  python main.py train <model.pth> [--headless]\n"
            "  python main.py test  <model.pth> <delay> [--headless]"
        )
        sys.exit(1)

    mode = sys.argv[1]
    model_path = sys.argv[2]
    if mode not in ("train", "test"):
        print("Mode must be either 'train' or 'test'.")
        sys.exit(1)

    headless = False
    delay = 0.0

    # TRAIN
    if mode == "train":
        if len(sys.argv) == 4:
            if sys.argv[3] != "--headless":
                print("Unknown argument. Use '--headless'.")
                sys.exit(1)
            headless = True
        elif len(sys.argv) > 4:
            print("Too many arguments for train mode.")
            sys.exit(1)

    # TEST
    elif mode == "test":
        if len(sys.argv) < 4:
            print("Test mode requires a delay value.")
            sys.exit(1)
        try:
            delay = float(sys.argv[3])
            if delay < 0:
                raise ValueError
        except ValueError:
            print("Delay must be a non-negative number.")
            sys.exit(1)
        if len(sys.argv) == 5:
            if sys.argv[4] != "--headless":
                print("Unknown argument. Use '--headless'.")
                sys.exit(1)
            headless = True
        elif len(sys.argv) > 5:
            print("Too many arguments for test mode.")
            sys.exit(1)

    agent = Agent()
    env = SnakeEnv(headless=headless)
    try:
        if mode == "train":
            training(agent, env, model_path)
        elif mode == "test":
            testing(agent, env, model_path, delay)
    finally:
        env.close()


if __name__ == "__main__":
    main()
