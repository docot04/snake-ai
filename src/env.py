import subprocess

# actions chosen by DQN directly
ACTION_STRAIGHT = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2

# control commands used by python environment
ACTION_RESET = 3
ACTION_QUIT = 4

# Python <-> C communication interface
class SnakeEnv:

    # C environment is run as a child process 
    def __init__(self, headless=False):
        command = ["./snake", "interface"]
        if headless:
            command.append("--headless")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        observation = self._read_observation()
        if observation is None:
            raise RuntimeError("Failed to start C-environment")

        # store initial environment state
        self.state, self.reward, self.score, self.done = observation

    # read one observation from C environment
    # format: OK <states[0]...state[10]> <reward float> <score> <done>\\n
    # return: (state, reward, score, done?)
    def _read_observation(self):
        line = self.process.stdout.readline()
        if not line:
            return None
        line = line.strip()
        parts = line.split()
        if parts[0] != "OK":
            return None
        values = parts[1:]
        state = list(map(int, values[0:11]))
        reward = float(values[11])
        score = int(values[12])
        done = bool(int(values[13]))
        return state, reward, score, done

    # send an action to C environment
    # format: <action>\\n
    def _send_action(self, action):
        self.process.stdin.write(f"{action}\n")
        self.process.stdin.flush()

    # reset the game and return to new initial state
    def reset(self):
        self._send_action(ACTION_RESET)
        observation = self._read_observation()
        if observation is None:
            raise RuntimeError("Failed to reset C-environment.")
        self.state, self.reward, self.score, self.done = observation
        return self.state

    # perform one action in C environment
    # actions: 0 (straight) 1 (left), 2 (right)
    # returns: state, reward, done, score
    def step(self, action):
        self._send_action(action)
        observation = self._read_observation()
        if observation is None:
            raise RuntimeError("C-environment stopped responding.")
        state, reward, score, done = observation
        self.state = state
        self.reward = reward
        self.score = score
        self.done = done
        return state, reward, done, score

    # exit C environment
    def close(self):
        try:
            self._send_action(ACTION_QUIT)
        except:
            pass
        try:
            self.process.stdin.close()
        except:
            pass
        self.process.wait()