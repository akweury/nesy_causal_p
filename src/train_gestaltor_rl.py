# Created by jing at 29.12.24

import os
import gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import config
from kandinsky_generator import generate_training_patterns
from utils import args_utils
from src import dataset
from percept import perception


class GestaltGroupingEnv(gymnasium.Env):
    """
    A minimal example environment for RL-based Gestalt principle selection.
    Each episode, we have n_objects scattered randomly.
    The agent picks which "Gestalt principle" to apply at each timestep.
    The environment returns a toy (random) reward and ends after a fixed number of steps.
    """

    def __init__(self, n_principles, max_threshold=1.0, bins=10):
        super(GestaltGroupingEnv, self).__init__()

        self.n_principles = n_principles
        self.max_threshold = max_threshold
        self.bins = bins
        self.data_idx = 0
        # Observation: positions of objects (toy example).
        # Real case: could include color, shape, or feature embeddings.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4, 10),  # x,y coords for each object
            dtype=np.float32
        )

        self.dataset = dataset.GSDataset()
        # Actions (Discrete):
        #  0 = Apply Proximity
        #  1 = Apply Similarity
        #  2 = Apply Closure
        #
        # In a real implementation, you might add more actions
        # or even use a continuous action space for parameterized thresholds.
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Initialize environment state
        self.reset()

    def next_data(self):
        self.data_idx += 1
        task_tensor = self.dataset.load_data(self.data_idx)
        return task_tensor

    def reset(self, seed=123):
        """
        Reset the environment state at the beginning of an episode.
        Returns: initial observation
        """
        # load object positions in [0,1]x[0,1]
        self.ocm = self.next_data().numpy()
        # Could also store "group labels" or other info if needed
        self.steps_taken = 0
        info = {
            "steps_taken": self.steps_taken
        }

        return self.ocm, info

    def step(self, action):
        """
        Apply one of the Gestalt principles (dummy implementation).
        Returns: (observation, reward, done, info)
        """
        # In a real system:
        #   1. You'd group/merge objects based on the selected principle.
        #   2. Compute how good the grouping is (F1 score vs. ground truth, etc.).
        #   3. That becomes your reward.
        # take the action, check if truncated/terminated
        terminated = False
        truncated = False
        principle_float = action[0]
        threshold_float = action[1]

        principle_idx = int(round(principle_float * self.n_principles - 1))
        threshold = threshold_float * self.max_threshold

        # input data
        groups = perception.cluster_by_principle(self.ocm, principle_idx, threshold)
        # For demonstration, we give a random reward:
        reward = perception.percept_reward(groups)

        # For a realistic environment, you'd transform self.positions or maintain
        # a 'current_grouping' data structure. Here, we just keep positions static.

        self.steps_taken += 1
        terminated = (self.steps_taken >= self.max_steps)
        if self.steps_taken < self.max_steps and terminated:
            truncated = True

        # info can hold debugging or diagnostic data
        info = {
            "action_chosen": action
        }

        # Observation stays the same in this toy example
        # (in reality, the grouping or "state" would likely change).
        observation = self.ocm

        return observation, reward, terminated, truncated, info


def init_io_folders(args, model_folder):
    os.makedirs(model_folder, exist_ok=True)


def main():
    # load exp arguments
    args = args_utils.get_args()
    init_io_folders(args, config.model_gestalt)
    generate_training_patterns.genGestaltTraining(args)
    # Initialize the custom environment
    env = GestaltGroupingEnv(n_principles=len(config.gestalt_action))

    # Check the environment (optional but recommended)
    check_env(env, warn=True)

    # Create the PPO model using a simple Multi-Layer Perceptron policy
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=5000)

    # Test the trained model on a few episodes
    test_episodes = 5
    for ep in range(test_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            # Predict action using the trained model
            action, _states = model.predict(obs)
            obs, reward, terminate, truncate, info = env.step(action)
            done = truncate or terminate
            total_reward += reward
        print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}")


if __name__ == "__main__":
    main()
