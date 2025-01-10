# Created by jing at 29.12.24

import os
import gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch

import config
from kandinsky_generator import generate_training_patterns
from utils import args_utils, chart_utils
from utils.chart_utils import van
from src import dataset
from percept import perception
from alpha import alpha
import eval_nsfr


class GestaltGroupingEnv(gymnasium.Env):
    """
    A minimal example environment for RL-based Gestalt principle selection.
    Each episode, we have n_objects scattered randomly.
    The agent picks which "Gestalt principle" to apply at each timestep.
    The environment returns a toy (random) reward and ends after a fixed number of steps.
    """

    def __init__(self, args, n_principles, obj_n, max_threshold=1.0, bins=10):
        super(GestaltGroupingEnv, self).__init__()
        self.args = args
        self.n_principles = n_principles
        self.max_threshold = max_threshold
        self.bins = bins
        # self.labels = None
        self.data_idx = 0
        self.max_steps = 3
        self.max_group_id = -1
        self.ocm = None
        self.imgs = None
        self.total_step = 0
        # Observation: positions of objects (toy example).
        # Real case: could include color, shape, or feature embeddings.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, obj_n, 10),  # x,y coords for each object
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
        self.data_idx = self.data_idx % self.dataset.__len__()
        task_tensor_pos, task_tensor_neg, task_imgs = self.dataset.load_data(self.data_idx)
        return task_tensor_pos, task_tensor_neg, task_imgs

    def reset(self, seed=123):
        """
        Reset the environment state at the beginning of an episode.
        Returns: initial observation
        """
        # load object positions in [0,1]x[0,1]
        self.ocm_pos, self.ocm_neg, self.imgs = self.next_data()
        self.imgs_pos = self.imgs[:3]
        self.imgs_neg = self.imgs[3:]
        self.visual_img = [chart_utils.hconcat_imgs(self.imgs)]

        # self.labels = torch.zeros(len(self.ocm))
        # Could also store "group labels" or other info if needed
        self.steps_taken = 0
        info = {
            "steps_taken": self.steps_taken
        }

        self.args.output_file_prefix = config.model_gestalt / f'data_{self.data_idx}'
        os.makedirs(self.args.output_file_prefix, exist_ok=True)

        # convert rgb image to black-white image
        self.segments = perception.percept_segments(self.args, self.imgs_pos, "pos")
        self.segments_neg = perception.percept_segments(self.args, self.imgs_neg, "neg")
        self.labels = [torch.zeros(len(seg)) - 1 for seg in self.segments]
        self.labels_neg = [torch.zeros(len(seg)) - 1 for seg in self.segments_neg]
        # self.ocm = [self.ocm[i][:len(self.segments[i])] for i in range(len(self.segments))]
        return self.ocm_pos, info

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
        # print(f"step: {self.steps_taken}")
        truncated = False
        principle_float = action[0]
        threshold_float = action[1]

        principle_idx = int(round(principle_float * self.n_principles - 1))
        threshold = threshold_float * self.max_threshold

        gcm, groups = perception.cluster_by_principle(self.steps_taken, self.args, self.ocm_pos, self.segments,
                                                      self.labels, principle_idx, threshold)
        ocm = [torch.from_numpy(self.ocm_pos[i][:len(self.segments[i])]) for i in range(len(self.segments))]

        lang = alpha.search_clauses(self.args, ocm, gcm, groups)

        # check negative
        gcm_neg, groups_neg = perception.cluster_by_principle(self.steps_taken, self.args, self.ocm_neg,
                                                              self.segments_neg, self.labels, principle_idx, threshold)
        ocm_neg = [torch.from_numpy(self.ocm_neg[i][:len(self.segments_neg[i])]) for i in range(len(self.segments_neg))]

        eval_nsfr.check_clauses(self.args, lang, ocm_neg, gcm_neg, groups_neg)
        # For demonstration, we give a random reward:
        reward = perception.percept_reward(lang)

        self.max_group_id = max([max(label) for label in self.labels])
        # For a realistic environment, you'd transform self.positions or maintain
        # a 'current_grouping' data structure. Here, we just keep positions static.

        self.steps_taken += 1
        self.total_step += 1
        terminated = (self.steps_taken >= self.max_steps)
        if reward == 0:
            truncated = True
            # if step % 10 == 0:
            # chart_utils.save_img(self.visual_img, config.models / "visual" / f"step_{self.total_step}.png")
        else:
            terminated = True
            print(f"{lang.clauses}")

        self.visual_img.append(
            chart_utils.visual_rl_step(self.imgs, ocm, groups, config.gestalt_action[principle_idx], reward))
        self.visual_img = chart_utils.vconcat_imgs(self.visual_img)
        chart_utils.save_img(self.visual_img, config.models / "visual" / f"step_{self.total_step}.png")
        # info can hold debugging or diagnostic data
        info = {"action_chosen": action}

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
    env = GestaltGroupingEnv(
        args=args,
        obj_n=args.obj_n,
        n_principles=len(config.gestalt_action))

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
