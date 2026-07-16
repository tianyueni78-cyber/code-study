import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

# ==================================================
# 模块一：定义简化AGV环境
# ==================================================

class SimpleAGVEnv(gym.Env):

    metadata = {"render_modes": ["console"]}

    def __init__(
        self,
        track_length=7,
        start_position=0,
        pickup_position=2,
        delivery_position=6,
        max_steps=30,
    ):
        super().__init__()

        self.track_length = track_length
        self.start_position = start_position
        self.pickup_position = pickup_position
        self.delivery_position = delivery_position
        self.max_steps = max_steps

        assert 0 <= self.start_position < self.track_length
        assert 0 <= self.pickup_position < self.track_length
        assert 0 <= self.delivery_position < self.track_length
        assert self.pickup_position != self.delivery_position

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array(
                [self.track_length - 1, 1, self.max_steps],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.position = self.start_position
        self.has_load = False
        self.current_step = 0

    def _get_observation(self):
        return np.array(
            [self.position, float(self.has_load), self.current_step],
            dtype=np.float32,
        )

    def _get_info(self):
        current_target = (
            self.delivery_position if self.has_load else self.pickup_position
        )
        return {
            "position": self.position,
            "has_load": self.has_load,
            "current_step": self.current_step,
            "current_target": current_target,
            "distance_to_target": abs(self.position - current_target),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self.start_position
        self.has_load = False
        self.current_step = 0
        return self._get_observation(), self._get_info()

    def step(self, action):
        assert self.action_space.contains(action), f"非法动作：{action}"

        old_position = self.position
        target_before_action = (
            self.delivery_position if self.has_load else self.pickup_position
        )
        old_distance = abs(old_position - target_before_action)

        hit_boundary = False

        if action == 0:
            if self.position > 0:
                self.position -= 1
            else:
                hit_boundary = True
        elif action == 2:
            if self.position < self.track_length - 1:
                self.position += 1
            else:
                hit_boundary = True

        self.current_step += 1
        reward = -0.1

        new_distance = abs(self.position - target_before_action)
        if new_distance < old_distance:
            reward += 0.2
        elif new_distance > old_distance:
            reward -= 0.2

        if hit_boundary:
            reward -= 0.5

        picked_up_now = False
        if self.position == self.pickup_position and not self.has_load:
            self.has_load = True
            picked_up_now = True
            reward += 2.0

        delivered = (
            self.position == self.delivery_position and self.has_load
        )
        if delivered:
            reward += 10.0

        terminated = delivered
        truncated = self.current_step >= self.max_steps and not terminated

        info = self._get_info()
        info["hit_boundary"] = hit_boundary
        info["picked_up_now"] = picked_up_now
        info["delivered"] = delivered

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        print(
            f"位置={self.position}，状态={'载货' if self.has_load else '空载'}，步数={self.current_step}"
        )


class AGVObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        base_env = env.unwrapped
        self.max_position = base_env.track_length - 1
        self.max_steps = base_env.max_steps
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

    def observation(self, observation):
        return np.array(
            [
                observation[0] / self.max_position,
                observation[1],
                observation[2] / self.max_steps,
            ],
            dtype=np.float32,
        )


def make_agv_env():
    env = SimpleAGVEnv()
    env = AGVObservationWrapper(env)
    return env


check_env_instance = make_agv_env()
check_env(check_env_instance, warn=True)
check_env_instance.close()

train_env = make_vec_env(make_agv_env, n_envs=1, seed=42)
eval_env = make_vec_env(make_agv_env, n_envs=1, seed=100)


class TrainingPrintCallback(BaseCallback):
    def __init__(self, print_freq=2000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self):
        if self.n_calls % self.print_freq == 0:
            print("Callback调用次数：", self.n_calls, "训练总步数：", self.num_timesteps)
        return True


Path("./agv_checkpoints/").mkdir(parents=True, exist_ok=True)
Path("./agv_best_model/").mkdir(parents=True, exist_ok=True)
Path("./agv_eval_logs/").mkdir(parents=True, exist_ok=True)

print_callback = TrainingPrintCallback(print_freq=2000)
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="./agv_checkpoints/",
    name_prefix="ppo_agv",
    verbose=1,
)
eval_callback = EvalCallback(
    eval_env=eval_env,
    eval_freq=2000,
    n_eval_episodes=10,
    deterministic=True,
    best_model_save_path="./agv_best_model/",
    log_path="./agv_eval_logs/",
    verbose=1,
)

model = PPO(
    policy="MlpPolicy",
    env=train_env,
    verbose=1,
    seed=42,
)

model.learn(
    total_timesteps=20000,
    callback=[print_callback, checkpoint_callback, eval_callback],
    progress_bar=True,
)

best_model_path = "./agv_best_model/best_model"

if os.path.exists(best_model_path + ".zip"):
    test_model = PPO.load(best_model_path)
    print("使用评估期间保存的最佳模型")
else:
    test_model = model
    print("没有发现最佳模型文件，使用当前模型")

test_env = make_agv_env()
observation, info = test_env.reset(seed=123)
episode_reward = 0.0
action_names = {0: "向左", 1: "等待", 2: "向右"}

for test_step in range(30):
    action, _ = test_model.predict(observation, deterministic=True)
    action_int = int(np.asarray(action).item())

    observation, reward, terminated, truncated, info = test_env.step(action_int)
    episode_reward += reward

    print(
        f"测试步数={test_step + 1}，动作={action_names[action_int]}，"
        f"位置={info['position']}，载货={info['has_load']}，奖励={reward:.2f}"
    )

    if terminated or truncated:
        print(
            f"本局结束，自然结束={terminated}，时间截断={truncated}，累计奖励={episode_reward:.2f}"
        )
        break

test_env.close()
eval_env.close()
train_env.close()
