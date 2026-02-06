'''import math
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class RewardCfg:
    alpha_progress: float = 1.0
    step_penalty: float = 0.1
    bump_penalty: float = 1.0
    goal_bonus: float = 100.0
    max_steps: int = 200


class SimpleLAM:
    """A very small 'LAM' planner that knows the map and suggests a direction.

    suggest_direction returns a pair (lam_dx, lam_dy) each in {-1,0,1}.
    suggest_reward_cfg optionally suggests small tuning to RewardCfg based on a
    reported success rate.
    """

    def __init__(self):
        # Map facts used by the LAM
        self.gap = (4, 5)
        self.horizontal_wall_xs = set(range(1, 5))  # x=1..4 at y=5
        self.horizontal_wall_y = 5
        self.vertical_wall_ys = set(range(1, 5))  # y=1..4 at x=5
        self.vertical_wall_x = 5

    def suggest_direction(self, current: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[int, int]:
        x, y = current
        gx, gy = goal

        # If agent is in the top-left region (left of vertical wall and above bottom wall),
        # steer toward the gap first if the goal is on the right side.
        need_gap = gx > 4 and (x <= 4 or y <= 5)
        if need_gap and (x, y) != self.gap:
            target = self.gap
        else:
            target = (gx, gy)

        tx, ty = target

        # Direction as sign of difference, clipped to -1/0/1
        def s(v):
            return 0 if v == 0 else (1 if v > 0 else -1)

        dx = s(tx - x)
        dy = s(ty - y)

        # Prefer the axis with larger absolute distance to target
        if abs(tx - x) >= abs(ty - y):
            # prioritize horizontal movement
            return (dx, 0)
        else:
            return (0, dy)

    def suggest_reward_cfg(self, prev_success_rate: Optional[float]) -> RewardCfg:
        # Simple heuristic: if success rate is low, reduce step penalty slightly
        cfg = RewardCfg()
        if prev_success_rate is None:
            return cfg
        if prev_success_rate < 0.2:
            cfg.step_penalty = max(0.01, cfg.step_penalty * 0.5)
            cfg.alpha_progress = cfg.alpha_progress * 1.0
        elif prev_success_rate < 0.5:
            cfg.step_penalty = max(0.03, cfg.step_penalty * 0.8)
        return cfg


class TwoWallsGap10x10LAMEnv(gym.Env):
    """10x10 gridworld with two walls and a gap.

    Observation: [x_norm, y_norm, gx_norm, gy_norm, lam_dx, lam_dy]
    Action: Box(-1,1, shape=(2,)) continuous. Dominant axis mapped to discrete move.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, goal: Tuple[int, int], start: Tuple[int,int] = (0,0), reward_cfg: Optional[RewardCfg] = None, lam: Optional[SimpleLAM] = None):
        super().__init__()
        self.width = 10
        self.height = 10
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.lam = lam if lam is not None else SimpleLAM()
        self.reward_cfg = reward_cfg if reward_cfg is not None else RewardCfg()

        # Action space: continuous 2D [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: x_norm, y_norm, gx_norm, gy_norm in [0,1]; lam_dx, lam_dy in [-1,1]
        low = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Walls definition
        self.h_wall_y = 5
        self.h_wall_xs = set(range(1, 5))  # (1,5) .. (4,5) blocked; (4,5) is gap override
        self.gap = (4, 5)
        self.v_wall_x = 5
        self.v_wall_ys = set(range(1, 5))  # (5,1)..(5,4)

        self.max_steps = self.reward_cfg.max_steps
        self._rng = np.random.default_rng()

        self._agent_pos = (0, 0)
        self._steps = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # On reset, place agent at configured start
        self._agent_pos = self.start
        self._steps = 0
        obs = self._get_obs()
        return obs, {}

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_wall(self, x: int, y: int) -> bool:
        # gap cell is open
        if (x, y) == self.gap:
            return False
        if y == self.h_wall_y and x in self.h_wall_xs:
            return True
        if x == self.v_wall_x and y in self.v_wall_ys:
            return True
        return False

    def _get_obs(self):
        x, y = self._agent_pos
        gx, gy = self.goal
        x_norm = x / (self.width - 1)
        y_norm = y / (self.height - 1)
        gx_norm = gx / (self.width - 1)
        gy_norm = gy / (self.height - 1)
        lam_dx, lam_dy = self.lam.suggest_direction(self._agent_pos, self.goal)
        obs = np.array([x_norm, y_norm, gx_norm, gy_norm, float(lam_dx), float(lam_dy)], dtype=np.float32)
        return obs

    def step(self, action):
        ax, ay = float(action[0]), float(action[1])
        self._steps += 1

        prev_dist = self._manhattan(self._agent_pos, self.goal)

        # Map continuous to discrete move
        move = (0, 0)
        thresh = 0.2
        if abs(ax) >= abs(ay):
            if abs(ax) > thresh:
                move = (1 if ax > 0 else -1, 0)
        else:
            if abs(ay) > thresh:
                move = (0, 1 if ay > 0 else -1)

        intended = (self._agent_pos[0] + move[0], self._agent_pos[1] + move[1])

        bumped = False
        if move != (0, 0):
            x2, y2 = intended
            if (not self._in_bounds(x2, y2)) or self._is_wall(x2, y2):
                bumped = True
                # stay in place
            else:
                self._agent_pos = (x2, y2)

        curr_dist = self._manhattan(self._agent_pos, self.goal)

        # Rewards
        r = 0.0
        r += self.reward_cfg.alpha_progress * (prev_dist - curr_dist)
        r -= self.reward_cfg.step_penalty
        if bumped:
            r -= self.reward_cfg.bump_penalty

        done = False
        info = {}
        if self._agent_pos == self.goal:
            r += self.reward_cfg.goal_bonus
            done = True
            info["is_success"] = True
        elif self._steps >= self.max_steps:
            done = True
            info["is_success"] = False

        obs = self._get_obs()
        return obs, float(r), done, False, info

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def render(self, mode="rgb_array"):
        # Build a small image for the grid: each cell is scaled up
        cell = 24
        img = np.ones((self.height * cell, self.width * cell, 3), dtype=np.uint8) * 255

        # Paint walls
        for x in range(self.width):
            for y in range(self.height):
                if self._is_wall(x, y):
                    img[y * cell:(y + 1) * cell, x * cell:(x + 1) * cell] = 0

        # Paint goal in green
        gx, gy = self.goal
        img[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell] = np.array([0, 200, 0], dtype=np.uint8)

        # Paint agent in blue
        ax, ay = self._agent_pos
        img[ay * cell:(ay + 1) * cell, ax * cell:(ax + 1) * cell] = np.array([50, 50, 255], dtype=np.uint8)

        return img

    def close(self):
        return None


__all__ = ["RewardCfg", "SimpleLAM", "TwoWallsGap10x10LAMEnv"]'''

#New Code for self thinking SAC
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch
import torch.nn as nn


@dataclass
class RewardCfg:
    """
    Configuration for rewards. 
    These values are now Mutable and controlled by the LAM.
    """
    alpha_progress: float = 1.0
    step_penalty: float = 0.1
    bump_penalty: float = 1.0
    goal_bonus: float = 100.0
    max_steps: int = 200


class LargeActionModel(nn.Module):
    """
    The new 'Commander' LAM.
    Input: A vector representing a command (e.g., encoded prompt).
    Output: Reward Shaping parameters (step_penalty, bump_penalty).
    """
    def __init__(self, input_size: int = 10, output_size: int = 2):
        super(LargeActionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Sigmoid() # Outputs normalized values [0, 1] we can scale later
        )

    def forward(self, command_vector):
        return self.network(command_vector)


class TwoWallsGap10x10LAMEnv(gym.Env):
    """10x10 gridworld with two walls and a gap.

    Updated for Commander-Scout Architecture:
    Observation: [x_norm, y_norm, gx_norm, gy_norm] (No more hints!)
    Action: Box(-1,1, shape=(2,)) continuous.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, goal: Tuple[int, int], start: Tuple[int,int] = (0,0), reward_cfg: Optional[RewardCfg] = None, lam_model: Optional[LargeActionModel] = None):
        super().__init__()
        self.width = 10
        self.height = 10
        self.start = tuple(start)
        self.goal = tuple(goal)
        
        # Load the LAM (Commander)
        self.lam_model = lam_model if lam_model is not None else LargeActionModel()
        self.reward_cfg = reward_cfg if reward_cfg is not None else RewardCfg()

        # Action space: continuous 2D [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: x_norm, y_norm, gx_norm, gy_norm in [0,1]
        # REMOVED: lam_dx, lam_dy (The SAC must now think for itself)
        low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Walls definition
        self.h_wall_y = 5
        self.h_wall_xs = set(range(1, 5))
        self.gap = (4, 5)
        self.v_wall_x = 5
        self.v_wall_ys = set(range(1, 5))

        self.max_steps = self.reward_cfg.max_steps
        self._rng = np.random.default_rng()

        self._agent_pos = (0, 0)
        self._steps = 0

    def update_lam_context(self, command_vector: torch.Tensor):
        """
        Takes a command from the Terminal (via LAM), and updates the Environment's physics/rewards.
        This is how the LAM 'controls' the SAC without hints.
        """
        with torch.no_grad():
            # Get raw output from LAM (values 0 to 1)
            lam_output = self.lam_model(command_vector)
            
            # Map Output 0 -> Step Penalty (Urgency)
            # If LAM outputs 0.9, step penalty becomes high (Don't dawdle!)
            # If LAM outputs 0.1, step penalty is low (Take your time)
            self.reward_cfg.step_penalty = float(lam_output[0] * 1.0) 
            
            # Map Output 1 -> Bump Penalty (Caution)
            # If LAM outputs 0.9, bump penalty is massive (Be careful!)
            self.reward_cfg.bump_penalty = float(lam_output[1] * 10.0)
            
            # print(f"[ENV] LAM Updated Rewards: StepCost={self.reward_cfg.step_penalty:.2f}, BumpCost={self.reward_cfg.bump_penalty:.2f}")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._agent_pos = self.start
        self._steps = 0
        obs = self._get_obs()
        return obs, {}

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_wall(self, x: int, y: int) -> bool:
        if (x, y) == self.gap:
            return False
        if y == self.h_wall_y and x in self.h_wall_xs:
            return True
        if x == self.v_wall_x and y in self.v_wall_ys:
            return True
        return False

    def _get_obs(self):
        x, y = self._agent_pos
        gx, gy = self.goal
        x_norm = x / (self.width - 1)
        y_norm = y / (self.height - 1)
        gx_norm = gx / (self.width - 1)
        gy_norm = gy / (self.height - 1)
        
        # REMOVED: lam.suggest_direction calls.
        # The SAC sees only reality, no hints.
        obs = np.array([x_norm, y_norm, gx_norm, gy_norm], dtype=np.float32)
        return obs

    def step(self, action):
        ax, ay = float(action[0]), float(action[1])
        self._steps += 1

        prev_dist = self._manhattan(self._agent_pos, self.goal)

        # Map continuous to discrete move
        move = (0, 0)
        thresh = 0.2
        if abs(ax) >= abs(ay):
            if abs(ax) > thresh:
                move = (1 if ax > 0 else -1, 0)
        else:
            if abs(ay) > thresh:
                move = (0, 1 if ay > 0 else -1)

        intended = (self._agent_pos[0] + move[0], self._agent_pos[1] + move[1])

        bumped = False
        if move != (0, 0):
            x2, y2 = intended
            if (not self._in_bounds(x2, y2)) or self._is_wall(x2, y2):
                bumped = True
            else:
                self._agent_pos = (x2, y2)

        curr_dist = self._manhattan(self._agent_pos, self.goal)

        # Rewards - Now fully dynamic based on self.reward_cfg (controlled by LAM)
        r = 0.0
        r += self.reward_cfg.alpha_progress * (prev_dist - curr_dist)
        r -= self.reward_cfg.step_penalty  # Variable pain for taking time
        if bumped:
            r -= self.reward_cfg.bump_penalty # Variable pain for hitting walls

        done = False
        info = {}
        if self._agent_pos == self.goal:
            r += self.reward_cfg.goal_bonus
            done = True
            info["is_success"] = True
        elif self._steps >= self.max_steps:
            done = True
            info["is_success"] = False

        obs = self._get_obs()
        return obs, float(r), done, False, info

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def render(self, mode="rgb_array"):
        cell = 24
        img = np.ones((self.height * cell, self.width * cell, 3), dtype=np.uint8) * 255

        for x in range(self.width):
            for y in range(self.height):
                if self._is_wall(x, y):
                    img[y * cell:(y + 1) * cell, x * cell:(x + 1) * cell] = 0

        gx, gy = self.goal
        img[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell] = np.array([0, 200, 0], dtype=np.uint8)

        ax, ay = self._agent_pos
        img[ay * cell:(ay + 1) * cell, ax * cell:(ax + 1) * cell] = np.array([50, 50, 255], dtype=np.uint8)

        return img

    def close(self):
        return None


__all__ = ["RewardCfg", "LargeActionModel", "TwoWallsGap10x10LAMEnv"]