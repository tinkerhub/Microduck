"""Terms the STS3215 task adds to Isaac Lab's MicroDuck terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.mdp.actions import BiasedJointPositionAction, BiasedJointPositionActionCfg
from isaaclab_tasks.contrib.microduck.mdp.observations import zero_non_finite
from isaaclab_tasks.contrib.microduck.mdp.rewards import _command_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


class RateLimitedJointPositionAction(BiasedJointPositionAction):
    """Joint position action as the STS3215 firmware applies it.

    The firmware does not track a new goal position directly: it moves an internal target towards it at no more
    than ``max_velocity`` and runs its position controller on that target (Rhoban BAM's STS3215 model). The internal
    target starts from the measured position after a reset, as when the servo powers on.

    With Newton's actuators the physics runs all substeps of a policy step in one call, so this term runs once per
    policy step: the internal target then advances by the whole step's allowance at once.
    """

    cfg: RateLimitedJointPositionActionCfg

    def __init__(self, cfg: RateLimitedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._last_sim_step = 0
        self._fw_target = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._seed = torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._seed.fill_(True)
        else:
            self._seed[env_ids] = True

    def apply_actions(self):
        # in place throughout: the step may run as a captured CUDA graph
        goal = self.processed_actions - self.encoder_bias
        q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self._fw_target.copy_(torch.where(self._seed, q, self._fw_target))
        self._seed.fill_(False)
        # allowance for the physics steps since the last call (1 or the decimation, depending on who owns it)
        steps = max(self._env._sim_step_counter - self._last_sim_step, 1)
        self._last_sim_step = self._env._sim_step_counter
        max_step = self.cfg.max_velocity * self._env.physics_dt * steps
        self._fw_target.copy_(torch.clamp(goal, self._fw_target - max_step, self._fw_target + max_step))
        self._asset.set_joint_position_target_index(target=self._fw_target, joint_ids=self._joint_ids)


@configclass
class RateLimitedJointPositionActionCfg(BiasedJointPositionActionCfg):
    class_type: type[RateLimitedJointPositionAction] = RateLimitedJointPositionAction
    max_velocity: float = 5.1
    """Fastest the firmware moves its internal target [rad/s]."""


def trunk_height_exp(
    env: ManagerBasedRLEnv, target: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward keeping the trunk frame near ``target`` [m] above flat ground, with tolerance ``std`` [m]."""
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return zero_non_finite(torch.exp(-torch.square((height - target) / std)))


def feet_down_when_standing(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, command_threshold: float = 0.01
) -> torch.Tensor:
    """1 while the command is to stand and both feet are on the ground, else 0."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    down = (sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0).all(dim=1)
    standing = _command_magnitude(env, command_name) < command_threshold
    return (down & standing).float()


def touching_ground(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 2.0) -> torch.Tensor:
    """True when a selected body of a filtered contact sensor is pushed by something other than its filter bodies.

    The sensor's net force counts every contact and its force matrix only those with the filter bodies (the rest
    of the robot), so what remains is the floor.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    robot = sensor.data.force_matrix_w.torch[:, sensor_cfg.body_ids].sum(dim=2)
    return (zero_non_finite(net - robot).norm(dim=-1) > threshold).any(dim=1)


class action_flicker_l2(ManagerTermBase):
    """Penalize the change in action rate, ``a_t - 2 a_t-1 + a_t-2``, squared and summed over the actions.

    A target that jumps back and forth every policy step scores high (the servo current reverses each step), a
    smooth gait barely registers. The action rate penalty alone does not separate the two.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_prev = torch.zeros_like(env.action_manager.prev_action)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._prev_prev.zero_()
        else:
            self._prev_prev[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        action, prev = env.action_manager.action, env.action_manager.prev_action
        cost = torch.sum(torch.square(action - 2.0 * prev + self._prev_prev), dim=1)
        self._prev_prev.copy_(prev)
        return zero_non_finite(cost)


class _DistanceTerm(ManagerTermBase):
    """Base for terms on how far the robot actually got: its average velocity over the last ``window_s`` seconds.

    The velocity is taken from the distance covered and the heading turned (positions, not the velocity state), in
    the heading frame the commands use, and averaged over about one stride. The trunk of a walking robot sways
    faster than it travels at these command speeds, so its velocity at one instant says little about whether it is
    following the command; where it got to does, and that cannot be collected by swaying or stepping on the spot.

    The window restarts when the episode or the command does, so a robot is never judged on distance covered under
    another command.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        n = max(round(cfg.params["window_s"] / env.step_dt), 1)
        self._steps = torch.zeros(env.num_envs, n, 3, device=env.device)  # vx, vy [m/s], yaw rate [rad/s] per step
        self._count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._slot = 0
        self._slots = torch.arange(n, device=env.device)
        self._pos = torch.zeros(env.num_envs, 3, device=env.device)
        self._yaw = torch.zeros(env.num_envs, device=env.device)
        self._command = torch.zeros(env.num_envs, 3, device=env.device)
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self._forward = torch.tensor([[1.0, 0.0, 0.0]], device=env.device).expand(env.num_envs, 3)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._fresh.fill_(True)
        else:
            self._fresh[env_ids] = True

    def _average_velocity(self, env: ManagerBasedRLEnv, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the window by this step; return the average ``(vx, vy, yaw rate)`` and the command."""
        pos, quat = self._asset.data.root_link_pos_w.torch, self._asset.data.root_link_quat_w.torch
        heading = math_utils.quat_apply(quat, self._forward)
        yaw = torch.atan2(heading[:, 1], heading[:, 0])
        moved = math_utils.quat_apply_inverse(math_utils.yaw_quat(quat), pos - self._pos)[:, :2]
        turned = math_utils.wrap_to_pi(yaw - self._yaw)
        step = torch.nan_to_num(torch.cat((moved, turned.unsqueeze(1)), dim=1) / env.step_dt, 0.0, 0.0, 0.0)
        self._pos.copy_(pos)
        self._yaw.copy_(yaw)

        command = env.command_manager.get_command(command_name)
        # after a reset the last position is from before it, so this step's distance means nothing
        self._count[self._fresh | (command != self._command).any(dim=1)] = 0
        self._command.copy_(command)
        self._steps[:, self._slot] = torch.where(self._fresh.unsqueeze(1), torch.zeros_like(step), step)
        self._count += (~self._fresh).long()
        self._count.clamp_(max=self._steps.shape[1])
        self._fresh.fill_(False)

        age = (self._slot - self._slots) % self._steps.shape[1]
        in_window = (age.unsqueeze(0) < self._count.unsqueeze(1)).unsqueeze(2)
        self._slot = (self._slot + 1) % self._steps.shape[1]
        return (self._steps * in_window).sum(dim=1) / self._count.clamp(min=1).unsqueeze(1), command


class track_distance_exp(_DistanceTerm):
    """Reward covering the commanded distance: 1 when the average velocity matches the command, with tolerance
    ``std`` [m/s], falling off like the instantaneous tracking reward it replaces."""

    def __call__(self, env: ManagerBasedRLEnv, command_name: str, std: float, window_s: float) -> torch.Tensor:
        velocity, command = self._average_velocity(env, command_name)
        error = torch.sum(torch.square(command[:, :2] - velocity[:, :2]), dim=1)
        return zero_non_finite(torch.exp(-error / std**2))


class track_turn_exp(_DistanceTerm):
    """Reward turning the commanded angle: as :class:`track_distance_exp`, for the yaw rate, ``std`` in rad/s."""

    def __call__(self, env: ManagerBasedRLEnv, command_name: str, std: float, window_s: float) -> torch.Tensor:
        velocity, command = self._average_velocity(env, command_name)
        return zero_non_finite(torch.exp(-torch.square(command[:, 2] - velocity[:, 2]) / std**2))


class command_progress(_DistanceTerm):
    """Reward getting somewhere when told to move: the share of the commanded pace that is reached.

    For a command to walk, the pace is the average velocity along the commanded direction; for a command to turn,
    the average yaw rate in the commanded sense. Each earns 1 at the commanded pace or faster and 0 when standing
    still, going sideways or going the wrong way, so the two add up to at most 2. Unlike the tracking rewards,
    which are flat far from the command, this grows with every bit of progress, and stepping on the spot earns
    nothing. Commands below ``min_speed`` [m/s] and ``min_turn`` [rad/s] ask for no progress.

    Standing still under a command loses this reward. It is not written as a penalty so that a robot that isn't
    walking yet never does better by falling over and ending the episode.
    """

    def __call__(
        self, env: ManagerBasedRLEnv, command_name: str, window_s: float, min_speed: float, min_turn: float
    ) -> torch.Tensor:
        velocity, command = self._average_velocity(env, command_name)
        speed = torch.norm(command[:, :2], dim=1)
        walked = torch.sum(velocity[:, :2] * command[:, :2], dim=1) / speed.clamp(min=min_speed) ** 2
        turned = velocity[:, 2] * command[:, 2] / command[:, 2].abs().clamp(min=min_turn) ** 2
        reached = walked.clamp(0.0, 1.0) * (speed > min_speed) + turned.clamp(0.0, 1.0) * (
            command[:, 2].abs() > min_turn
        )
        return zero_non_finite(reached)
