"""Flat velocity walking for the STS3215 Microduck.

Starts from Isaac Lab's MicroDuck flat walking task with its gearbox-play variant (same observations, actions,
rewards and curriculum; the robot model uses the same body and joint names) and changes what differs for this robot:

- the model and servos (identified STS3215, 10-40 ms command delay, firmware target rate limit, gearbox play);
- slower commands, foot heights and step timing scaled to the bigger robot;
- tracking judged on the distance covered over a stride (the trunk sways as fast as this robot walks), with
  tolerances scaled down with the command ranges, and a reward for progress along the command that standing still forfeits;
- rewards for what the first run got wrong: torque, joint acceleration and landing-force penalties against
  shaking and overheating servos, a penalty on targets flicking back and forth every step (all ramped in once
  the gaits are learned), trunk height and a pose reward while walking against crouching, both feet down
  when told to stand, a stronger joint limit penalty, a contact penalty for the thighs against the trunk and hip
  yaw servos, and an episode end when a thigh touches the floor;
- randomization of what isn't measured yet (servo friction, trunk and head mass).
"""

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck import mdp
from isaaclab_tasks.contrib.microduck.flat_env_cfg import (
    _FOOT_SENSOR_CFG,
    MicroDuckVelocityFlatEnvCfg,
    _schedule,
)
from isaaclab_tasks.utils.hydra import PresetCfg

from . import mdp as sts_mdp
from .robot import BAM, SOLE_TO_ANKLE, STAND_HEIGHT, STS3215_CFG

SWING_HEIGHT = 0.026
"""Foot lift during swing [m] (the original 20 mm, scaled 1.3x)."""

AIR_TIME = (0.18, 0.45)
"""Rewarded swing duration [s]: the original 0.125-0.3 s, slowed for the STS3215's lower speed."""

TRUNK_HEIGHT = STAND_HEIGHT - 0.008
"""Rewarded trunk frame height [m]: the stand pose, less a little knee bend for walking."""

TRACKING_WINDOW = 0.6
"""Time over which the distance covered is compared with the command [s]: about one stride."""

LANDING_FORCE = 25.0
"""Foot force above which landing is penalized [N]: about 1.7 times the robot's weight."""

_SERVOS = SceneEntityCfg("robot", joint_names=["^(?!passive_).*"])
_GEOMETRY = "{ENV_REGEX_NS}/Robot/Geometry/trunk_base"


def _choose_backlash(cfg) -> None:
    """Replace each preset in ``cfg``'s fields with its ``backlash`` option (or ``default`` if it has none)."""
    for name, value in list(vars(cfg).items()):
        if isinstance(value, PresetCfg):
            setattr(cfg, name, getattr(value, "backlash", None) or value.default)


@configclass
class Sts3215VelocityFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """STS3215 Microduck flat velocity walking with Newton MJWarp, BAM servos and gearbox play."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = STS3215_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        for group in (self.observations.policy, self.observations.critic, self.rewards):
            _choose_backlash(group)

        # the firmware rate-limits its target, so the policy's commands reach the motor no faster than this
        self.actions.joint_pos = sts_mdp.RateLimitedJointPositionActionCfg(
            asset_name="robot",
            joint_names=self.actions.joint_pos.joint_names,
            preserve_order=True,
            scale=1.0,
            use_default_offset=True,
            max_velocity=BAM["max_velocity"],
        )

        # thighs against the trunk (with the hip yaw servos), and foot against foot
        self.scene.self_collision = ContactSensorCfg(
            prim_path=f"{_GEOMETRY}/.*/(upper_leg_left|upper_leg_right|ankle_left)",
            filter_prim_paths_expr=[_GEOMETRY, f"{_GEOMETRY}/.*/ankle_right"],
            update_period=self.sim.dt,
        )

        # slower walking: the servo's no-load speed is about half the original servo's
        ranges = self.commands.base_velocity.ranges
        ranges.lin_vel_x = (-0.15, 0.25)
        ranges.lin_vel_y = (-0.1, 0.1)
        ranges.ang_vel_z = (-0.6, 0.6)

        # Tracking is judged on the distance covered over about a stride, not on the velocity at each instant: while
        # stepping, the trunk's velocity swings by +-0.15 m/s and +-0.35 rad/s around its average, as much as the
        # commands themselves, so instantaneous tracking paid a robot walking at the right pace less than one
        # standing still, and every run ended up standing for the slow, backward, sideways and turning commands.
        # The tolerances are scaled with the commands (original 0.15 m/s and 0.35 rad/s for +-0.4 m/s and +-1 rad/s).
        tracked = {"command_name": "base_velocity", "window_s": TRACKING_WINDOW}
        self.rewards.track_lin_vel = RewTerm(func=sts_mdp.track_distance_exp, weight=4.0, params={**tracked, "std": 0.10})
        self.rewards.track_ang_vel = RewTerm(func=sts_mdp.track_turn_exp, weight=4.0, params={**tracked, "std": 0.25})
        # and progress along the command pays up to 3 per second and axis, which standing still or stepping on the
        # spot under a command to move gives up
        self.rewards.progress = RewTerm(
            func=sts_mdp.command_progress, weight=3.0, params={**tracked, "min_speed": 0.03, "min_turn": 0.1}
        )

        target = SWING_HEIGHT + SOLE_TO_ANKLE
        self.rewards.foot_clearance.params["target_height"] = target
        self.rewards.foot_swing_height.params["target_height"] = target
        self.rewards.foot_swing_height.weight = -0.25 * (target / 0.02) ** 2
        self.rewards.air_time.params["threshold_min"], self.rewards.air_time.params["threshold_max"] = AIR_TIME

        # Smoothness and effort come after the gait. The policy explores with noise on every target, and these terms
        # penalize that noise: applied from the start they shrank it before a foot ever stayed in the air long enough
        # to be rewarded, and the robot settled on standing still. So until iteration 1500 only the original task's
        # light action rate penalty applies.
        #
        # effort: final weights sized so the first run's gait costs about 0.2-0.7 per second each (tracking pays ~6.5)
        self.rewards.joint_torques = RewTerm(func=base_mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": _SERVOS})
        self.rewards.joint_acc = RewTerm(func=base_mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": _SERVOS})
        self.rewards.landing_force = RewTerm(
            func=base_mdp.contact_forces, weight=0.0, params={"threshold": LANDING_FORCE, "sensor_cfg": _FOOT_SENSOR_CFG}
        )
        for name, final in (("joint_torques", -0.05), ("joint_acc", -1e-5), ("landing_force", -0.1)):
            setattr(self.curriculum, f"{name}_weight", _schedule(
                f"rewards.{name}.weight", (1500, final / 3), (2250, 2 * final / 3), (3000, final)
            ))
        self.curriculum.action_rate_weight = _schedule(
            "rewards.action_rate_l2.weight", (1500, -0.3), (2000, -0.6), (2500, -1.0), (3250, -1.5), (4000, -2.0)
        )
        # Targets flicking every step cost 0.42 per unit weight and second, a smooth gait 0.06 (last policy's
        # actions), but exploration noise of std s costs 84 s^2: more than either until late in training (s is about
        # 0.1 from iteration 4500). So this comes last and stays small.
        self.rewards.action_flicker = RewTerm(func=sts_mdp.action_flicker_l2, weight=0.0)
        self.curriculum.action_flicker_weight = _schedule(
            "rewards.action_flicker.weight", (4500, -0.25), (5500, -0.5)
        )

        # posture: stand tall, stay near the stand pose while walking too, and keep both feet down when standing
        self.rewards.trunk_height = RewTerm(
            func=sts_mdp.trunk_height_exp, weight=1.0, params={"target": TRUNK_HEIGHT, "std": 0.025}
        )
        pose = self.rewards.pose.params
        pose["standing_only"] = False
        pose["std_standing"] = {k: 1.5 * v for k, v in pose["std_standing"].items()}
        self.rewards.feet_down_standing = RewTerm(
            func=sts_mdp.feet_down_when_standing,
            weight=1.0,
            params={"command_name": "base_velocity", "sensor_cfg": _FOOT_SENSOR_CFG},
        )

        # a joint held against its limit stalls its servo: the hip yaws sat there through a whole run at -1
        self.rewards.dof_pos_limits.weight = -10.0

        # the thighs can rest on the floor, and kneeling on them with the feet in the air would collect the
        # step-timing reward
        self.terminations.knelt = DoneTerm(
            func=sts_mdp.touching_ground,
            params={"sensor_cfg": SceneEntityCfg("self_collision", body_names=["upper_leg_.*"])},
        )

        # randomize what isn't measured: one servo was identified, and the trunk and head contents are estimates
        self.events.randomize_joint_friction.params["scale_range"] = (0.8, 1.2)
        self.events.mass_inertia.params["mass_distribution_params"] = (0.85, 1.15)
        self.events.head_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["jaw_soft"]),
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )
        self.events.push_robot.params["velocity_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}


@configclass
class Sts3215TurnVelocityFlatEnvCfg(Sts3215VelocityFlatEnvCfg):
    """Same task, aimed at the commands the first BAM policy ignores: turning, and walking while turning.

    A wider yaw range than the base task's +-0.6 rad/s gives the policy margin past the target, less reversing puts
    more of the samples into walking while turning, and fewer standing episodes leave more time on moving commands.
    These numbers are starting values, not tuned.
    """

    def __post_init__(self):
        super().__post_init__()
        cmd = self.commands.base_velocity
        cmd.ranges.ang_vel_z = (-0.8, 0.8)
        cmd.ranges.lin_vel_x = (-0.05, 0.25)
        cmd.rel_standing_envs = 0.1
