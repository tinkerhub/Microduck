"""Evaluate a trained policy and export it for the robot.

Default: fixed commands (stand, walk, back, sidestep, turn, arc), several robots per command, each command held
for the whole run, headless. Prints speed tracking, posture, gait timing, action smoothness and servo use against
the STS3215 envelope (torque is sampled at every physics step), saves the time series and the joint trajectories
(degrees, for cad/sts3215/check_rom.py --traj), and exports the policy to ONNX and TorchScript in <run>/exported/.

With --robust: the training environment instead (random commands, sensor noise, randomized robot, pushes) with
many robots, to count falls.

    cd ~/IsaacLab && .venv/bin/python ~/microduck/rl/eval_policy.py \
        --task Microduck-STS3215-Velocity-Flat --checkpoint latest [--robust] [--delay N]
"""

import contextlib
import os
import sys

import warp as wp

wp.config.enable_backward = False

import numpy as np  # noqa: E402
import torch  # noqa: E402

import microduck_sts3215  # noqa: E402, F401  (registers the tasks)
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint  # noqa: E402
from isaaclab_rl.entrypoints.backends.runner_rsl_rl import create_rsl_rl_runner  # noqa: E402
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from microduck_sts3215.robot import SERVO, STANDING_POSE  # noqa: E402, F401

# name: (vx m/s, vy m/s, yaw rad/s)
COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "walk_slow": (0.10, 0.0, 0.0),
    "walk": (0.25, 0.0, 0.0),
    "back": (-0.15, 0.0, 0.0),
    "side": (0.0, 0.10, 0.0),
    "turn": (0.0, 0.0, 0.6),
    "arc": (0.15, 0.0, 0.3),
}
REPEAT = 16  # robots per command
SECONDS = 12.0
WARMUP_S = 2.0  # ignored in the statistics (start from the stand pose)
ROBUST_ENVS, ROBUST_SECONDS = 1024, 20.0
FEET = ["ankle_left", "ankle_right"]
CONTACT_N = 1.0


def main(argv):
    robust = "--robust" in argv
    argv = [a for a in argv if a != "--robust"]
    delay = 0  # --delay N: apply each action N policy steps (20 ms each) late, as a slow servo bus would
    if "--delay" in argv:
        k = argv.index("--delay")
        delay, argv = int(argv[k + 1]), argv[:k] + argv[k + 2:]
    n_envs = ROBUST_ENVS if robust else len(COMMANDS) * REPEAT
    seconds = ROBUST_SECONDS if robust else SECONDS
    args_cli = _parse_args(argv + ["--num_envs", str(n_envs)])
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not robust)
    schedules = vars(env_cfg.curriculum)
    if not robust:
        for name in ("base_velocity", "head_pose", "body_pose"):
            getattr(env_cfg.commands, name).resampling_time_range = (1e9, 1e9)
        env_cfg.commands.head_pose.ranges = tuple((0.0, 0.0) for _ in env_cfg.commands.head_pose.ranges)
        env_cfg.commands.body_pose.ranges = tuple((0.0, 0.0) for _ in env_cfg.commands.body_pose.ranges)
        env_cfg.curriculum = None  # it would widen the head command range again
    env_cfg.episode_length_s = seconds + 5.0

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root)
        run_dir = os.path.dirname(path)
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg, inference=True)
        runner.load(path, map_location=agent_cfg.device)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        base = env.unwrapped
        robot = base.scene["robot"]
        sensor = base.scene["contact_forces"]
        self_contact = base.scene["self_collision"]
        servo = [i for i, n in enumerate(robot.joint_names) if not n.startswith("passive_")]
        play = [robot.joint_names.index(f"passive_{robot.joint_names[i]}_backlash") for i in servo]
        vel_term = base.command_manager.get_term("base_velocity")
        feet = [robot.body_names.index(f) for f in FEET]
        sensor_feet = [sensor.body_names.index(f) for f in FEET]
        origins = base.scene.env_origins
        print(f"\nmode={'robust' if robust else 'fixed'} envs={n_envs} step_dt={base.step_dt} "
              f"physics_dt={base.physics_dt} obs_noise={env_cfg.observations.policy.enable_corruption} "
              f"events={[k for k, v in env_cfg.events.__dict__.items() if v is not None]}")

        # Servo torque and speed at every physics step (the policy step only sees the last of 4).
        sub = {"tau": [], "dq": []}
        write = robot.write_data_to_sim

        def write_and_record():
            write()
            sub["tau"].append(robot.actuators.applied_effort.torch[:, servo].clone())
            sub["dq"].append(robot.data.joint_vel.torch[:, servo].clone())

        robot.write_data_to_sim = write_and_record

        cmd = None
        if not robust:
            cmd = torch.tensor(list(COMMANDS.values()), device=base.device, dtype=torch.float32)
            cmd = cmd.repeat_interleave(REPEAT, dim=0)

        def hold_commands():
            if cmd is not None:
                vel_term.vel_command_b[:] = cmd
                vel_term.is_standing_env[:] = cmd.abs().sum(dim=1) == 0

        # reward weights as at the end of training: the last stage of each weight's schedule
        rewards = base.reward_manager
        for term in schedules.values():
            address = getattr(term, "params", {}).get("address", "")
            if address.startswith("rewards.") and address.endswith(".weight"):
                cfg = rewards.get_term_cfg(address.split(".")[1])
                cfg.weight = term.params["modify_params"]["stages"][-1][1]
        forward = torch.tensor([[1.0, 0.0, 0.0]], device=base.device).expand(n_envs, 3)
        hold_commands()
        obs = env.get_observations()
        steps = int(seconds / base.step_dt)
        keys = ("q", "play", "action", "v_b", "w_b", "foot_z", "foot_f", "self_f", "root_pos", "root_quat", "grav_b",
                "done", "cmd", "reward", "yaw")
        log = {k: [] for k in keys}
        pending = []
        with torch.inference_mode():
            for _ in range(steps):
                hold_commands()
                pending.append(policy(obs))
                obs, _, dones, _ = env.step(pending.pop(0) if len(pending) > delay else torch.zeros_like(pending[0]))
                policy.reset(dones)
                d = robot.data
                log["q"].append(d.joint_pos.torch[:, servo].clone())
                log["play"].append(d.joint_pos.torch[:, play].clone())
                log["self_f"].append(self_contact.data.force_matrix_w.torch.norm(dim=-1).flatten(1))
                log["action"].append(base.action_manager.action.clone())
                log["v_b"].append(d.root_link_lin_vel_b.torch.clone())
                log["w_b"].append(d.root_link_ang_vel_b.torch.clone())
                log["foot_z"].append(d.body_link_pos_w.torch[:, feet, 2] - origins[:, 2:3])
                log["foot_f"].append(sensor.data.net_forces_w.torch[:, sensor_feet].norm(dim=-1))
                log["root_pos"].append(d.root_link_pos_w.torch - origins)
                log["root_quat"].append(d.root_link_quat_w.torch.clone())
                log["grav_b"].append(d.projected_gravity_b.torch.clone())
                log["done"].append(dones.clone())
                log["cmd"].append(base.command_manager.get_command("base_velocity").clone())
                log["reward"].append(rewards._step_reward.clone())
                heading = math_utils.quat_apply(d.root_link_quat_w.torch, forward)
                log["yaw"].append(torch.atan2(heading[:, 1], heading[:, 0]))
        S = {k: torch.stack(v).cpu().numpy() for k, v in log.items()}  # (T, env, ...)
        S["tau_sub"] = torch.stack(sub["tau"][-steps * base.cfg.decimation:]).cpu().numpy()
        S["dq_sub"] = torch.stack(sub["dq"][-steps * base.cfg.decimation:]).cpu().numpy()
        joints = [robot.joint_names[i] for i in servo]
        action_joints = list(base.action_manager.get_term("joint_pos")._joint_names)
        default_q = robot.data.default_joint_pos.torch[0, servo].cpu().numpy()

        if delay:
            print(f"\naction delay: {delay} policy steps ({delay * base.step_dt * 1000:.0f} ms)")
        if robust:
            report_robust(S, base.step_dt)
            return
        report(S, joints, action_joints, default_q, base.step_dt, base.physics_dt)
        report_distance(S, base.step_dt)
        report_rewards(S, rewards.active_terms, base.step_dt)
        pairs = [f"{b}/{f}" for b in self_contact.body_names for f in ("trunk", "ankle_right")]
        w = slice(int(WARMUP_S / base.step_dt), None)
        print("\nself-contact by pair (share of steps over 1 N, all commands): " +
              ", ".join(f"{n} {(S['self_f'][w, :, k] > CONTACT_N).mean():.1%}" for k, n in enumerate(pairs)))
        if delay:
            return  # a what-if run: keep the saved results and export from the undelayed run
        out = os.path.join(run_dir, "eval")
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, "eval.npz"), joints=np.array(joints), commands=np.array(list(COMMANDS)),
                            action_joints=np.array(action_joints), default_q=default_q, repeat=REPEAT,
                            dt=base.step_dt, physics_dt=base.physics_dt, **S)
        # link angles for the CAD check: servo plus gearbox play
        traj = {name: np.degrees(S["q"][:, i * REPEAT] + S["play"][:, i * REPEAT]) for i, name in enumerate(COMMANDS)}
        np.savez(os.path.join(out, "traj.npz"), joints=np.array(joints), **traj)
        export_dir = os.path.join(run_dir, "exported")
        runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
        print(f"\nsaved {out}/eval.npz, {out}/traj.npz; exported {export_dir}/policy.onnx")


def saturated(tau, dq):
    """True where the torque sits on the DC-motor limit: stall at rest, falling to zero at no-load speed."""
    stall, w0 = SERVO["stall_nm"], SERVO["no_load_rad_s"]
    hi = np.clip(stall * (1.0 - dq / w0), 0.0, stall)
    lo = np.clip(stall * (-1.0 - dq / w0), -stall, 0.0)
    return (tau >= hi - 0.02 * stall) | (tau <= lo + 0.02 * stall)


def report(S, joints, action_joints, default_q, dt, physics_dt):
    stall, w0 = SERVO["stall_nm"], SERVO["no_load_rad_s"]
    w = slice(int(WARMUP_S / dt), None)
    ws = slice(int(WARMUP_S / physics_dt), None)
    env_of = lambda i: slice(i * REPEAT, (i + 1) * REPEAT)  # noqa: E731

    print(f"\n1. tracking and posture ({REPEAT} robots per command, {SECONDS - WARMUP_S:.0f} s each)")
    print(f"{'command':10s} {'cmd vx/vy/wz':>17s} {'got vx/vy/wz':>17s}  fell  tilt  trunk  knee  sway  "
          "self-contact")
    knee = [joints.index("left_knee"), joints.index("right_knee")]
    for i, (name, c) in enumerate(COMMANDS.items()):
        e = env_of(i)
        v, wz = S["v_b"][w, e].mean((0, 1)), S["w_b"][w, e, 2].mean()
        g = S["grav_b"][w, e]
        tilt = np.degrees(np.arccos(np.clip(-g[..., 2], -1, 1)))
        fell = int(S["done"][:, e].sum())
        sway = np.degrees(np.abs(S["w_b"][w, e, :2]).mean())
        print(f"{name:10s} {c[0]:5.2f} {c[1]:5.2f} {c[2]:5.2f} {v[0]:5.2f} {v[1]:5.2f} {wz:5.2f}  {fell:4d} "
              f"{tilt.mean():4.1f}° {S['root_pos'][w, e, 2].mean() * 1000:4.0f}mm "
              f"{np.degrees(np.abs(S['q'][w, e][..., knee]).mean()):4.0f}° {sway:4.0f}°/s  "
              f"{(S['self_f'][w, e].max(-1) > CONTACT_N).mean():6.1%}")

    print("\n2. gait (contact = foot force > 1 N)")
    print(f"{'command':10s} steps/s  stance  both-down  both-up  swing peak (ankle above its stance height)")
    for i, name in enumerate(COMMANDS):
        e = env_of(i)
        con = S["foot_f"][w, e] > CONTACT_N  # (T, env, 2)
        touchdowns = (con[1:] & ~con[:-1]).sum() / (REPEAT * con.shape[0] * dt)
        fz = S["foot_z"][w, e]
        stance_h = np.median(fz[con]) if con.any() else np.nan
        peaks = []
        for r in range(REPEAT):
            for f in range(2):
                air, z = ~con[:, r, f], fz[:, r, f]
                edges = np.flatnonzero(np.diff(air.astype(int)))
                for a, b in zip(edges[:-1], edges[1:]):
                    if air[a + 1]:
                        peaks.append(z[a + 1:b + 1].max() - stance_h)
        peak = f"{np.median(peaks) * 1000:3.0f} mm (max {np.max(peaks) * 1000:3.0f})" if peaks else "no swings"
        print(f"{name:10s} {touchdowns:6.1f}  {con.mean():6.0%}  {con.all(-1).mean():8.0%} {(~con).all(-1).mean():8.0%}  "
              f"{peak}")

    print("\n3. action smoothness (target jump per 20 ms policy step, degrees) and oscillation")
    print(f"{'command':10s} mean jump  p99 jump  main freq of leg targets")
    legs = [k for k, n in enumerate(action_joints) if "hip" in n or "knee" in n or "ankle" in n]
    for i, name in enumerate(COMMANDS):
        a = S["action"][w, env_of(i)][..., legs]
        jump = np.degrees(np.abs(np.diff(a, axis=0)))
        spec = np.abs(np.fft.rfft(a - a.mean(0), axis=0)).mean((1, 2))
        freq = np.fft.rfftfreq(a.shape[0], dt)[spec.argmax()]
        print(f"{name:10s} {jump.mean():8.1f}° {np.percentile(jump, 99):8.1f}°  {freq:5.1f} Hz")

    tau, dq = S["tau_sub"][ws], S["dq_sub"][ws]
    sat = saturated(tau, dq)
    print(f"\n4. servo use at physics rate (19 kg: stall {stall} N·m, no-load {w0} rad/s); rms torque sets heating")
    names = list(COMMANDS)
    print(f"{'joint':16s} " + " ".join(f"{n[:9]:>9s}" for n in names) + "   | saturated  peak ω  mean |ω|")
    for j in np.argsort(-np.sqrt((tau ** 2).mean((0, 1)))):
        rms = [np.sqrt((tau[:, env_of(i), j] ** 2).mean()) / stall for i in range(len(names))]
        print(f"{joints[j]:16s} " + " ".join(f"{r:9.0%}" for r in rms) +
              f"   | {sat[..., j].mean():8.0%} {np.abs(dq[..., j]).max() / w0:6.0%} {np.abs(dq[..., j]).mean() / w0:6.0%}")
    rms_all = np.sqrt((tau ** 2).mean((0, 2))) / stall
    print("rms over all 14 servos, per command: " +
          ", ".join(f"{n} {rms_all[env_of(i)].mean():.0%}" for i, n in enumerate(names)))
    power = (np.abs(tau * dq)).sum(-1).mean(0)
    print("mean mechanical power, per command:  " +
          ", ".join(f"{n} {power[env_of(i)].mean():.1f} W" for i, n in enumerate(names)))

    print("\n5. mean joint offset from the stand pose (degrees, left leg / head), per command")
    show = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch"]
    print(f"{'':10s} " + " ".join(f"{s.replace('left_', ''):>10s}" for s in show))
    for i, name in enumerate(COMMANDS):
        off = np.degrees(S["q"][w, env_of(i)].mean((0, 1)) - default_q)
        print(f"{name:10s} " + " ".join(f"{off[joints.index(s)]:10.0f}" for s in show))


def report_distance(S, dt):
    """Where the robots got to, from their positions: the check that a policy is walking and not just scoring."""
    w0 = int(WARMUP_S / dt)
    pos, yaw, done = S["root_pos"][w0:, :, :2], S["yaw"][w0:], S["done"][w0:].astype(bool)
    step = pos[1:] - pos[:-1]
    c, s = np.cos(yaw[1:]), np.sin(yaw[1:])
    ahead, left = c * step[..., 0] + s * step[..., 1], -s * step[..., 0] + c * step[..., 1]
    turn = (yaw[1:] - yaw[:-1] + np.pi) % (2 * np.pi) - np.pi
    keep = ~done[1:]  # a robot that fell is put back at its start
    seconds = (len(pos) - 1) * dt
    print(f"\n6. distance covered in {seconds:.0f} s, from positions (asked -> covered, and the share of the command)")
    print(f"{'command':10s} {'ahead [m]':>16s} {'left [m]':>16s} {'turned [deg]':>16s}   share")
    for i, (name, cmd) in enumerate(COMMANDS.items()):
        e = slice(i * REPEAT, (i + 1) * REPEAT)
        got = [(x[:, e] * keep[:, e]).sum(0).mean() for x in (ahead, left, turn)]
        asked = [v * seconds for v in cmd]
        share = [f"{g / a:4.0%}" for g, a in zip(got, asked) if a]
        print(f"{name:10s} {asked[0]:6.2f} -> {got[0]:6.2f} {asked[1]:6.2f} -> {got[1]:6.2f} "
              f"{np.degrees(asked[2]):6.0f} -> {np.degrees(got[2]):6.0f}   {' '.join(share) or '-'}")


def report_rewards(S, terms, dt):
    """Reward each term pays per second at the final weights, per command: what the policy is choosing between."""
    w = slice(int(WARMUP_S / dt), None)
    rate = S["reward"][w].mean(axis=0).reshape(len(COMMANDS), REPEAT, -1).mean(axis=1)  # (command, term)
    print("\n7. reward per second at the final weights, per command")
    print(f"{'term':<22}" + "".join(f"{n:>10}" for n in COMMANDS))
    for k, term in enumerate(terms):
        if np.abs(rate[:, k]).max() >= 0.02:
            print(f"{term:<22}" + "".join(f"{v:>10.2f}" for v in rate[:, k]))
    print(f"{'total':<22}" + "".join(f"{v:>10.2f}" for v in rate.sum(axis=1)))


def report_robust(S, dt):
    w = slice(int(WARMUP_S / dt), None)
    n = S["done"].shape[1]
    resets = S["done"].sum(0)
    err_xy = np.linalg.norm(S["cmd"][w, :, :2] - S["v_b"][w, :, :2], axis=-1)
    err_wz = np.abs(S["cmd"][w, :, 2] - S["w_b"][w, :, 2])
    tilt = np.degrees(np.arccos(np.clip(-S["grav_b"][w, :, 2], -1, 1)))
    stall = SERVO["stall_nm"]
    rms = np.sqrt((S["tau_sub"] ** 2).mean()) / stall
    print(f"\nrobust run: {n} robots, {ROBUST_SECONDS:.0f} s, random commands, noise, randomized robot, pushes")
    print(f"  robots that fell at least once: {(resets > 0).sum()} / {n} ({(resets > 0).mean():.1%}), falls {resets.sum()}")
    print(f"  speed error xy: mean {err_xy.mean():.3f} m/s, p95 {np.percentile(err_xy, 95):.3f}")
    print(f"  turn error:     mean {err_wz.mean():.3f} rad/s, p95 {np.percentile(err_wz, 95):.3f}")
    print(f"  tilt: mean {tilt.mean():.1f}°, p99 {np.percentile(tilt, 99):.1f}°; trunk height {S['root_pos'][w, :, 2].mean() * 1000:.0f} mm")
    print(f"  rms servo torque: {rms:.0%} of stall")


if __name__ == "__main__":
    main(sys.argv[1:])
