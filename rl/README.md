# Training (Isaac Lab)

Isaac Lab tasks for the STS3215 Microduck. Physics is Newton / MuJoCo-Warp; Isaac Sim is not needed.

| Task | What it trains |
|---|---|
| `Microduck-STS3215-Velocity-Flat` | walk on flat ground following forward, sideways and turn commands |
| `Microduck-STS3215-Velocity-Flat-Turn` | same, with a wider turn range (±0.8 rad/s), little reversing and fewer standing episodes |

It starts from Isaac Lab's MicroDuck walking task (same observations, actions and rewards; the model uses the same
body and joint names) and changes what differs for this robot:

- **Robot model:** `sim/sts3215/robot_walk.xml`, converted to USD.
- **Servos:** Rhoban BAM's identified STS3215 model (`cad/sts3215/layout.json`, `servo.bam`), with the firmware's
  limit on how fast its target moves (5.1 rad/s), a 10–40 ms command delay, a 2S battery's voltage range and sag,
  ±0.5° gearbox play per joint, and the torque capped at the label's 1.91 N·m stall.
- **Collisions:** the soles against the floor, and the thighs against the trunk and hip yaw servos. A thigh touching
  the floor ends the episode (kneeling would otherwise collect the step-timing reward).
- **Rewards:** added penalties on servo torque, joint acceleration and hard landings, and rewards for trunk height,
  staying near the stand pose and keeping both feet down when told to stand.
- **Commands:** slower, with forward speed −0.15 to 0.25 m/s, sideways ±0.1 m/s and turning ±0.6 rad/s.
- **Gait:** foot lift and step timing scaled to the bigger robot.
- **Randomization:** servo friction, and trunk and head mass, since those are not measured yet.

## Setup (Linux, NVIDIA GPU)

Isaac Lab's environment needs the `importers` extra for the MJCF converter:

```bash
git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
```

```bash
cd ~/IsaacLab && uv sync --extra importers
```

Install this package into that environment:

```bash
uv pip install -e ~/microduck/rl --python ~/IsaacLab/.venv/bin/python
```

A plain `uv sync` removes the extra and this package again. Rerun both commands above after any sync.

After every model change (`cad/sts3215/build_mjcf.py`), convert the model to USD:

```bash
cd ~/microduck && ~/IsaacLab/.venv/bin/python rl/convert_usd.py
```

Check that it stands with the servos holding the default pose:

```bash
cd ~/microduck && ~/IsaacLab/.venv/bin/python rl/check_stand.py
```

## Train

Use tmux so the run survives a dropped connection:

```bash
tmux new -s md_train
```

```bash
cd ~/IsaacLab && .venv/bin/python scripts/reinforcement_learning/train.py --rl_library rsl_rl --task Microduck-STS3215-Velocity-Flat --num_envs 4096
```

On an RTX 4000 Ada this runs at about 108k steps/s, 0.9 s per iteration, so the default 8,000 iterations take
about 2 h. Checkpoints go to `~/IsaacLab/logs/rsl_rl/microduck_sts3215_flat/<run>/model_<iteration>.pt` every
250 iterations.

For a quick config check, add `--num_envs 64 --max_iterations 3`.

## Watch training

To follow a run in the Viser web viewer (port 8080), replay its newest checkpoint; newer checkpoints are swapped in
as training saves them, without reconnecting:

```bash
tmux new -d -s md_replay "bash ~/microduck/rl/watch_replay.sh"
```

Open `http://<box>:8080`, or tunnel it: `ssh -N -L 8080:127.0.0.1:8080 <box>`. It shares the GPU and slows training
by about 20 %. Stop it with `tmux kill-session -t md_replay`. To replay a given run, run `rl/live_replay.py` with
`--checkpoint <path to model_N.pt>` (see the command in the script).

## Evaluate and export

Run the trained policy at fixed commands (stand, walk, back, sidestep, turn, arc), headless:

```bash
cd ~/IsaacLab && .venv/bin/python ~/microduck/rl/eval_policy.py --task Microduck-STS3215-Velocity-Flat --checkpoint latest
```

It prints speed tracking, posture, gait timing, action smoothness and servo use against the torque-speed envelope.
Add `--robust` to run the training environment (random commands, noise, pushes) with 1,024 robots and count falls,
or `--delay N` to apply every action N policy steps (20 ms each) late. It saves `eval/eval.npz` and
`eval/traj.npz` in the run folder, and exports the policy to `exported/policy.onnx` (with `policy.onnx.data`) and
`policy.pt`. The training model only collides at the soles, so check the recorded motions against the CAD:

```bash
uv run --with python-fcl python cad/sts3215/check_rom.py --traj <run>/eval/traj.npz
```

## Turn variant

The first BAM policy walks forward but ignores turn commands. `Microduck-STS3215-Velocity-Flat-Turn` trains the same
task with a wider turn range (±0.8 rad/s, past the ±0.6 rad/s target), forward speed −0.05 to 0.25 m/s so that more
samples walk while turning, and 10 % standing episodes. Its logs go to `microduck_sts3215_flat_turn`. Train and
evaluate it with `--task Microduck-STS3215-Velocity-Flat-Turn` in the commands above. Its numbers are starting
values and have not been trained yet.
