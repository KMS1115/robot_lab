# robot_lab

## Openpath-monster

`Openpath-monster` is the public name to use for the Monster platform in external and company-facing documentation.

## Recommended Versions

- `robot_lab`: `v2.3.2`
- `Isaac Lab`: `v2.3.2`
- `Isaac Sim`: `5.1.0`
- `Python`: `3.11`
- `OS`: Ubuntu `22.04`

## Installation

Install `robot_lab` into the same Python environment where Isaac Lab is already installed:

```bash
git clone https://github.com/fan-ziqi/robot_lab.git
cd robot_lab
python -m pip install -e source/robot_lab
```

Optional check:

```bash
python scripts/tools/list_envs.py
```

## Training Conditions

- Confirm the Monster asset exists at `source/robot_lab/data/Robots/monster/monster.urdf`.
- Use the Isaac Lab Python environment that matches the versions above.
- Run training on Linux and prefer `--headless` for long jobs.
- Recommended order: train `flat` first, then train `rough`.

## Monster Tasks

- `RobotLab-Isaac-Velocity-Flat-Monster-v0`
- `RobotLab-Isaac-Velocity-Rough-Monster-v0`

## Recommended Commands

Train `Openpath-monster` on flat terrain first:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-Monster-v0 --headless
```

After the flat policy is stable, train the rough terrain task:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Rough-Monster-v0 --headless
```

Play a trained policy:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Monster-v0
```
