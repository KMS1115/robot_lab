Place JiNi BeyondMimic motion files here.

Expected default file name:
- `jini_motion.npz`

Recommended generation flow:
1. Export a JiNi-retargeted CSV from Maya in the format
   `base_xyz, base_quat_xyzw, joint_pos[10]`
2. Convert it with `scripts/tools/beyondmimic/csv_to_npz_jini.py`
3. Replay it with `scripts/tools/beyondmimic/replay_npz_jini.py`
4. Train with `RobotLab-Isaac-BeyondMimic-Flat-JiNi-v0`
