Place JiNi AMP motion files here.

Expected default file names:
- `jini_moonwalk_30hz.npz`
- `jini_stand_with_one_foot_30hz.npz`

Recommended generation flow:
1. Prepare `JiNi_moonwalk_30hz.csv` in the repository root.
2. Convert it with `python csv2npz_jini.py`.
3. Train with `RobotLab-Isaac-JiNi-AMP-Moonwalk-Direct-v0`.

Stand-with-one-foot flow:
1. Prepare `JiNi_stand_with_one_foot_30hz.csv` in the repository root.
2. Convert it with `python csv2npz_jini.py --csv JiNi_stand_with_one_foot_30hz.csv --output source/robot_lab/robot_lab/tasks/direct/jini_amp/motions/jini_stand_with_one_foot_30hz.npz`.
3. Train with `RobotLab-Isaac-JiNi-AMP-StandWithOneFoot-Direct-v0`.
