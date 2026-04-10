"""JiNi humanoid motion data converter (CSV -> NPZ) for AMP."""

import os
import argparse
import numpy as np

# ====================================================================
# CONFIGURATION
# ====================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_JINI_DIR = os.path.join(SCRIPT_DIR, "source", "robot_lab", "data", "Robots", "jini")
DEFAULT_OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "source", "robot_lab", "robot_lab", "tasks", "direct", "jini_amp", "motions"
)

CSV_FILE = os.path.join(SCRIPT_DIR, "JiNi_moonwalk_30hz.csv")
URDF_FILE = os.path.join(REPO_JINI_DIR, "jini.urdf")
MESH_DIR = os.path.join(REPO_JINI_DIR, "meshes")
NPZ_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "jini_moonwalk_30hz.npz")

# Frame range (None = use all frames)
START_IDX = None
END_IDX = None

# Sampling rate of the CSV
FPS = 30


def parse_args():
    parser = argparse.ArgumentParser(description="Convert JiNi CSV motion data into AMP NPZ format.")
    parser.add_argument("--csv", default=CSV_FILE, help="Input CSV path.")
    parser.add_argument("--urdf", default=URDF_FILE, help="JiNi URDF path.")
    parser.add_argument("--mesh-dir", default=MESH_DIR, help="Mesh directory for the URDF.")
    parser.add_argument("--output", default=NPZ_FILE, help="Output NPZ path.")
    parser.add_argument("--fps", type=int, default=FPS, help="Sampling rate of the CSV file.")
    parser.add_argument("--start-idx", type=int, default=START_IDX, help="Optional start frame index (0-based).")
    parser.add_argument("--end-idx", type=int, default=END_IDX, help="Optional end frame index (exclusive).")
    return parser.parse_args()


# ====================================================================
# Quaternion utilities
# ====================================================================

def quaternion_inverse(q):
    """Inverse of quaternion (w, x, y, z)."""
    w, x, y, z = q
    norm_sq = w * w + x * x + y * y + z * z
    if norm_sq < 1e-8:
        norm_sq = 1e-8
    return np.array([w, -x, -y, -z], dtype=q.dtype) / norm_sq


def quaternion_multiply(q1, q2):
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=q1.dtype)


def compute_angular_velocity(q_prev, q_next, dt, eps=1e-8):
    """
    Compute angular velocity from adjacent quaternions (w, x, y, z).
    """
    q_inv = quaternion_inverse(q_prev)
    q_rel = quaternion_multiply(q_inv, q_next)
    norm_q_rel = np.linalg.norm(q_rel)
    if norm_q_rel < eps:
        return np.zeros(3, dtype=np.float32)
    q_rel /= norm_q_rel
    w = np.clip(q_rel[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(1.0 - w * w)
    if sin_half < eps:
        return np.zeros(3, dtype=np.float32)
    axis = q_rel[1:] / sin_half
    return (angle / dt) * axis


# ====================================================================
# Pinocchio FK
# ====================================================================

def build_pin_robot(urdf_path, mesh_dir):
    """Load URDF and construct a Pinocchio RobotWrapper with free-flyer."""
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise ImportError("pinocchio is required to convert JiNi AMP motion data.") from exc
    robot = pin.RobotWrapper.BuildFromURDF(
        urdf_path, mesh_dir, pin.JointModelFreeFlyer()
    )
    return robot


# ====================================================================
# Main
# ====================================================================

def main():
    args = parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    if not os.path.isfile(args.urdf):
        raise FileNotFoundError(f"URDF file not found: {args.urdf}")
    if not os.path.isdir(args.mesh_dir):
        raise FileNotFoundError(f"Mesh directory not found: {args.mesh_dir}")

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to run csv2npz_jini.py.") from exc

    try:
        import pinocchio as pin
    except ImportError as exc:
        raise ImportError("pinocchio is required to run csv2npz_jini.py.") from exc

    # 1. Read CSV data
    df = pd.read_csv(args.csv, header=None)

    if args.start_idx is not None or args.end_idx is not None:
        data_orig = df.iloc[args.start_idx:args.end_idx].to_numpy(dtype=np.float32)
    else:
        data_orig = df.to_numpy(dtype=np.float32)

    N = data_orig.shape[0]
    print(f"Loading CSV: {args.csv}")
    print(f"  Total {N} frames, {data_orig.shape[1]} columns")

    # Root and joint data
    root_data = data_orig[:, :7]   # (N, 7): x, y, z, qx, qy, qz, qw
    joint_data = data_orig[:, 7:]  # (N, 10)

    dt = 1.0 / args.fps

    # 2. Joint names (URDF joint order)
    joint_names = [
        "Left-Hip-Yaw",
        "Left-Hip-Roll",
        "Left-Hip-Pitch",
        "Left-Knee-Pitch",
        "Left-Ankle-Pitch",
        "Right-Hip-Yaw",
        "Right-Hip-Roll",
        "Right-Hip-Pitch",
        "Right-Knee-Pitch",
        "Right-Ankle-Pitch",
    ]
    dof_names = np.array(joint_names, dtype=np.str_)

    # 3. Joint positions
    dof_positions = joint_data.copy()  # (N, 10)

    # 4. Joint velocities (central difference)
    dof_velocities = np.zeros_like(dof_positions)
    dof_velocities[1:-1] = (dof_positions[2:] - dof_positions[:-2]) / (2 * dt)
    dof_velocities[0] = (dof_positions[1] - dof_positions[0]) / dt
    dof_velocities[-1] = (dof_positions[-1] - dof_positions[-2]) / dt

    # 5. Body link names for AMP observations
    body_names = [
        "base_link",
        "JiNi-Left-Link-1",
        "JiNi-Left-Link-2",
        "JiNi-Left-Link-3",
        "JiNi-Left-Link-4",
        "JiNi-Left-Link-5",
        "JiNi-Right-Link-1",
        "JiNi-Right-Link-2",
        "JiNi-Right-Link-3",
        "JiNi-Right-Link-4",
        "JiNi-Right-Link-5",
    ]
    body_names = np.array(body_names, dtype=np.str_)
    B = len(body_names)

    body_positions = np.zeros((N, B, 3), dtype=np.float32)
    body_rotations = np.zeros((N, B, 4), dtype=np.float32)

    # 6. Pinocchio forward kinematics
    robot = build_pin_robot(args.urdf, args.mesh_dir)
    model = robot.model
    data_pk = robot.data
    nq = model.nq

    print(f"  Pinocchio nq={nq}, CSV root(7) + joints({joint_data.shape[1]}) = {7 + joint_data.shape[1]}")

    q_pin = pin.neutral(model)

    for i in range(N):
        # Set root pose
        q_pin[0:3] = root_data[i, 0:3]       # position (x, y, z)
        q_pin[3:7] = root_data[i, 3:7]       # quaternion (qx, qy, qz, qw)
        # Set joint angles
        dofD = joint_data.shape[1]
        q_pin[7:7 + dofD] = joint_data[i, :]

        pin.forwardKinematics(model, data_pk, q_pin)
        pin.updateFramePlacements(model, data_pk)

        for j, link_name in enumerate(body_names):
            fid = model.getFrameId(link_name)
            link_tf = data_pk.oMf[fid]
            body_positions[i, j, :] = link_tf.translation
            quat_xyzw = pin.Quaternion(link_tf.rotation)
            # Store as (w, x, y, z)
            body_rotations[i, j, :] = np.array(
                [quat_xyzw.w, quat_xyzw.x, quat_xyzw.y, quat_xyzw.z],
                dtype=np.float32,
            )

    # 7. Body linear velocities (central difference)
    body_linear_velocities = np.zeros_like(body_positions)
    body_linear_velocities[1:-1] = (body_positions[2:] - body_positions[:-2]) / (2 * dt)
    body_linear_velocities[0] = (body_positions[1] - body_positions[0]) / dt
    body_linear_velocities[-1] = (body_positions[-1] - body_positions[-2]) / dt

    # 8. Body angular velocities
    body_angular_velocities = np.zeros((N, B, 3), dtype=np.float32)
    for j in range(B):
        quats = body_rotations[:, j, :]  # (N, 4) in (w, x, y, z)
        angular_vels = np.zeros((N, 3), dtype=np.float32)
        if N > 1:
            angular_vels[0] = compute_angular_velocity(quats[0], quats[1], dt)
            angular_vels[-1] = compute_angular_velocity(quats[-2], quats[-1], dt)
        for k in range(1, N - 1):
            av1 = compute_angular_velocity(quats[k - 1], quats[k], dt)
            av2 = compute_angular_velocity(quats[k], quats[k + 1], dt)
            angular_vels[k] = 0.5 * (av1 + av2)
        body_angular_velocities[:, j, :] = angular_vels

    # 9. Save NPZ
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    data_dict = {
        "fps": args.fps,
        "dof_names": dof_names,
        "body_names": body_names,
        "dof_positions": dof_positions,
        "dof_velocities": dof_velocities,
        "body_positions": body_positions,
        "body_rotations": body_rotations,
        "body_linear_velocities": body_linear_velocities,
        "body_angular_velocities": body_angular_velocities,
    }
    np.savez(args.output, **data_dict)

    print(f"\nConversion completed! Saved to {args.output}")
    print(f"  fps: {args.fps}")
    print(f"  dof_names: {dof_names.shape}")
    print(f"  body_names: {body_names.shape}")
    print(f"  dof_positions: {dof_positions.shape}")
    print(f"  dof_velocities: {dof_velocities.shape}")
    print(f"  body_positions: {body_positions.shape}")
    print(f"  body_rotations: {body_rotations.shape}")
    print(f"  body_linear_velocities: {body_linear_velocities.shape}")
    print(f"  body_angular_velocities: {body_angular_velocities.shape}")


if __name__ == "__main__":
    main()
