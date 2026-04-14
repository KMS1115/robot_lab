"""
JiNi One-Foot Stand Motion Data Converter (CSV -> NPZ)

Converts JiNi stand-with-one-foot CSV motion data to NPZ format for AMP training.

USAGE:
    python csv2npz_jini_stand.py

INPUT:
    - JiNi_stand_with_one_foot_30hz.csv (10 joint DOFs + root pose, 30fps)
    - JiNi URDF file

OUTPUT:
    - JiNi_stand_with_one_foot_30hz.npz for AMP (Adversarial Motion Priors)
"""

import os
import numpy as np
import pandas as pd
import pinocchio as pin

# ====================================================================
# CONFIGURATION
# ====================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(SCRIPT_DIR, "JiNi_stand_with_one_foot_30hz.csv")
URDF_FILE = os.path.join(SCRIPT_DIR, "JiNi-v1-20260331-foot-col-box.urdf")
MESH_DIR = os.path.join(SCRIPT_DIR, "jini-stl")
NPZ_FILE = os.path.join(SCRIPT_DIR, "JiNi_stand_with_one_foot_30hz.npz")

FPS = 30


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
    """Compute angular velocity from adjacent quaternions (w, x, y, z)."""
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
# Main
# ====================================================================

def main():
    df = pd.read_csv(CSV_FILE, header=None)
    data_orig = df.to_numpy(dtype=np.float32)

    N = data_orig.shape[0]
    print(f"Loading CSV: {CSV_FILE}")
    print(f"  Total {N} frames, {data_orig.shape[1]} columns")

    root_data = data_orig[:, :7]   # (N, 7): x, y, z, qx, qy, qz, qw
    joint_data = data_orig[:, 7:]  # (N, 10)

    dt = 1.0 / FPS

    joint_names = [
        "Left-Hip-Yaw", "Left-Hip-Roll", "Left-Hip-Pitch",
        "Left-Knee-Pitch", "Left-Ankle-Pitch",
        "Right-Hip-Yaw", "Right-Hip-Roll", "Right-Hip-Pitch",
        "Right-Knee-Pitch", "Right-Ankle-Pitch",
    ]
    dof_names = np.array(joint_names, dtype=np.str_)

    dof_positions = joint_data.copy()

    # Joint velocities (central difference)
    dof_velocities = np.zeros_like(dof_positions)
    dof_velocities[1:-1] = (dof_positions[2:] - dof_positions[:-2]) / (2 * dt)
    dof_velocities[0] = (dof_positions[1] - dof_positions[0]) / dt
    dof_velocities[-1] = (dof_positions[-1] - dof_positions[-2]) / dt

    body_names = [
        "base_link",
        "JiNi-Left-Link-1", "JiNi-Left-Link-2", "JiNi-Left-Link-3",
        "JiNi-Left-Link-4", "JiNi-Left-Link-5",
        "JiNi-Right-Link-1", "JiNi-Right-Link-2", "JiNi-Right-Link-3",
        "JiNi-Right-Link-4", "JiNi-Right-Link-5",
    ]
    body_names = np.array(body_names, dtype=np.str_)
    B = len(body_names)

    body_positions = np.zeros((N, B, 3), dtype=np.float32)
    body_rotations = np.zeros((N, B, 4), dtype=np.float32)

    # Pinocchio FK
    robot = pin.RobotWrapper.BuildFromURDF(
        URDF_FILE, MESH_DIR, pin.JointModelFreeFlyer()
    )
    model = robot.model
    data_pk = robot.data

    q_pin = pin.neutral(model)

    for i in range(N):
        q_pin[0:3] = root_data[i, 0:3]
        q_pin[3:7] = root_data[i, 3:7]
        q_pin[7:7 + joint_data.shape[1]] = joint_data[i, :]

        pin.forwardKinematics(model, data_pk, q_pin)
        pin.updateFramePlacements(model, data_pk)

        for j, link_name in enumerate(body_names):
            fid = model.getFrameId(link_name)
            link_tf = data_pk.oMf[fid]
            body_positions[i, j, :] = link_tf.translation
            quat_xyzw = pin.Quaternion(link_tf.rotation)
            body_rotations[i, j, :] = np.array(
                [quat_xyzw.w, quat_xyzw.x, quat_xyzw.y, quat_xyzw.z],
                dtype=np.float32,
            )

    # Body linear velocities (central difference)
    body_linear_velocities = np.zeros_like(body_positions)
    body_linear_velocities[1:-1] = (body_positions[2:] - body_positions[:-2]) / (2 * dt)
    body_linear_velocities[0] = (body_positions[1] - body_positions[0]) / dt
    body_linear_velocities[-1] = (body_positions[-1] - body_positions[-2]) / dt

    # Body angular velocities
    body_angular_velocities = np.zeros((N, B, 3), dtype=np.float32)
    for j in range(B):
        quats = body_rotations[:, j, :]
        angular_vels = np.zeros((N, 3), dtype=np.float32)
        if N > 1:
            angular_vels[0] = compute_angular_velocity(quats[0], quats[1], dt)
            angular_vels[-1] = compute_angular_velocity(quats[-2], quats[-1], dt)
        for k in range(1, N - 1):
            av1 = compute_angular_velocity(quats[k - 1], quats[k], dt)
            av2 = compute_angular_velocity(quats[k], quats[k + 1], dt)
            angular_vels[k] = 0.5 * (av1 + av2)
        body_angular_velocities[:, j, :] = angular_vels

    # Save NPZ
    data_dict = {
        "fps": FPS,
        "dof_names": dof_names,
        "body_names": body_names,
        "dof_positions": dof_positions,
        "dof_velocities": dof_velocities,
        "body_positions": body_positions,
        "body_rotations": body_rotations,
        "body_linear_velocities": body_linear_velocities,
        "body_angular_velocities": body_angular_velocities,
    }
    np.savez(NPZ_FILE, **data_dict)

    print(f"\nConversion completed! Saved to {NPZ_FILE}")
    print(f"  fps: {FPS}")
    print(f"  dof_positions: {dof_positions.shape}")
    print(f"  dof_velocities: {dof_velocities.shape}")
    print(f"  body_positions: {body_positions.shape}")
    print(f"  body_rotations: {body_rotations.shape}")
    print(f"  body_linear_velocities: {body_linear_velocities.shape}")
    print(f"  body_angular_velocities: {body_angular_velocities.shape}")


if __name__ == "__main__":
    main()
