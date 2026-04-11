"""JiNi humanoid motion data converter (CSV -> NPZ) for AMP."""

import os
import argparse
import xml.etree.ElementTree as ET
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
    parser.add_argument("--mesh-dir", default=MESH_DIR, help="Unused compatibility argument.")
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
# URDF FK
# ====================================================================

def parse_vec3(text):
    return np.fromstring(text, sep=" ", dtype=np.float32)


def rpy_to_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz @ ry @ rx


def axis_angle_to_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float32)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float32,
    )


def quat_xyzw_to_matrix(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def matrix_to_quat_wxyz(R):
    trace = np.trace(R)
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float32)
    quat /= max(np.linalg.norm(quat), 1e-8)
    return quat


def load_urdf_kinematic_tree(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joints_by_child = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        joints_by_child[child] = {
            "name": joint.attrib["name"],
            "parent": parent,
            "child": child,
            "origin_xyz": parse_vec3(origin.attrib.get("xyz", "0 0 0")),
            "origin_rpy": parse_vec3(origin.attrib.get("rpy", "0 0 0")),
            "axis": parse_vec3(axis.attrib.get("xyz", "0 0 1")) if axis is not None else np.array([0, 0, 1], dtype=np.float32),
        }
    return joints_by_child


def compute_link_transforms(root_pos, root_quat_xyzw, joint_positions, body_names, joints_by_child, joint_name_to_idx):
    transforms = {
        "base_link": {
            "R": quat_xyzw_to_matrix(root_quat_xyzw),
            "p": np.asarray(root_pos, dtype=np.float32),
        }
    }

    def get_transform(link_name):
        if link_name in transforms:
            return transforms[link_name]
        joint = joints_by_child[link_name]
        parent_tf = get_transform(joint["parent"])
        R_origin = rpy_to_matrix(joint["origin_rpy"])
        R_joint = axis_angle_to_matrix(joint["axis"], joint_positions[joint_name_to_idx[joint["name"]]])
        R_child = parent_tf["R"] @ R_origin @ R_joint
        p_child = parent_tf["p"] + parent_tf["R"] @ joint["origin_xyz"]
        transforms[link_name] = {"R": R_child, "p": p_child.astype(np.float32)}
        return transforms[link_name]

    body_positions = np.zeros((len(body_names), 3), dtype=np.float32)
    body_rotations = np.zeros((len(body_names), 4), dtype=np.float32)
    for i, body_name in enumerate(body_names):
        tf = get_transform(body_name)
        body_positions[i] = tf["p"]
        body_rotations[i] = matrix_to_quat_wxyz(tf["R"])
    return body_positions, body_rotations


# ====================================================================
# Main
# ====================================================================

def main():
    args = parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    if not os.path.isfile(args.urdf):
        raise FileNotFoundError(f"URDF file not found: {args.urdf}")

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to run csv2npz_jini.py.") from exc

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
        "Left_Hip_Yaw",
        "Left_Hip_Roll",
        "Left_Hip_Pitch",
        "Left_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Right_Hip_Yaw",
        "Right_Hip_Roll",
        "Right_Hip_Pitch",
        "Right_Knee_Pitch",
        "Right_Ankle_Pitch",
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
        "JiNi_Left_Link_1",
        "JiNi_Left_Link_2",
        "JiNi_Left_Link_3",
        "JiNi_Left_Link_4",
        "JiNi_Left_Link_5",
        "JiNi_Right_Link_1",
        "JiNi_Right_Link_2",
        "JiNi_Right_Link_3",
        "JiNi_Right_Link_4",
        "JiNi_Right_Link_5",
    ]
    body_names = np.array(body_names, dtype=np.str_)
    B = len(body_names)

    body_positions = np.zeros((N, B, 3), dtype=np.float32)
    body_rotations = np.zeros((N, B, 4), dtype=np.float32)

    # 6. URDF forward kinematics
    joints_by_child = load_urdf_kinematic_tree(args.urdf)
    joint_name_to_idx = {name: i for i, name in enumerate(joint_names)}

    for i in range(N):
        body_positions[i], body_rotations[i] = compute_link_transforms(
            root_pos=root_data[i, 0:3],
            root_quat_xyzw=root_data[i, 3:7],
            joint_positions=joint_data[i],
            body_names=body_names.tolist(),
            joints_by_child=joints_by_child,
            joint_name_to_idx=joint_name_to_idx,
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
