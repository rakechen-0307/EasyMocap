import os
import json
import torch
import smplx
import numpy as np
from scipy.spatial.transform import Rotation as R

# Standard SMPL Joint Names
SMPL_JOINT_NAMES = [
    'pelvis', 'thigh_l', 'thigh_r', 'stomach', 'calf_l', 'calf_r', 'diaphragm',
    'foot_l', 'foot_r', 'chest', 'toe_l', 'toe_r', 'neck', 'clavicle_l',
    'clavicle_r', 'head', 'arm_l', 'arm_r', 'forearm_l', 'forearm_r',
    'hand_l', 'hand_r', 'weapon_l', 'weapon_r'
]

# Standard SMPL Parent Indices
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

def calculate_custom_offsets(shapes, smpl_model_path, gender='male'):
    """
    Generates dynamic bone offsets based on the subject's shape parameters.
    """
    # Initialize the SMPL model
    smpl_model = smplx.create(smpl_model_path, model_type='smpl', gender=gender)
    
    # Format shapes for PyTorch
    betas = torch.tensor(shapes).float().unsqueeze(0)  # Shape: (1, 10)
    
    # Forward pass to get 3D joint locations in rest pose
    output = smpl_model(betas=betas, return_verts=False)
    
    # Extract the first 24 standard SMPL joints
    joints_3d = output.joints.detach().numpy()[0, :24, :]
    
    # Calculate relative offsets (Child - Parent)
    custom_offsets = np.zeros((24, 3))
    for i, p in enumerate(SMPL_PARENTS):
        if p == -1:
            custom_offsets[i] = [0.0, 0.0, 0.0]  # Root offset is 0 relative to itself in BVH
        else:
            custom_offsets[i] = joints_3d[i] - joints_3d[p]
            
    return custom_offsets

def write_bvh(filepath, motions, offsets, frame_time=0.033):
    poses = motions['poses'] # (N, 72)
    trans = motions['trans'] # (N, 3)
    n_frames = poses.shape[0]
    
    # Reshape to access joint 0
    poses_reshaped = poses.reshape(n_frames, 24, 3)
    
    # Create the correction rotation (Rx = 180 degrees)
    r_correction = R.from_euler('x', 180, degrees=True)
    
    # Apply correction to every frame of the Root (Index 0)
    for i in range(n_frames):
        r_root = R.from_rotvec(poses_reshaped[i, 0])
        r_new = r_correction * r_root
        poses_reshaped[i, 0] = r_new.as_rotvec()

    # Build Tree
    children = {i: [] for i in range(len(SMPL_JOINT_NAMES))}
    for i, p in enumerate(SMPL_PARENTS):
        if p != -1:
            children[p].append(i)

    dfs_order = []
    def get_dfs_order(joint_idx):
        dfs_order.append(joint_idx)
        for child_idx in children[joint_idx]:
            get_dfs_order(child_idx)
    get_dfs_order(0)

    # Write File
    with open(filepath, 'w') as f:
        f.write("HIERARCHY\n")
        
        def write_joint_hierarchy(joint_idx, level):
            indent = "\t" * level
            name = SMPL_JOINT_NAMES[joint_idx]
            offset = offsets[joint_idx]  # <-- USING CUSTOM OFFSETS HERE
            
            if level == 0:
                f.write(f"ROOT {name}\n")
            else:
                f.write(f"{indent}JOINT {name}\n")
            
            f.write(f"{indent}{{\n")
            f.write(f"{indent}\tOFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
            
            if level == 0:
                f.write(f"{indent}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
            else:
                f.write(f"{indent}\tCHANNELS 3 Zrotation Xrotation Yrotation\n")
            
            if len(children[joint_idx]) > 0:
                for child_idx in children[joint_idx]:
                    write_joint_hierarchy(child_idx, level + 1)
            else:
                f.write(f"{indent}\tEnd Site\n")
                f.write(f"{indent}\t{{\n")
                f.write(f"{indent}\t\tOFFSET 0.000000 0.000000 0.000000\n")
                f.write(f"{indent}\t}}\n")

            f.write(f"{indent}}}\n")

        write_joint_hierarchy(0, 0)

        f.write("MOTION\n")
        f.write(f"Frames: {n_frames}\n")
        f.write(f"Frame Time: {frame_time:.6f}\n")

        for frame_idx in range(n_frames):
            row_data = []
            
            # Translation
            t = trans[frame_idx]
            row_data.extend([t[0], t[1], t[2]])

            for joint_idx in dfs_order:
                rot_vec = poses_reshaped[frame_idx, joint_idx]
                r = R.from_rotvec(rot_vec)
                euler_deg = r.as_euler('zxy', degrees=True) 
                row_data.extend(euler_deg)
            
            f.write(" ".join(f"{x:.6f}" for x in row_data) + "\n")

    print(f"Saved Sequence BVH to {filepath}")

def read_json(path):
    with open(path) as f:
        return json.load(f)

def process_sequence(input_dir, output_file, smpl_model_path):
    from glob import glob
    
    files = sorted(glob(os.path.join(input_dir, '*.json')))
    if not files:
        print(f"No .json files found in {input_dir}")
        return

    print(f"Found {len(files)} frames. processing...")

    all_poses = []
    all_trans = []
    base_shapes = None

    for filename in files:
        try:
            data = read_json(filename)
            if isinstance(data, list):
                entry = data[0] 
            elif isinstance(data, dict) and 'annots' in data:
                entry = data['annots'][0]
            else:
                continue

            rh = np.array(entry['Rh']).reshape(-1)
            poses = np.array(entry['poses']).reshape(-1)
            trans = np.array(entry['Th']).reshape(-1)
            
            # Extract shapes only once (assuming same person across sequence)
            if base_shapes is None and 'shapes' in entry:
                base_shapes = np.array(entry['shapes']).reshape(-1)

            if poses.size == 69:
                full_pose = np.concatenate((rh, poses))
            elif poses.size == 72:
                full_pose = poses.copy()
                full_pose[:3] = rh 
            else:
                continue
            
            all_poses.append(full_pose)
            all_trans.append(trans)

        except Exception as e:
            print(f"Skipping frame {filename}: {e}")
            continue

    if not all_poses:
        print("No valid motion data extracted.")
        return
        
    if base_shapes is None:
        print("Warning: No 'shapes' parameter found. Make sure your JSON includes it.")
        return

    final_poses = np.vstack(all_poses)
    final_trans = np.vstack(all_trans)

    print("Generating custom skeleton offsets...")
    custom_offsets = calculate_custom_offsets(base_shapes, smpl_model_path)

    write_bvh(output_file, {'poses': final_poses, 'trans': final_trans}, custom_offsets)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Input directory containing numbered JSONs')
    parser.add_argument('--out', type=str, required=True, help='Output filename (e.g. anim.bvh)')
    parser.add_argument('--smpl_dir', type=str, required=True, help='Path to the directory containing SMPL models')
    args = parser.parse_args()

    process_sequence(args.path, args.out, args.smpl_dir)

if __name__ == '__main__':
    main()