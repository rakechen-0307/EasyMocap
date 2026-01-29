import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

# Standard SMPL Joint Names
SMPL_JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
    'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar',
    'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
    'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
]

# Standard SMPL Parent Indices
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# Mean SMPL Offsets (Male)
SMPL_OFFSETS = np.array([
    [0.0, 0.0, 0.0],          # Pelvis (Root)
    [0.05858135, -0.07485655, -0.01076985], # L_Hip
    [-0.06030973, -0.07470079, -0.01263084],# R_Hip
    [0.00443945, 0.13255003, -0.02336406],  # Spine1
    [0.04546878, -0.38555849, 0.01662963],  # L_Knee
    [-0.04634289, -0.39276251, 0.01633527], # R_Knee
    [0.00448844, 0.13459153, -0.01103504],  # Spine2
    [0.00979727, -0.42672321, -0.03810842], # L_Ankle
    [-0.01183556, -0.42878771, -0.03554174],# R_Ankle
    [0.00226459, 0.13054353, -0.02166699],  # Spine3
    [0.04660314, -0.07542279, 0.11749724],  # L_Foot
    [-0.04618765, -0.07062489, 0.11656886], # R_Foot
    [-0.00164673, 0.13783935, 0.01898734],  # Neck
    [0.08272535, 0.05886477, -0.01755913],  # L_Collar
    [-0.08182963, 0.06202167, -0.01777013], # R_Collar
    [0.00511599, 0.09349814, 0.0415392],    # Head
    [0.11470433, -0.02237064, -0.00898711], # L_Shoulder
    [-0.11905067, -0.02230303, -0.01358364],# R_Shoulder
    [0.26477611, -0.01323386, -0.02337774], # L_Elbow
    [-0.26622838, -0.01423405, -0.02989299],# R_Elbow
    [0.24522966, -0.01235123, -0.01189334], # L_Wrist
    [-0.25203794, -0.01309325, -0.01449627],# R_Wrist
    [0.08479366, -0.01041183, -0.01321451], # L_Hand
    [-0.08579707, -0.01026639, -0.01460677] # R_Hand
])

def write_bvh(filepath, motions, frame_time=0.033):
    poses = motions['poses'] # (N, 72)
    trans = motions['trans'] # (N, 3)
    n_frames = poses.shape[0]
    
    # Reshape to access joint 0
    poses_reshaped = poses.reshape(n_frames, 24, 3)
    
    # Create the correction rotation (Rx = 180 degrees)
    r_correction = R.from_euler('x', 180, degrees=True)
    
    # Apply correction to every frame of the Root (Index 0)
    for i in range(n_frames):
        # Current root rotation
        r_root = R.from_rotvec(poses_reshaped[i, 0])
        # Multiply: Correction * Original
        r_new = r_correction * r_root
        # Store back
        poses_reshaped[i, 0] = r_new.as_rotvec()

    # --- 3. Build Tree ---
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

    # --- 4. Write File ---
    with open(filepath, 'w') as f:
        f.write("HIERARCHY\n")
        
        def write_joint_hierarchy(joint_idx, level):
            indent = "\t" * level
            name = SMPL_JOINT_NAMES[joint_idx]
            offset = SMPL_OFFSETS[joint_idx]
            
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

def process_sequence(input_dir, output_file):
    from glob import glob
    
    # 1. Gather all JSON files
    files = sorted(glob(os.path.join(input_dir, '*.json')))
    if not files:
        print(f"No .json files found in {input_dir}")
        return

    print(f"Found {len(files)} frames. processing...")

    all_poses = []
    all_trans = []

    # 2. Iterate and Collect Data
    for filename in files:
        try:
            data = read_json(filename)
            # Handle list vs dict
            if isinstance(data, list):
                entry = data[0] # Assume one person per frame file
            elif isinstance(data, dict) and 'annots' in data:
                entry = data['annots'][0]
            else:
                continue

            # Extract Rh, Poses, Th
            rh = np.array(entry['Rh'])
            poses = np.array(entry['poses'])
            trans = np.array(entry['Th'])

            # Flatten and Merge
            # Logic: If pose is 69, add Rh. If 72, use as is.
            poses = poses.reshape(-1)
            rh = rh.reshape(-1)
            
            if poses.size == 69:
                full_pose = np.concatenate((rh, poses))
            elif poses.size == 72:
                full_pose = poses.copy()
                full_pose[:3] = rh 
            else:
                continue
            
            all_poses.append(full_pose)
            all_trans.append(trans.reshape(3))

        except Exception as e:
            print(f"Skipping frame {filename}: {e}")
            continue

    if not all_poses:
        print("No valid motion data extracted.")
        return

    # 3. Stack into single array (N, 72) and (N, 3)
    final_poses = np.vstack(all_poses)
    final_trans = np.vstack(all_trans)

    # 4. Write Single BVH
    write_bvh(output_file, {'poses': final_poses, 'trans': final_trans})

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Input directory containing numbered JSONs')
    parser.add_argument('--out', type=str, required=True, help='Output filename (e.g. anim.bvh)')
    args = parser.parse_args()

    process_sequence(args.path, args.out)

if __name__ == '__main__':
    main()