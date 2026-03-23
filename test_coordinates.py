import numpy as np

# Load camera data
camera_data = np.load('data/cameras/ARG_CRO_225412.npz')
R = camera_data['R'][0]
t = camera_data['t'][0]

# Camera center in world coords
camera_center = -R.T @ t

print("Camera position:", camera_center)
print(f"  X={camera_center[0]:.2f}, Y={camera_center[1]:.2f}, Z={camera_center[2]:.2f}")

# Load pitch points
pitch_points = []
with open('data/pitch_points.txt', 'r') as f:
    for line in f:
        x, y, z = line.strip().split()
        pitch_points.append([float(x), float(y), float(z)])
pitch_points = np.array(pitch_points)

print(f"\nField range:")
print(f"  X: {pitch_points[:, 0].min():.2f} to {pitch_points[:, 0].max():.2f}")
print(f"  Y: {pitch_points[:, 1].min():.2f} to {pitch_points[:, 1].max():.2f}")
print(f"  Z: {pitch_points[:, 2].min():.2f} to {pitch_points[:, 2].max():.2f}")

# Load a skeleton
skel_3d = np.load('data/skel_3d/ARG_CRO_225412.npy', allow_pickle=True)
print(f"\nFirst frame skeleton (camera coords):")
if len(skel_3d[0]) > 0:
    print(f"  Shape: {skel_3d[0][0].shape}")
    print(f"  Sample joint (pelvis #8): {skel_3d[0][0][8]}")
