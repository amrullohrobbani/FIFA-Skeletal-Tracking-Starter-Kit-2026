"""
Interactive 3D visualization of skeleton tracking with playback controls.

Features:
- 3D view of skeletons in world coordinates
- Interactive camera rotation with mouse
- Frame playback controls (space=play/pause, left/right arrows=frame navigation)
- Soccer field visualization
- Camera position visualization

Controls:
- Mouse: Rotate view
- Mouse wheel: Zoom
- Space: Play/Pause
- Left/Right arrows: Previous/Next frame
- Up/Down arrows: Jump 10 frames
- 'r': Reset view
- 'q': Quit

Author: GitHub Copilot
Date: March 11, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse


# Skeleton connections for 25 keypoints (BODY25 format)
SKELETON_CONNECTIONS_25 = [
    (0, 1),   # Nose to Neck
    (1, 2),   # Neck to RShoulder
    (2, 3),   # RShoulder to RElbow
    (3, 4),   # RElbow to RWrist
    (1, 5),   # Neck to LShoulder
    (5, 6),   # LShoulder to LElbow
    (6, 7),   # LElbow to LWrist
    (1, 8),   # Neck to MidHip
    (8, 9),   # MidHip to RHip
    (9, 10),  # RHip to RKnee
    (10, 11), # RKnee to RAnkle
    (8, 12),  # MidHip to LHip
    (12, 13), # LHip to LKnee
    (13, 14), # LKnee to LAnkle
    (0, 15),  # Nose to REye
    (0, 16),  # Nose to LEye
    (15, 17), # REye to REar
    (16, 18), # LEye to LEar
    (14, 19), # LAnkle to LBigToe
    (19, 20), # LBigToe to LSmallToe
    (14, 21), # LAnkle to LHeel
    (11, 22), # RAnkle to RBigToe
    (22, 23), # RBigToe to RSmallToe
    (11, 24), # RAnkle to RHeel
]

# 15-joint skeleton connections (after OPENPOSE_TO_OURS mapping)
# OPENPOSE_TO_OURS = [0, 2, 5, 3, 6, 4, 7, 9, 12, 10, 13, 11, 14, 22, 19]
# Joints: [Nose, RShoulder, LShoulder, RElbow, LElbow, RWrist, LWrist, 
#          RHip, LHip, RKnee, LKnee, RAnkle, LAnkle, RBigToe, LBigToe]
SKELETON_CONNECTIONS_15 = [
    (0, 1),   # Nose to RShoulder
    (0, 2),   # Nose to LShoulder
    (1, 3),   # RShoulder to RElbow
    (3, 5),   # RElbow to RWrist
    (2, 4),   # LShoulder to LElbow
    (4, 6),   # LElbow to LWrist
    (1, 7),   # RShoulder to RHip
    (2, 8),   # LShoulder to LHip
    (7, 8),   # RHip to LHip
    (7, 9),   # RHip to RKnee
    (9, 11),  # RKnee to RAnkle
    (8, 10),  # LHip to LKnee
    (10, 12), # LKnee to LAnkle
    (11, 13), # RAnkle to RBigToe
    (12, 14), # LAnkle to LBigToe
]


def transform_camera_to_world(skel_3d_camera, R, t, skel_2d, K, dist_coeffs):
    """
    Transform skeleton from camera coordinates to world coordinates.
    Uses foot position ray-casting to find ground plane intersection.
    """
    
    def ray_from_xy(xy, K, R, t, k1, k2):
        """Cast ray from 2D pixel through camera."""
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_norm = (xy[0] - cx) / fx
        y_norm = (xy[1] - cy) / fy
        
        # Undistort
        r2 = x_norm**2 + y_norm**2
        distortion = 1 + k1 * r2 + k2 * r2**2
        x_undist = x_norm * distortion
        y_undist = y_norm * distortion
        
        # Ray in camera space
        ray_cam = np.array([x_undist, y_undist, 1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Transform to world space
        ray_world = R.T @ ray_cam
        origin_world = -R.T @ t
        
        return origin_world, ray_world
    
    def intersection_over_plane(origin, direction, plane_z=0.0):
        """Find intersection of ray with ground plane Z=plane_z."""
        if abs(direction[2]) < 1e-6:
            return None
        
        t_intersect = (plane_z - origin[2]) / direction[2]
        if t_intersect < 0:
            return None
        
        return origin + t_intersect * direction
    
    skel_3d_world = []
    
    for person_idx, (skeleton_cam, skeleton_2d) in enumerate(zip(skel_3d_camera, skel_2d)):
        if skeleton_cam.shape[0] == 0 or skeleton_cam.shape[1] != 3:
            skel_3d_world.append(skeleton_cam)
            continue
        
        # Find lowest visible keypoint (foot on ground)
        valid_2d = ~np.isnan(skeleton_2d).any(axis=1)
        if not valid_2d.any():
            skel_3d_world.append(skeleton_cam)
            continue
        
        # Get lowest point in image (highest Y coordinate)
        valid_y = skeleton_2d[valid_2d, 1]
        lowest_idx_in_valid = np.argmax(valid_y)
        lowest_idx = np.where(valid_2d)[0][lowest_idx_in_valid]
        
        # Cast ray from foot 2D position to find ground intersection
        foot_2d = skeleton_2d[lowest_idx, :2]
        k1, k2 = dist_coeffs[0], dist_coeffs[1] if len(dist_coeffs) > 1 else 0.0
        origin, direction = ray_from_xy(foot_2d, K, R, t, k1, k2)
        foot_3d_world = intersection_over_plane(origin, direction, plane_z=0.0)
        
        if foot_3d_world is None:
            skel_3d_world.append(skeleton_cam)
            continue
        
        # Transform skeleton: rotate and translate
        skeleton_world = skeleton_cam @ R
        skeleton_world -= skeleton_world[lowest_idx]
        skeleton_world += foot_3d_world
        
        skel_3d_world.append(skeleton_world)
    
    return skel_3d_world


def load_data(sequence_name, data_dir='data', submission_file='outputs/submission_full.npz'):
    """Load all necessary data for visualization."""
    data_path = Path(data_dir)
    
    # Try to load from submission file (world coordinates already computed)
    if Path(submission_file).exists():
        print(f"Loading world coordinates from {submission_file}")
        submission_data = np.load(submission_file)
        if sequence_name in submission_data:
            skel_3d_world = submission_data[sequence_name]  # Shape: (21, num_frames, 15, 3)
            # Convert from (people, frames, joints, 3) to (frames, people, joints, 3)
            # IMPORTANT: Preserve original person_idx (0-20) for consistent IDs across frames
            num_people, num_frames, num_joints, _ = skel_3d_world.shape
            skel_3d_world_frames = []
            for frame_idx in range(num_frames):
                frame_skels = []  # List of (original_person_idx, skeleton) tuples
                for person_idx in range(num_people):
                    skel = skel_3d_world[person_idx, frame_idx]
                    # Check if person is valid (not all zeros/nans)
                    if not np.all(skel == 0) and not np.all(np.isnan(skel)):
                        # Store as tuple: (original submission index, skeleton data)
                        frame_skels.append((person_idx, skel))
                skel_3d_world_frames.append(frame_skels)
            
            print(f"Loaded {num_frames} frames with world coordinates (15 joints)")
            use_world_coords = True
        else:
            print(f"Sequence {sequence_name} not found in submission file, will transform from camera coords")
            use_world_coords = False
    else:
        print(f"Submission file not found, will transform from camera coords")
        use_world_coords = False
    
    if not use_world_coords:
        # Load 3D skeletons (camera coordinates) - fallback
        skel_3d_path = data_path / 'skel_3d' / f'{sequence_name}.npy'
        skel_3d_world_frames = np.load(skel_3d_path, allow_pickle=True)
        
        # Load 2D skeletons (for reference)
        skel_2d_path = data_path / 'skel_2d' / f'{sequence_name}.npy'
        skel_2d = np.load(skel_2d_path, allow_pickle=True)
    else:
        # For world coords, we don't need 2D or camera params for transformation
        skel_2d = None
    
    # Load camera calibration (for camera position visualization)
    camera_path = data_path / 'cameras' / f'{sequence_name}.npz'
    camera_data = np.load(camera_path)
    K = camera_data['K']
    dist_coeffs = camera_data['k']
    R = camera_data['R']
    t = camera_data['t']
    
    # Load pitch points
    pitch_points_path = data_path / 'pitch_points.txt'
    pitch_points = []
    with open(pitch_points_path, 'r') as f:
        for line in f:
            x, y, z = line.strip().split()
            pitch_points.append([float(x), float(y), float(z)])
    pitch_points = np.array(pitch_points)
    
    return skel_3d_world_frames, skel_2d, K, dist_coeffs, R, t, pitch_points, use_world_coords


class Interactive3DVisualizer:
    def __init__(self, sequence_name, data_dir='data'):
        self.sequence_name = sequence_name
        self.data_dir = data_dir
        
        # Load data
        print(f"Loading data for sequence: {sequence_name}")
        (self.skel_3d_world, self.skel_2d, self.K, self.dist_coeffs, self.R, self.t, 
         self.pitch_points, self.use_world_coords) = load_data(sequence_name, data_dir)
        
        self.num_frames = len(self.skel_3d_world)
        self.current_frame = 0
        self.playing = False
        self.play_speed = 30  # FPS
        
        # Determine number of joints based on data
        if len(self.skel_3d_world[0]) > 0:
            # Extract skeleton from tuple (person_idx, skeleton)
            first_entry = self.skel_3d_world[0][0]
            if isinstance(first_entry, tuple):
                first_skel = first_entry[1]  # Get skeleton from (person_idx, skeleton)
            else:
                first_skel = first_entry  # Fallback for camera coords
            self.num_joints = first_skel.shape[0]
            self.skeleton_connections = SKELETON_CONNECTIONS_15 if self.num_joints == 15 else SKELETON_CONNECTIONS_25
            print(f"Using {self.num_joints}-joint skeleton")
        else:
            self.num_joints = 15
            self.skeleton_connections = SKELETON_CONNECTIONS_15
        
        # Setup plot
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Store zoom state to preserve between frames
        self.stored_xlim = None
        self.stored_ylim = None
        self.stored_zlim = None
        
        # Setup initial view
        self.setup_plot()
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Animation object (for play functionality)
        self.anim = None
        
        print("\nControls:")
        print("  Mouse: Rotate view")
        print("  Mouse wheel: Zoom in/out")
        print("  Right-click drag: Pan view")
        print("  Space: Play/Pause")
        print("  Left/Right arrows: Previous/Next frame")
        print("  Up/Down arrows: Jump 10 frames")
        print("  'r': Reset view")
        print("  'q': Quit")
        print(f"\nTotal frames: {self.num_frames}")
        
        # Debug: Check coordinate ranges
        R = self.R[0] if len(self.R.shape) > 2 else self.R
        t = self.t[0] if len(self.t.shape) > 1 else self.t
        camera_center = -R.T @ t
        print(f"\nCoordinate System Check:")
        print(f"  Camera position: X={camera_center[0]:.2f}, Y={camera_center[1]:.2f}, Z={camera_center[2]:.2f}")
        print(f"  Field X range: {self.pitch_points[:, 0].min():.2f} to {self.pitch_points[:, 0].max():.2f}")
        print(f"  Field Y range: {self.pitch_points[:, 1].min():.2f} to {self.pitch_points[:, 1].max():.2f}")
        print(f"  Field Z range: {self.pitch_points[:, 2].min():.2f} to {self.pitch_points[:, 2].max():.2f}")
        
        # Check first frame skeleton positions
        if len(self.skel_3d_world[0]) > 0:
            first_entry = self.skel_3d_world[0][0]
            if isinstance(first_entry, tuple):
                person_idx, first_skel = first_entry
                pelvis_idx = 7 if self.num_joints == 15 else 8
                if pelvis_idx < first_skel.shape[0]:
                    pelvis = first_skel[pelvis_idx]
                    print(f"  Sample player (frame 0, submission slot {person_idx}): X={pelvis[0]:.2f}, Y={pelvis[1]:.2f}, Z={pelvis[2]:.2f}")
            else:
                first_skel = first_entry
                pelvis_idx = 8
                if pelvis_idx < first_skel.shape[0]:
                    pelvis = first_skel[pelvis_idx]
                    print(f"  Sample player position (frame 0, person 0): X={pelvis[0]:.2f}, Y={pelvis[1]:.2f}, Z={pelvis[2]:.2f}")
    
    def setup_plot(self):
        """Initialize the 3D plot."""
        self.ax.clear()
        
        # Set labels (X=width, Y=length, Z=height)
        self.ax.set_xlabel('X (m) - Field Width')
        self.ax.set_ylabel('Y (m) - Field Length')
        self.ax.set_zlabel('Z (m) - Height')
        
        # Set aspect ratio - important for viewing
        self.ax.set_box_aspect([1.5, 1.0, 0.1])
        
        # Set view angle (initial camera position)
        # elev = elevation angle (higher = more from above)
        # azim = azimuth angle (rotation around vertical axis)
        self.ax.view_init(elev=20, azim=-60)
        
        self.update_frame()
    
    def draw_skeleton(self, skeleton, person_id, ax):
        """Draw a single skeleton in 3D."""
        if skeleton is None or skeleton.shape[0] == 0 or skeleton.shape[1] != 3:
            return
        
        # Color scheme for different players
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        color = colors[person_id % len(colors)]
        
        # Draw joints
        valid_joints = ~np.isnan(skeleton).any(axis=1)
        if valid_joints.any():
            ax.scatter(skeleton[valid_joints, 0], 
                      skeleton[valid_joints, 1], 
                      skeleton[valid_joints, 2],
                      c=color, s=20, alpha=0.9, edgecolors='black', linewidths=0.5)
        
        # Draw bones
        for connection in self.skeleton_connections:
            j1, j2 = connection
            if j1 < skeleton.shape[0] and j2 < skeleton.shape[0]:
                if not np.isnan(skeleton[j1]).any() and not np.isnan(skeleton[j2]).any():
                    xs = [skeleton[j1, 0], skeleton[j2, 0]]
                    ys = [skeleton[j1, 1], skeleton[j2, 1]]
                    zs = [skeleton[j1, 2], skeleton[j2, 2]]
                    ax.plot(xs, ys, zs, c=color, linewidth=3, alpha=0.8)
        
        # Add player number label at head position
        if valid_joints.any() and skeleton.shape[0] > 0:
            head_idx = 0  # Nose keypoint
            if head_idx < skeleton.shape[0] and not np.isnan(skeleton[head_idx]).any():
                ax.text(skeleton[head_idx, 0], skeleton[head_idx, 1] + 0.3, skeleton[head_idx, 2],
                       f'P{person_id}', fontsize=10, color=color, fontweight='bold')
    
    def draw_field(self, ax):
        """Draw the soccer field from pitch points."""
        if len(self.pitch_points) > 0:
            # Get field boundaries (X=width, Y=length, Z=height with Z=0 as ground)
            min_x, max_x = self.pitch_points[:, 0].min(), self.pitch_points[:, 0].max()
            min_y, max_y = self.pitch_points[:, 1].min(), self.pitch_points[:, 1].max()
            
            # Draw filled green ground plane at Z=0
            from matplotlib.patches import Rectangle
            from mpl_toolkits.mplot3d import art3d
            
            # Extend ground plane beyond field boundaries
            margin = 15  # meters beyond field
            ground_rect = Rectangle((min_x - margin, min_y - margin), 
                                     (max_x - min_x) + 2*margin, 
                                     (max_y - min_y) + 2*margin, 
                                     facecolor='green', alpha=0.4)
            ax.add_patch(ground_rect)
            art3d.pathpatch_2d_to_3d(ground_rect, z=0, zdir="z")
            
            # Draw brighter field surface on top
            field_rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                                   facecolor='limegreen', alpha=0.7, edgecolor='white', linewidth=3)
            ax.add_patch(field_rect)
            art3d.pathpatch_2d_to_3d(field_rect, z=0, zdir="z")
            
            # Draw field marking points (white dots)
            ax.scatter(self.pitch_points[::5, 0],
                      self.pitch_points[::5, 1],
                      self.pitch_points[::5, 2],
                      c='white', s=10, alpha=0.8, marker='.')
            
            # Draw center circle and lines (sample some pitch points)
            ax.scatter(self.pitch_points[:, 0],
                      self.pitch_points[:, 1],
                      self.pitch_points[:, 2],
                      c='white', s=1, alpha=0.5)
    
    def draw_camera(self, ax, frame_idx):
        """Draw camera position and orientation."""
        # Camera position in world coordinates (R and t are constant across frames)
        R = self.R[0] if len(self.R.shape) > 2 else self.R
        t = self.t[0] if len(self.t.shape) > 1 else self.t
        
        # Camera center in world coords: C = -R^T @ t
        camera_center = -R.T @ t
        
        # Draw camera position
        ax.scatter([camera_center[0]], [camera_center[1]], [camera_center[2]],
                  c='orange', s=100, marker='^', label='Camera')
        
        # Draw viewing direction (camera Z-axis in world coords)
        view_direction = R.T @ np.array([0, 0, 5])  # 5m forward
        view_end = camera_center + view_direction
        ax.plot([camera_center[0], view_end[0]],
               [camera_center[1], view_end[1]],
               [camera_center[2], view_end[2]],
               'orange', linewidth=2, alpha=0.7)
    
    def update_frame(self):
        """Update visualization for current frame."""
        # Store current zoom state before clearing
        if self.stored_xlim is not None:
            prev_xlim = self.stored_xlim
            prev_ylim = self.stored_ylim
            prev_zlim = self.stored_zlim
        else:
            prev_xlim = None
            prev_ylim = None
            prev_zlim = None
        
        self.ax.clear()
        
        # Set background color
        self.ax.set_facecolor('lightblue')
        self.fig.patch.set_facecolor('white')
        
        # Draw field first (so it's behind everything)
        self.draw_field(self.ax)
        
        # Draw skeletons
        frame_skeletons = self.skel_3d_world[self.current_frame]
        
        # Check if we have tuples (person_idx, skeleton) or just skeletons
        if len(frame_skeletons) > 0 and isinstance(frame_skeletons[0], tuple):
            # Using submission data with preserved person_idx
            for person_idx, skeleton in frame_skeletons:
                self.draw_skeleton(skeleton, person_idx, self.ax)
            num_players = len(frame_skeletons)
        else:
            # Using camera coords (fallback)
            for person_id, skeleton in enumerate(frame_skeletons):
                self.draw_skeleton(skeleton, person_id, self.ax)
            num_players = len([s for s in frame_skeletons if s is not None and s.shape[0] > 0])
        
        # Draw camera
        self.draw_camera(self.ax, self.current_frame)
        
        # Set title with player count
        self.ax.set_title(f'{self.sequence_name} - Frame {self.current_frame}/{self.num_frames-1} | Players: {num_players}', 
                         fontsize=14, fontweight='bold')
        
        # Hide axis labels and ticks
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')
        self.ax.set_zlabel('')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        
        # Set reasonable axis limits based on field size
        if len(self.pitch_points) > 0:
            margin = 10  # meters
            x_range = [self.pitch_points[:, 0].min() - margin, self.pitch_points[:, 0].max() + margin]
            y_range = [self.pitch_points[:, 1].min() - margin, self.pitch_points[:, 1].max() + margin]
            z_range = [-0.5, 4]  # Ground (Z=0) to 4m height (capture full body + some overhead)
            
            self.ax.set_xlim(x_range)
            self.ax.set_ylim(y_range)
            self.ax.set_zlim(z_range)
        
        # Set aspect ratio (X=width, Y=length, Z=height)
        # Soccer field is ~105m x 68m, Z is much smaller
        self.ax.set_box_aspect([1.5, 1.0, 0.1])
        
        # Remove grid and axis panes for cleaner look
        self.ax.grid(False)
        # Make axis panes invisible
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        # Make axis lines invisible
        self.ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        self.ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        self.ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # Restore zoom state if it was stored (preserve zoom between frames)
        if prev_xlim is not None:
            self.ax.set_xlim(prev_xlim)
            self.ax.set_ylim(prev_ylim)
            self.ax.set_zlim(prev_zlim)
        
        # Store current zoom state for next frame
        self.stored_xlim = self.ax.get_xlim()
        self.stored_ylim = self.ax.get_ylim()
        self.stored_zlim = self.ax.get_zlim()
        
        self.fig.canvas.draw_idle()
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == ' ':  # Space bar - play/pause
            self.toggle_play()
        elif event.key == 'left':  # Previous frame
            self.current_frame = max(0, self.current_frame - 1)
            self.update_frame()
        elif event.key == 'right':  # Next frame
            self.current_frame = min(self.num_frames - 1, self.current_frame + 1)
            self.update_frame()
        elif event.key == 'up':  # Jump forward 10 frames
            self.current_frame = min(self.num_frames - 1, self.current_frame + 10)
            self.update_frame()
        elif event.key == 'down':  # Jump backward 10 frames
            self.current_frame = max(0, self.current_frame - 10)
            self.update_frame()
        elif event.key == 'r':  # Reset view
            self.ax.view_init(elev=20, azim=-60)
            self.update_frame()
        elif event.key == 'q':  # Quit
            plt.close(self.fig)
    
    def on_scroll(self, event):
        """Handle mouse scroll for zoom."""
        if event.inaxes != self.ax:
            return
            
        # Get current axis limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()
        
       # Calculate center of view
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2
        
        # Calculate current ranges
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        z_range = zlim[1] - zlim[0]
        
        # Zoom factor
        if event.button == 'up':
            scale = 0.85  # Zoom in (smaller range)
        else:
            scale = 1.15  # Zoom out (larger range)
        
        # Apply zoom to all axes maintaining center
        new_x_range = x_range * scale
        new_y_range = y_range * scale
        new_z_range = z_range * scale
        
        self.ax.set_xlim(x_center - new_x_range / 2, x_center + new_x_range / 2)
        self.ax.set_ylim(y_center - new_y_range / 2, y_center + new_y_range / 2)
        self.ax.set_zlim(z_center - new_z_range / 2, z_center + new_z_range / 2)
        
        # Store zoom state for frame changes
        self.stored_xlim = self.ax.get_xlim()
        self.stored_ylim = self.ax.get_ylim()
        self.stored_zlim = self.ax.get_zlim()
        
        self.fig.canvas.draw_idle()
    
    def toggle_play(self):
        """Toggle play/pause."""
        if self.playing:
            # Stop playing
            self.playing = False
            if self.anim is not None:
                self.anim.event_source.stop()
                self.anim = None
        else:
            # Start playing
            self.playing = True
            self.anim = FuncAnimation(
                self.fig, 
                self.animate, 
                interval=1000/self.play_speed,  # milliseconds per frame
                blit=False
            )
    
    def animate(self, frame_num):
        """Animation function for playback."""
        if self.playing:
            self.current_frame = (self.current_frame + 1) % self.num_frames
            self.update_frame()
    
    def show(self):
        """Display the interactive visualization."""
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Interactive 3D skeleton visualization')
    parser.add_argument('--sequence', '-s', type=str, required=True,
                       help='Sequence name (e.g., ARG_CRO_225412)')
    parser.add_argument('--data_dir', '-d', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--frame', '-f', type=int, default=0,
                       help='Starting frame (default: 0)')
    
    args = parser.parse_args()
    
    # Create and show visualizer
    viz = Interactive3DVisualizer(args.sequence, args.data_dir)
    viz.current_frame = args.frame
    viz.update_frame()
    viz.show()


if __name__ == '__main__':
    main()
