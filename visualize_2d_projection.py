"""
Visualize 2D projections similar to main.py --visualize mode, but without requiring video/images.
Creates a blank canvas and draws:
- 3D pitch points projected to 2D
- Bounding boxes
- 2D skeleton keypoints and bones
- Camera frustum visualization

Note: On some systems, OpenCV VideoWriter may not work due to codec issues.
In that case, frames are saved as images, which can be converted to video with ffmpeg.

Author: GitHub Copilot
Date: March 10, 2026
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import subprocess
import sys


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

# Reduced connections for 15 keypoints
SKELETON_CONNECTIONS_15 = [
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
]

# OPENPOSE_TO_OURS mapping (15 keypoints used in main.py)
# our[i] corresponds to BODY25[ OPENPOSE_TO_OURS[i] ]
# our: 0=Nose, 1=RShoulder, 2=LShoulder, 3=RElbow, 4=LElbow,
#       5=RWrist, 6=LWrist, 7=RHip, 8=LHip, 9=RKnee,
#       10=LKnee, 11=RAnkle, 12=LAnkle, 13=RBigToe, 14=LBigToe
OPENPOSE_TO_OURS = [0, 2, 5, 3, 6, 4, 7, 9, 12, 10, 13, 11, 14, 22, 19]

# Connections for the 15-keypoint "ours" format used in results/submission files.
# Derived from SKELETON_CONNECTIONS_25 by mapping BODY25 indices → ours indices.
# Neck (BODY25[1]) and MidHip (BODY25[8]) are absent, so we bridge:
#   Neck    → RShoulder-LShoulder cross-bar + Nose-to-shoulders
#   MidHip  → RShoulder-RHip, LShoulder-LHip, RHip-LHip cross-bar
SKELETON_CONNECTIONS_RESULTS = [
    (0, 1),   # Nose to RShoulder
    (0, 2),   # Nose to LShoulder
    (1, 2),   # RShoulder to LShoulder
    (1, 3),   # RShoulder to RElbow
    (3, 5),   # RElbow to RWrist
    (2, 4),   # LShoulder to LElbow
    (4, 6),   # LElbow to LWrist
    (1, 7),   # RShoulder to RHip
    (2, 8),   # LShoulder to LHip
    (7, 8),   # RHip to LHip
    (7, 9),   # RHip to RKnee
    (9, 11),  # RKnee to RAnkle
    (11, 13), # RAnkle to RBigToe
    (8, 10),  # LHip to LKnee
    (10, 12), # LKnee to LAnkle
    (12, 14), # LAnkle to LBigToe
]


def create_video_from_images(image_dir, output_path, fps=25, cleanup_images=False):
    """
    Create video from a directory of images using ffmpeg.
    Falls back to pure Python method if ffmpeg is not available.
    
    Args:
        image_dir: Directory containing sequentially named images
        output_path: Output video file path
        fps: Frames per second
        cleanup_images: Delete images after creating video
    
    Returns:
        bool: True if successful, False otherwise
    """
    image_files = sorted(list(Path(image_dir).glob("frame_*.jpg")))
    if not image_files:
        print(f"No images found in {image_dir}")
        return False
    
    print(f"\nCreating video from {len(image_files)} images...")
    
    # Try ffmpeg first (best quality and speed)
    try:
        pattern = str(Path(image_dir) / "frame_%05d.jpg")
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-framerate', str(fps),
            '-i', pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # High quality
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Video created successfully: {output_path}")
            if cleanup_images:
                for img in image_files:
                    img.unlink()
                print(f"Cleaned up {len(image_files)} image files")
            return True
    except FileNotFoundError:
        print("ffmpeg not found, trying OpenCV...")
    except Exception as e:
        print(f"ffmpeg failed: {e}")
    
    # Fallback to OpenCV  
    try:
        first_img = cv2.imread(str(image_files[0]))
        H, W = first_img.shape[:2]
        
        # Try MJPG in AVI (most compatible on Windows)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        output_path = str(output_path).replace('.mp4', '.avi')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        
        if not writer.isOpened():
            print("OpenCV VideoWriter failed to open")
            return False
        
        for img_path in tqdm(image_files, desc="Writing video"):
            frame = cv2.imread(str(img_path))
            writer.write(frame)
        
        writer.release()
        print(f"Video created successfully: {output_path}")
        
        if cleanup_images:
            for img in image_files:
                img.unlink()
            print(f"Cleaned up {len(image_files)} image files")
        return True
    
    except Exception as e:
        print(f"OpenCV video creation failed: {e}")
        return False


def project_points(pts_3d, R, t, K, dist_coeffs):
    """Project 3D points to 2D image plane using camera parameters."""
    if len(pts_3d) == 0:
        return np.array([]).reshape(0, 2)
    
    rvec = cv2.Rodrigues(R)[0]
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, t, K, dist_coeffs)
    return pts_2d.reshape(-1, 2)


def draw_pitch_projection(canvas, pitch_points, R, t, K, dist_coeffs):
    """Draw 3D pitch points projected onto 2D canvas.
    
    This matches EXACTLY the logic in lib/camera_tracker.py draw_projection() method:
    - Uses cv2.projectPoints with Rodrigues rotation vector
    - Draws 5x5 yellow filled rectangles (center ± 2 pixels)
    - Clips coordinates to image bounds
    """
    # Project 3D points to 2D (identical to main.py)
    pts_2d = project_points(pitch_points, R, t, K, dist_coeffs)
    
    # Get max size in (width, height) order - matches main.py's vis.shape[1::-1]
    max_size = canvas.shape[1::-1]  # (W, H)
    
    for pt in pts_2d:
        # Check if point is valid and within canvas bounds
        valid = (pt >= 0).all() & (pt < max_size).all()
        if not valid:
            continue
        
        # Draw as small yellow filled rectangle (5x5 pixels: center ± 2)
        center = pt.astype(int)
        bl = (center - np.array([2, 2])).clip(min=0, max=max_size)
        tr = (center + np.array([2, 2])).clip(min=0, max=max_size)
        cv2.rectangle(canvas, tuple(bl), tuple(tr), (0, 255, 255), -1)
    
    return canvas


def draw_bounding_boxes(canvas, boxes):
    """Draw bounding boxes for all persons."""
    for person_idx, box in enumerate(boxes):
        if np.isnan(box).any():
            continue
        
        x1, y1, x2, y2 = box.astype(int)
        color = tuple(map(int, (np.random.RandomState(person_idx).rand(3) * 255).tolist()))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        
        # Draw person ID
        cv2.putText(canvas, f"P{person_idx}", (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return canvas


def draw_skeleton_2d(canvas, skel_2d, boxes, num_keypoints=25, kp15=False):
    """Draw 2D skeleton keypoints and connections.

    If kp15=True, only draw the 15 keypoints matching the submission format
    (selected via OPENPOSE_TO_OURS from the 25-keypoint data).
    """
    if kp15:
        connections = SKELETON_CONNECTIONS_RESULTS
    else:
        connections = SKELETON_CONNECTIONS_25 if num_keypoints == 25 else SKELETON_CONNECTIONS_15

    for person_idx, skeleton in enumerate(skel_2d):
        # Skip if no bounding box
        if np.isnan(boxes[person_idx]).any():
            continue

        # Generate consistent color per person
        color = tuple(map(int, (np.random.RandomState(person_idx).rand(3) * 200 + 55).tolist()))

        # Remap to 15-keypoint "ours" format if requested
        if kp15:
            skeleton = skeleton[OPENPOSE_TO_OURS]  # (15, 2)

        # Filter valid keypoints
        valid = ~np.isnan(skeleton).any(axis=1)

        # Draw bones first (so joints are on top)
        for connection in connections:
            if valid[connection[0]] and valid[connection[1]]:
                pt1 = tuple(skeleton[connection[0]].astype(int))
                pt2 = tuple(skeleton[connection[1]].astype(int))
                cv2.line(canvas, pt1, pt2, color, 2)

        # Draw joints
        for i, (x, y) in enumerate(skeleton):
            if valid[i]:
                cv2.circle(canvas, (int(x), int(y)), 4, color, -1)
                cv2.circle(canvas, (int(x), int(y)), 4, (255, 255, 255), 1)

    return canvas


def draw_skeleton_2d_ghost(canvas, skel_2d_frame, boxes_frame, opacity=1, num_keypoints=25, kp15=False):
    """Draw 2D skeleton from data at reduced opacity (ghost/reference layer).

    If kp15=True, only draw the 15 keypoints matching the submission format.
    """
    overlay = canvas.copy()
    if kp15:
        connections = SKELETON_CONNECTIONS_RESULTS
    else:
        connections = SKELETON_CONNECTIONS_25 if num_keypoints == 25 else SKELETON_CONNECTIONS_15

    for person_idx, skeleton in enumerate(skel_2d_frame):
        if np.isnan(boxes_frame[person_idx]).any():
            continue
        color = tuple(map(int, (np.random.RandomState(person_idx + 200).rand(3) * 150 + 50).tolist()))

        if kp15:
            skeleton = skeleton[OPENPOSE_TO_OURS]  # (15, 2)

        valid = ~np.isnan(skeleton).any(axis=1)

        for connection in connections:
            if valid[connection[0]] and valid[connection[1]]:
                pt1 = tuple(skeleton[connection[0]].astype(int))
                pt2 = tuple(skeleton[connection[1]].astype(int))
                cv2.line(overlay, pt1, pt2, color, 2)

        for i, (x, y) in enumerate(skeleton):
            if valid[i]:
                cv2.circle(overlay, (int(x), int(y)), 3, color, -1)

    cv2.addWeighted(overlay, opacity, canvas, 1 - opacity, 0, canvas)
    return canvas


def draw_skeleton_3d_projected(canvas, skel_3d, boxes, R, t, K, dist_coeffs, num_keypoints=25):
    """Draw 3D skeleton projected to 2D (like main.py does)."""
    connections = SKELETON_CONNECTIONS_25 if num_keypoints == 25 else SKELETON_CONNECTIONS_15
    
    for person_idx, skeleton_3d in enumerate(skel_3d):
        # Skip if no bounding box
        if np.isnan(boxes[person_idx]).any():
            continue
        
        # Filter valid keypoints
        valid = ~np.isnan(skeleton_3d).any(axis=1)
        if not valid.any():
            continue
        
        # Project 3D skeleton to 2D
        skeleton_2d = project_points(skeleton_3d, R, t, K, dist_coeffs)
        
        # Generate consistent color per person
        color = tuple(map(int, (np.random.RandomState(person_idx + 100).rand(3) * 200 + 55).tolist()))
        
        # Draw bones
        for connection in connections:
            if connection[0] < num_keypoints and connection[1] < num_keypoints:
                if valid[connection[0]] and valid[connection[1]]:
                    pt1 = tuple(skeleton_2d[connection[0]].astype(int))
                    pt2 = tuple(skeleton_2d[connection[1]].astype(int))
                    cv2.line(canvas, pt1, pt2, color, 3)
        
        # Draw joints
        for i, (x, y) in enumerate(skeleton_2d):
            if valid[i]:
                cv2.circle(canvas, (int(x), int(y)), 2, color, -1)
                cv2.circle(canvas, (int(x), int(y)), 2, (255, 255, 255), 1)
    
    return canvas


def _draw_dashed_line(canvas, pt1, pt2, color, thickness=2, dash_len=8, gap_len=5):
    """Draw a dashed line between pt1 and pt2."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length  # unit vector
    step = dash_len + gap_len
    pos = 0.0
    while pos < length:
        start = pos
        end = min(pos + dash_len, length)
        sx, sy = int(x1 + ux * start), int(y1 + uy * start)
        ex, ey = int(x1 + ux * end),   int(y1 + uy * end)
        cv2.line(canvas, (sx, sy), (ex, ey), color, thickness)
        pos += step


def draw_results_projected(canvas, results_frame, boxes_frame, R, t, K, dist_coeffs):
    """
    Draw results (world-space 3D) projected to 2D.

    Args:
        results_frame: (persons, 15, 3) in world coordinates from submission NPZ
    """
    H_c, W_c = canvas.shape[:2]

    for person_idx, skel_world in enumerate(results_frame):
        if person_idx >= len(boxes_frame) or np.isnan(boxes_frame[person_idx]).any():
            continue
        valid_3d = ~np.isnan(skel_world).any(axis=1)
        if not valid_3d.any():
            continue

        # Only project valid 3D points to avoid OpenCV producing garbage from NaN inputs
        skel_2d = np.full((15, 2), np.nan, dtype=np.float64)
        skel_2d[valid_3d] = project_points(skel_world[valid_3d], R, t, K, dist_coeffs)

        # Also mask out any points that projected outside the canvas
        in_bounds = (
            (skel_2d[:, 0] >= 0) & (skel_2d[:, 0] < W_c) &
            (skel_2d[:, 1] >= 0) & (skel_2d[:, 1] < H_c)
        )
        valid = valid_3d & in_bounds

        color = tuple(map(int, (np.random.RandomState(person_idx + 300).rand(3) * 200 + 55).tolist()))

        for connection in SKELETON_CONNECTIONS_RESULTS:
            if valid[connection[0]] and valid[connection[1]]:
                pt1 = tuple(skel_2d[connection[0]].astype(int))
                pt2 = tuple(skel_2d[connection[1]].astype(int))
                _draw_dashed_line(canvas, pt1, pt2, color, thickness=1, dash_len=2, gap_len=2)

        for i in range(15):
            if valid[i]:
                x, y = int(skel_2d[i, 0]), int(skel_2d[i, 1])
                cv2.circle(canvas, (x, y), 3, color, 1)  # ring with transparent centre

    return canvas


def draw_topdown_minimap(skel_3d_camera, boxes, camera_R, camera_t, pitch_points, 
                         skel_2d, K, dist_coeffs,
                         canvas_size=(400, 600), show_camera=True):
    """
    Draw top-down tactical view of the soccer field with player positions.
    
    Args:
        skel_3d_camera: 3D skeleton data in CAMERA coordinates (N_players, 15, 3)
        boxes: Bounding boxes for current frame (N_players, 4)
        camera_R: Camera rotation matrix (3, 3)
        camera_t: Camera translation vector (3,)
        pitch_points: All pitch marking points (714, 3)
        skel_2d: 2D skeleton keypoints (N_players, 15, 2+)
        K: Camera intrinsic matrix (3, 3)
        dist_coeffs: Distortion coefficients (5,)
        canvas_size: Canvas size (height, width) in pixels
        show_camera: Show camera position and viewing frustum
    
    Returns:
        Minimap canvas as numpy array
    """
    H, W = canvas_size
    minimap = np.ones((H, W, 3), dtype=np.uint8)  # BLACK background
    
    # Get actual field bounds from pitch_points
    world_min_x = pitch_points[:, 0].min()
    world_max_x = pitch_points[:, 0].max()
    world_min_y = pitch_points[:, 1].min()
    world_max_y = pitch_points[:, 1].max()
    
    # Add minimal margin for camera/players outside field
    margin = 3.0  # Minimal margin in meters
    world_min_x -= margin
    world_max_x += margin
    world_min_y -= margin
    world_max_y += margin
    
    # Scaling: world coordinates -> pixel coordinates (minimal padding)
    padding = 2  # Minimal pixel padding
    scale_x = (W - 2*padding) / (world_max_x - world_min_x)
    scale_y = (H - 2*padding) / (world_max_y - world_min_y)
    scale = min(scale_x, scale_y)  # Keep aspect ratio
    
    def world_to_pixel(x, y):
        """Convert world XY coordinates to pixel coordinates."""
        px = int((x - world_min_x) * scale + padding)
        py = int(H - padding - (y - world_min_y) * scale)  # Flip Y axis
        return px, py
    
    def ray_from_xy(xy, K, R, t, k1, k2):
        """Cast ray from 2D pixel through camera. Same as main.py."""
        # Normalize pixel coordinates
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_norm = (xy[0] - cx) / fx
        y_norm = (xy[1] - cy) / fy
        
        # Undistort (simplified radial distortion)
        r2 = x_norm**2 + y_norm**2
        distortion = 1 + k1 * r2 + k2 * r2**2
        x_undist = x_norm * distortion
        y_undist = y_norm * distortion
        
        # Ray direction in camera space
        ray_cam = np.array([x_undist, y_undist, 1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Transform to world space
        ray_world = camera_R.T @ ray_cam
        origin_world = -camera_R.T @ camera_t
        
        return origin_world, ray_world
    
    def intersection_over_plane(origin, direction, plane_z=0.0):
        """Find intersection of ray with plane Z=plane_z."""
        if abs(direction[2]) < 1e-6:
            return None  # Ray parallel to plane
        
        t_intersect = (plane_z - origin[2]) / direction[2]
        if t_intersect < 0:
            return None  # Intersection behind origin
        
        intersection = origin + t_intersect * direction
        return intersection
    
    # Transform camera-relative 3D to world coordinates
    # Strategy: Find foot position on ground, then place skeleton there
    skel_3d_world = []
    players_drawn = 0
    
    for person_idx, (skeleton_cam, skeleton_2d, box) in enumerate(zip(skel_3d_camera, skel_2d, boxes)):
        if np.isnan(box).any():
            skel_3d_world.append(None)
            continue
        
        # Find lowest visible keypoint (likely foot on ground)
        valid_2d = ~np.isnan(skeleton_2d).any(axis=1)
        if not valid_2d.any():
            skel_3d_world.append(None)
            continue
        
        # Get Y coordinates of valid points (higher Y = lower in image)
        valid_y = skeleton_2d[valid_2d, 1]
        lowest_idx_in_valid = np.argmax(valid_y)
        lowest_idx = np.where(valid_2d)[0][lowest_idx_in_valid]
        
        # Cast ray from 2D foot position to find ground intersection
        foot_2d = skeleton_2d[lowest_idx, :2]
        k1, k2 = dist_coeffs[0], dist_coeffs[1]
        origin, direction = ray_from_xy(foot_2d, K, camera_R, camera_t, k1, k2)
        foot_3d_world = intersection_over_plane(origin, direction, plane_z=0.0)
        
        if foot_3d_world is None:
            skel_3d_world.append(None)
            continue
        
        # Transform skeleton from camera coords to world coords
        # Rotate skeleton
        skeleton_world = skeleton_cam @ camera_R  # Each joint: (1,3) @ (3,3) -> (1,3)
        
        # Translate so lowest joint is at foot_3d_world
        skeleton_world -= skeleton_world[lowest_idx]
        skeleton_world += foot_3d_world
        
        skel_3d_world.append(skeleton_world)
    
    # Now skel_3d_world contains world coordinates
    
    # Draw green field background (fill the entire field area)
    field_corners = []
    for pt in pitch_points:
        px, py = world_to_pixel(pt[0], pt[1])
        if 0 <= px < W and 0 <= py < H:
            field_corners.append([px, py])
    
    if len(field_corners) > 3:
        # Create convex hull to fill field area
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(np.array(field_corners))
            hull_points = np.array(field_corners)[hull.vertices].astype(np.int32)
            cv2.fillPoly(minimap, [hull_points], (50, 200, 50))  # Bright green field
        except:
            # Fallback: just fill a rectangle
            xs = [p[0] for p in field_corners]
            ys = [p[1] for p in field_corners]
            cv2.rectangle(minimap, (min(xs), min(ys)), (max(xs), max(ys)), (50, 200, 50), -1)
    
    # Draw pitch markings from pitch_points (white dots)
    for pt in pitch_points[::2]:  # Every 2nd point to avoid overcrowding
        px, py = world_to_pixel(pt[0], pt[1])
        if 0 <= px < W and 0 <= py < H:
            cv2.circle(minimap, (px, py), 1, (255, 255, 255), -1)  # White dots
    
    # Draw players (using pelvis joint = index 8 in world coordinates)
    for person_idx, skeleton_world in enumerate(skel_3d_world):
        if skeleton_world is None:
            continue
        
        # Use pelvis joint (index 8) for player position
        pelvis = skeleton_world[8]
        if np.isnan(pelvis).any():
            continue
        
        px, py = world_to_pixel(pelvis[0], pelvis[1])
        
        # Generate consistent color per person
        color = tuple(map(int, (np.random.RandomState(person_idx).rand(3) * 200 + 55).tolist()))
        
        # Draw player as circle with outline
        if 0 <= px < W and 0 <= py < H:
            cv2.circle(minimap, (px, py), 8, color, -1)
            cv2.circle(minimap, (px, py), 8, (255, 255, 255), 2)
            
            # Draw player number
            cv2.putText(minimap, str(person_idx), (px-4, py+4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            players_drawn += 1
    
    # Draw camera position and viewing frustum
    if show_camera:
        # Camera position in world coords
        camera_pos = -camera_R.T @ camera_t
        cam_x, cam_y = camera_pos[0], camera_pos[1]
        cam_px, cam_py = world_to_pixel(cam_x, cam_y)
        
        if 0 <= cam_px < W and 0 <= cam_py < H:
            # Draw camera position
            cv2.circle(minimap, (cam_px, cam_py), 6, (0, 150, 255), -1)  # Orange
            cv2.circle(minimap, (cam_px, cam_py), 6, (255, 255, 255), 2)
            
            # Draw viewing direction (simplified frustum)
            # Camera looks in -Z direction in camera space
            view_dir = camera_R.T @ np.array([0, 0, -1])  # Transform to world space
            view_length = 15.0  # meters
            
            view_end_x = cam_x + view_dir[0] * view_length
            view_end_y = cam_y + view_dir[1] * view_length
            view_end_px, view_end_py = world_to_pixel(view_end_x, view_end_y)
            
            # Draw view direction arrow
            cv2.arrowedLine(minimap, (cam_px, cam_py), (view_end_px, view_end_py),
                          (0, 150, 255), 2, tipLength=0.3)
    
    return minimap


def create_visualization(sequence_name, frame_idx=0, show_pitch=True, show_boxes=True,
                        show_2d_skeleton=True, show_3d_projection=False,
                        data_cache=None, use_video=False, video_cap=None, show_minimap=False,
                        results_data=None, show_results=False, show_ghost=False, ghost_opacity=0.4,
                        kp15=False):
    """
    Create visualization similar to main.py --visualize mode.
    
    Args:
        sequence_name: Name of the sequence
        frame_idx: Frame to visualize
        show_pitch: Show pitch point projections
        show_boxes: Show bounding boxes
        show_2d_skeleton: Show 2D skeleton keypoints
        show_3d_projection: Show 3D skeleton projected to 2D
        data_cache: Pre-loaded data dict (for efficiency)
        use_video: Use actual video/images as background
        video_cap: Pre-opened cv2.VideoCapture object (for efficiency)
        show_minimap: Show top-down tactical minimap (picture-in-picture)
    """
    root = Path("data/")
    
    # Load data (or use cache)
    if data_cache is None:
        root = Path("data/")
        cameras = dict(np.load(root / "cameras" / f"{sequence_name}.npz"))
        boxes = np.load(root / "boxes" / f"{sequence_name}.npy")
        skel_2d = np.load(root / "skel_2d" / f"{sequence_name}.npy")
        skel_3d = np.load(root / "skel_3d" / f"{sequence_name}.npy")
        pitch_points = np.loadtxt(root / "pitch_points.txt")
        
        # Check if tracked camera data exists (from main.py --export_camera)
        tracked_camera_path = Path("outputs/calibration") / f"{sequence_name}.npz"
        if tracked_camera_path.exists():
            tracked_cameras = dict(np.load(tracked_camera_path))
            cameras['R'] = tracked_cameras['R']
            cameras['t'] = tracked_cameras['t']
            has_tracked_camera = True
            print(f"[OK] Loaded tracked camera poses from: {tracked_camera_path}")
        else:
            has_tracked_camera = False
            print(f"[WARNING] No tracked camera data found!")
            print(f"   Looking for: {tracked_camera_path}")
            print(f"   Using initial pose (frame 0) for all frames.")
            print(f"   This will only be accurate for frame 0!")
            print()
            print(f"   To generate tracked camera data:")
            print(f"   1. Obtain video files (data/videos/{sequence_name}.mp4)")
            print(f"   2. Run: python main.py --export_camera -s data/sequences_val.txt")
            print()
    else:
        cameras = data_cache['cameras']
        boxes = data_cache['boxes']
        skel_2d = data_cache['skel_2d']
        skel_3d = data_cache['skel_3d']
        pitch_points = data_cache['pitch_points']
        has_tracked_camera = cameras['R'].shape[0] > 1
    
    # Get frame data
    K = cameras['K'][frame_idx]
    dist_coeffs = cameras['k'][frame_idx]
    
    # Get camera pose for this frame
    if cameras['R'].shape[0] > 1:
        # Per-frame tracked camera data
        R = cameras['R'][frame_idx]
        t = cameras['t'][frame_idx]
    else:
        # Only initial pose available
        R = cameras['R'][0]
        t = cameras['t'][0]
    
    # Estimate canvas size from intrinsics (image dimensions)
    cx, cy = K[0, 2], K[1, 2]
    W, H = int(cx * 2), int(cy * 2)
    
    # Try to load actual video frame or image
    if use_video:
        # Try to load from images folder first
        image_path = Path("data/images") / sequence_name / f"{frame_idx:05d}.jpg"
        if not image_path.exists():
            image_path = image_path.with_suffix('.png')
        
        if image_path.exists():
            canvas = cv2.imread(str(image_path))
            if canvas is None:
                print(f"Warning: Failed to load image {image_path}")
                canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
        elif video_cap is not None:
            # Use video capture
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, canvas = video_cap.read()
            if not ret:
                print(f"Warning: Failed to read frame {frame_idx} from video")
                canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
        else:
            # Try to open video
            video_path = Path("data/videos") / f"{sequence_name}.mp4"
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, canvas = cap.read()
                cap.release()
                if not ret:
                    print(f"Warning: Failed to read frame {frame_idx} from video")
                    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
            else:
                canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    else:
        # Create blank canvas (white background for better visibility)
        canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
        
        # Draw grid for reference
        grid_color = (240, 240, 240)
        for i in range(0, W, 100):
            cv2.line(canvas, (i, 0), (i, H), grid_color, 1)
        for i in range(0, H, 100):
            cv2.line(canvas, (0, i), (W, i), grid_color, 1)
    
    # Draw pitch projection (yellow rectangles)
    if show_pitch:
        # Count visible points before drawing
        pts_2d_test = project_points(pitch_points, R, t, K, dist_coeffs)
        max_size = canvas.shape[1::-1]
        visible_count = ((pts_2d_test >= 0).all(axis=1) & (pts_2d_test < max_size).all(axis=1)).sum()
        
        canvas = draw_pitch_projection(canvas, pitch_points, R, t, K, dist_coeffs)
    else:
        visible_count = 0
    
    # Draw bounding boxes
    # Draw bounding boxes
    if show_boxes:
        canvas = draw_bounding_boxes(canvas, boxes[frame_idx])
    
    # Draw 2D data skeleton: ghost (low opacity) in compare mode, full opacity otherwise
    if show_ghost:
        canvas = draw_skeleton_2d_ghost(canvas, skel_2d[frame_idx], boxes[frame_idx],
                                        opacity=ghost_opacity, kp15=kp15)
    elif show_2d_skeleton and not show_results:
        canvas = draw_skeleton_2d(canvas, skel_2d[frame_idx], boxes[frame_idx], kp15=kp15)

    # Draw 3D skeletons projected to 2D (predicted positions)
    if show_3d_projection:
        canvas = draw_skeleton_3d_projected(canvas, skel_3d[frame_idx], boxes[frame_idx],
                                            R, t, K, dist_coeffs)

    # Compare mode: project results (world-space) to 2D
    if show_results and results_data is not None:
        results_frame = results_data[:, frame_idx, :, :]  # (persons, 15, 3)
        canvas = draw_results_projected(canvas, results_frame, boxes[frame_idx],
                                        R, t, K, dist_coeffs)

    # Add info text
    info_text = f"Sequence: {sequence_name} | Frame: {frame_idx}/{len(boxes)-1}"
    cv2.putText(canvas, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (0, 0, 0), 2)
    
    # Add camera tracking status
    if has_tracked_camera:
        camera_status = f"Camera: Tracked (per-frame)"
        color = (0, 150, 0)  # Green
    else:
        camera_status = f"Camera: Static (Frame 0 pose) - INACCURATE!"
        color = (0, 0, 255)  # Red
    cv2.putText(canvas, camera_status, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add pitch visibility info
    if show_pitch:
        pitch_info = f"Pitch points visible: {visible_count}/{len(pitch_points)} ({100*visible_count/len(pitch_points):.1f}%)"
        cv2.putText(canvas, pitch_info, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Legend for results / compare mode
    if show_results:
        legend_y = H - 40
        if show_ghost:
            cv2.putText(canvas, f"Ghost (opacity={ghost_opacity:.2f}): data 2D skeleton",
                        (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
            legend_y += 20
        cv2.putText(canvas, "Solid: results projected from world 3D",
                    (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    # Add minimap (picture-in-picture)
    if show_minimap:
        # Create minimap with world coordinate transformation
        minimap_height = int(canvas.shape[0] * 0.6)  # 60% of canvas height
        minimap_width = int(minimap_height * 0.67)     # Maintain field aspect ratio
        minimap = draw_topdown_minimap(
            skel_3d[frame_idx], boxes[frame_idx],
            R, t, pitch_points,
            skel_2d[frame_idx], K, dist_coeffs,
            canvas_size=(minimap_height, minimap_width),
            show_camera=True
        )
        
        # Position in center-bottom with margin
        margin = 20
        y_start = canvas.shape[0] - minimap_height - margin
        x_start = (canvas.shape[1] - minimap_width) // 2
        y_end = y_start + minimap_height
        x_end = x_start + minimap_width
        
        # Ensure minimap fits in canvas
        if x_start >= 0 and y_end <= canvas.shape[0]:
            # Create mask for non-black pixels (field and elements only)
            # Black background (0,0,0) becomes transparent
            minimap_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(minimap_gray, 10, 255, cv2.THRESH_BINARY)
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            
            # Blend minimap onto canvas (only non-black parts)
            roi = canvas[y_start:y_end, x_start:x_end]
            blended = (minimap * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
            canvas[y_start:y_end, x_start:x_end] = blended
    
    return canvas


def visualize_sequence(sequence_name, start_frame=0, num_frames=None,
                      save_video=False, save_images=False, show_live=True,
                      show_pitch=True, show_boxes=True, show_2d_skeleton=True,
                      show_3d_projection=False, use_video=False, show_minimap=False,
                      results_file=None, show_results=False, show_ghost=False, ghost_opacity=0.4,
                      kp15=False, save_png=False):
    """
    Visualize multiple frames of a sequence.
    
    Args:
        sequence_name: Name of the sequence
        start_frame: Starting frame index
        num_frames: Number of frames to process (None = all)
        save_video: Save as MP4 video
        save_images: Save individual frame images
        show_live: Show live OpenCV window
        show_pitch: Show pitch point projections
        show_boxes: Show bounding boxes
        show_2d_skeleton: Show 2D skeleton keypoints
        show_3d_projection: Show 3D skeleton projected to 2D
        show_minimap: Show top-down tactical minimap (picture-in-picture)
    """
    root = Path("data/")
    
    # Load data once
    print(f"Loading sequence: {sequence_name}")
    cameras = dict(np.load(root / "cameras" / f"{sequence_name}.npz"))
    
    # Check if tracked camera data exists (from main.py --export_camera)
    tracked_camera_path = Path("outputs/calibration") / f"{sequence_name}.npz"
    if tracked_camera_path.exists():
        tracked_cameras = dict(np.load(tracked_camera_path))
        # Replace R and t with tracked versions (per-frame)
        cameras['R'] = tracked_cameras['R']
        cameras['t'] = tracked_cameras['t']
        print(f"✓ Using tracked camera poses (per-frame)")
    else:
        print(f"⚠ Warning: Using initial camera pose only (static)")
        print(f"  For accurate visualization, run: python main.py -s data/sequences_val.txt -o outputs/test.npz --export_camera")
    
    data_cache = {
        'cameras': cameras,
        'boxes': np.load(root / "boxes" / f"{sequence_name}.npy"),
        'skel_2d': np.load(root / "skel_2d" / f"{sequence_name}.npy"),
        'skel_3d': np.load(root / "skel_3d" / f"{sequence_name}.npy"),
        'pitch_points': np.loadtxt(root / "pitch_points.txt")
    }

    # Load results file for compare mode
    results_data = None
    if results_file is not None:
        results_npz = dict(np.load(results_file, allow_pickle=False))
        if sequence_name in results_npz:
            results_data = results_npz[sequence_name]  # (persons, frames, 15, 3)
            print(f"✓ Loaded results: shape={results_data.shape}")
        else:
            print(f"⚠ Sequence '{sequence_name}' not found in results file: {results_file}")
            print(f"  Available keys: {list(results_npz.keys())[:5]}")

    total_frames = len(data_cache['boxes'])
    print(f"Total frames: {total_frames}")
    
    if num_frames is None:
        num_frames = total_frames - start_frame
    
    end_frame = min(start_frame + num_frames, total_frames)
    
    print(f"Visualizing frames {start_frame}-{end_frame}")
    
    # Open video capture if using video background
    video_cap = None
    if use_video:
        # Check if images exist
        image_path = Path("data/images") / sequence_name / f"{start_frame:05d}.jpg"
        if not image_path.exists():
            image_path = image_path.with_suffix('.png')
        
        if image_path.exists():
            print(f"✓ Using images from: data/images/{sequence_name}/")
        else:
            # Try to open video
            video_path = Path("data/videos") / f"{sequence_name}.mp4"
            if video_path.exists():
                video_cap = cv2.VideoCapture(str(video_path))
                if video_cap.isOpened():
                    print(f"✓ Using video from: {video_path}")
                else:
                    print(f"⚠ Warning: Failed to open video {video_path}")
                    use_video = False
            else:
                print(f"⚠ Warning: No images or video found. Using blank canvas.")
                use_video = False
    
    # Setup output
    video_writer = None
    frame_count = 0
    video_requested = save_video  # Track if video was originally requested
    if save_video or save_images:
        output_dir = Path("outputs") / sequence_name
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-create video writer if needed
    if save_video:
        # Always save images as backup/fallback
        save_images = True
        
        # Generate first frame to get canvas size
        first_canvas = create_visualization(
            sequence_name, start_frame, show_pitch, show_boxes,
            show_2d_skeleton, show_3d_projection, data_cache=data_cache,
            use_video=use_video, video_cap=video_cap, show_minimap=show_minimap,
            results_data=results_data, show_results=show_results, show_ghost=show_ghost, ghost_opacity=ghost_opacity,
            kp15=kp15
        )
        H, W = first_canvas.shape[:2]
        
        # For Windows, MJPG codec in AVI container is most reliable
        video_path = output_dir / f"{sequence_name}_visualization.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 25.0, (W, H))
        
        if video_writer.isOpened():
            # Write first frame
            if video_writer.write(first_canvas):
                frame_count = 1
                print(f"Saving video to: {video_path}")
            else:
                video_writer.release()
                video_writer = None
                print("Warning: Video writer failed. Will create video from images after processing.")
                save_video = False  # Will try ffmpeg method later
        else:
            video_writer = None
            print("Warning: Video writer failed to initialize. Will create video from images after processing.")
            save_video = False  # Will try ffmpeg method later
    
    # Process remaining frames (skip first if already written to video)
    start_idx = start_frame + (1 if video_writer is not None and frame_count > 0 else 0)
    for frame_idx in tqdm(range(start_idx, end_frame), desc="Processing frames"):
        canvas = create_visualization(
            sequence_name, frame_idx, show_pitch, show_boxes,
            show_2d_skeleton, show_3d_projection, data_cache=data_cache,
            use_video=use_video, video_cap=video_cap, show_minimap=show_minimap,
            results_data=results_data, show_results=show_results, show_ghost=show_ghost, ghost_opacity=ghost_opacity,
            kp15=kp15
        )
        
        # Save frame to video
        if video_writer is not None:
            if video_writer.write(canvas):
                frame_count += 1
        
        if save_images:
            if save_png:
                img_path = output_dir / f"frame_{frame_idx:05d}.png"
                cv2.imwrite(str(img_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                img_path = output_dir / f"frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(img_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        # Show live
        if show_live:
            cv2.imshow(f"Visualization - {sequence_name}", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Stopped by user")
                break
            elif key == ord(' '):  # Pause on space
                print("Paused - press any key to continue...")
                cv2.waitKey(0)
    
    # Cleanup
    if video_cap is not None:
        video_cap.release()
    
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved! ({frame_count}/{end_frame - start_frame} frames written)")
        video_created = True
    else:
        video_created = False
    
    if show_live:
        cv2.destroyAllWindows()
    
    # Print summary
    if save_images:
        print(f"\nImages saved to: {output_dir}/")
        
        # If video was requested but not created, try creating from images
        if video_requested and not video_created:
            video_path = output_dir / f"{sequence_name}_visualization.mp4"
            print("\nAttempting to create video from saved images...")
            if create_video_from_images(output_dir, video_path, fps=25):
                video_created = True
            else:
                print("\nVideo creation failed. You can create it manually with:")
                print(f"  ffmpeg -framerate 25 -i {output_dir}/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p {video_path}")
    
    print("\nDone!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize 2D projections (like main.py --visualize) without video"
    )
    parser.add_argument("--sequence", "-s", type=str, default="ARG_CRO_225412",
                       help="Sequence name")
    parser.add_argument("--frame", "-f", type=int, default=0,
                       help="Frame index (for single frame mode)")
    parser.add_argument("--start", type=int, default=0,
                       help="Starting frame (for video mode)")
    parser.add_argument("--num_frames", "-n", type=int, default=None,
                       help="Number of frames to process (default: all)")
    parser.add_argument("--mode", "-m", choices=['single', 'video'], default='single',
                       help="Visualization mode")
    parser.add_argument("--save_video", action="store_true",
                       help="Save as MP4 video")
    parser.add_argument("--save_images", action="store_true",
                       help="Save individual frame images")
    parser.add_argument("--no_live", action="store_true",
                       help="Don't show live window")
    parser.add_argument("--no_pitch", action="store_true",
                       help="Hide pitch point projections")
    parser.add_argument("--no_boxes", action="store_true",
                       help="Hide bounding boxes")
    parser.add_argument("--no_skeleton", action="store_true",
                       help="Hide 2D skeleton keypoints")
    parser.add_argument("--show_3d_projection", action="store_true",
                       help="Show 3D skeleton projected to 2D")
    parser.add_argument("--use_video", action="store_true",
                       help="Use actual video/images as background (from data/videos/ or data/images/)")
    parser.add_argument("--minimap", action="store_true",
                       help="Show top-down tactical minimap (picture-in-picture)")
    parser.add_argument("--results", "-r", type=str, default=None,
                       help="Path to results NPZ file (e.g. outputs/submission_val.npz)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare mode: ghost 2D data skeleton + projected results from --results file")
    parser.add_argument("--ghost_opacity", type=float, default=0.9,
                       help="Opacity of ghost 2D skeleton in compare mode (0.0-1.0, default: 0.4)")
    parser.add_argument("--keypoints15", action="store_true",
                       help="Show only the 15 submission keypoints (via OPENPOSE_TO_OURS) instead of all 25")
    parser.add_argument("--save_png", action="store_true",
                       help="Save images as lossless PNG instead of JPEG")
    args = parser.parse_args()

    # Load results when --results is provided (used for both --results alone and --compare)
    results_data = None
    if args.results is not None:
        results_npz = dict(np.load(args.results, allow_pickle=False))
        if args.sequence in results_npz:
            results_data = results_npz[args.sequence]  # (persons, frames, 15, 3)
            print(f"✓ Loaded results for '{args.sequence}': shape={results_data.shape}")
        else:
            print(f"⚠ Sequence '{args.sequence}' not found in {args.results}")
            print(f"  Available keys: {list(results_npz.keys())[:5]}")

    if args.mode == 'single':
        # Single frame visualization
        canvas = create_visualization(
            args.sequence, args.frame,
            show_pitch=not args.no_pitch,
            show_boxes=not args.no_boxes,
            show_2d_skeleton=not args.no_skeleton,
            show_3d_projection=args.show_3d_projection,
            use_video=args.use_video,
            show_minimap=args.minimap,
            results_data=results_data,
            show_results=args.results is not None,
            show_ghost=args.compare,
            ghost_opacity=args.ghost_opacity,
            kp15=args.keypoints15
        )

        # Save or show
        if args.save_images:
            output_dir = Path("outputs") / args.sequence
            output_dir.mkdir(parents=True, exist_ok=True)
            if args.save_png:
                img_path = output_dir / f"frame_{args.frame:05d}.png"
                cv2.imwrite(str(img_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                img_path = output_dir / f"frame_{args.frame:05d}.jpg"
                cv2.imwrite(str(img_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"Saved to {img_path}")

        if not args.no_live:
            cv2.imshow(f"Frame {args.frame} - {args.sequence}", canvas)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        # Video mode
        visualize_sequence(
            args.sequence,
            start_frame=args.start,
            num_frames=args.num_frames,
            save_video=args.save_video,
            save_images=args.save_images,
            show_live=not args.no_live,
            show_pitch=not args.no_pitch,
            show_boxes=not args.no_boxes,
            show_2d_skeleton=not args.no_skeleton,
            show_3d_projection=args.show_3d_projection,
            use_video=args.use_video,
            show_minimap=args.minimap,
            results_file=args.results,
            show_results=args.results is not None,
            show_ghost=args.compare,
            ghost_opacity=args.ghost_opacity,
            kp15=args.keypoints15,
            save_png=args.save_png
        )


if __name__ == "__main__":
    main()
