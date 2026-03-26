"""
Interactive 3D visualization of skeleton tracking using VTK.

Closely mirrors the look of visualize_3d_interactive.py (matplotlib) but uses
VTK for hardware-accelerated rendering and smooth playback.

Features:
- Hardware-accelerated OpenGL rendering via VTK
- Tube bones + sphere joints (matching the matplotlib stick-figure look)
- Same default view angle as the matplotlib version (derived from real camera pos)
- Soccer field with pitch-point markers
- Camera position indicator (orange cone)
- On-screen HUD: frame counter, player count, play status
- Frame scrubber slider
- Smooth ~30 FPS playback timer

Controls:
- Mouse left drag          : Rotate
- Mouse middle / Shift+LMB : Pan
- Mouse right / Scroll     : Zoom
- Space                    : Play / Pause
- Left / Right             : ±1 frame
- Up / Down                : ±10 frames
- r                        : Reset camera to default view
- q / Escape               : Quit

Usage:
    python visualize_3d_vtk.py --sequence ARG_CRO_225412
"""

import numpy as np
import vtk
from pathlib import Path
import argparse
import subprocess


def _primary_screen_size():
    """Return (width, height) of the primary monitor only.
    Uses xrandr on Linux/X11; falls back to 1920×1080."""
    import re
    try:
        out = subprocess.check_output(
            ['xrandr', '--current'], text=True, stderr=subprocess.DEVNULL)
        # Prefer the line flagged 'primary'
        for line in out.splitlines():
            if ' connected primary' in line:
                m = re.search(r'(\d+)x(\d+)\+', line)
                if m:
                    return int(m.group(1)), int(m.group(2))
        # Fall back to first connected output
        for line in out.splitlines():
            if ' connected' in line:
                m = re.search(r'(\d+)x(\d+)\+', line)
                if m:
                    return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return 1920, 1080  # safe default — no VTK context needed


# ---------------------------------------------------------------------------
# Skeleton topology  (identical to visualize_3d_interactive.py)
# ---------------------------------------------------------------------------

SKELETON_CONNECTIONS_25 = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (0, 16), (15, 17), (16, 18),
    (14, 19), (19, 20), (14, 21),
    (11, 22), (22, 23), (11, 24),
]

# 15-joint format
# Joints: [Nose, RShoulder, LShoulder, RElbow, LElbow, RWrist, LWrist,
#          RHip, LHip, RKnee, LKnee, RAnkle, LAnkle, RBigToe, LBigToe]
SKELETON_CONNECTIONS_15 = [
    (0, 1),   # Nose → RShoulder
    (0, 2),   # Nose → LShoulder
    (1, 3),   # RShoulder → RElbow
    (3, 5),   # RElbow → RWrist
    (2, 4),   # LShoulder → LElbow
    (4, 6),   # LElbow → LWrist
    (1, 7),   # RShoulder → RHip
    (2, 8),   # LShoulder → LHip
    (7, 8),   # RHip → LHip
    (7, 9),   # RHip → RKnee
    (9, 11),  # RKnee → RAnkle
    (8, 10),  # LHip → LKnee
    (10, 12), # LKnee → LAnkle
    (11, 13), # RAnkle → RBigToe
    (12, 14), # LAnkle → LBigToe
]

# Player colour palette – matches the matplotlib version's colour order
PLAYER_COLORS = [
    (1.00, 0.10, 0.10),  # red
    (0.10, 0.30, 1.00),  # blue
    (0.10, 0.75, 0.10),  # green
    (1.00, 0.55, 0.00),  # orange
    (0.60, 0.10, 0.80),  # purple
    (0.00, 0.85, 0.85),  # cyan
    (1.00, 0.10, 0.75),  # magenta
    (0.95, 0.85, 0.00),  # yellow
]

MAX_PLAYERS = 110  # YoloX tracks up to ~104 people per sequence


# ---------------------------------------------------------------------------
# Data loading  (identical logic to visualize_3d_interactive.py)
# ---------------------------------------------------------------------------

def transform_camera_to_world(skel_3d_camera, R, t, skel_2d, K, dist_coeffs):
    """Transform skeletons from camera coordinates to world coordinates."""

    def ray_from_xy(xy, K, R, t, k1, k2):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_norm = (xy[0] - cx) / fx
        y_norm = (xy[1] - cy) / fy
        r2 = x_norm**2 + y_norm**2
        distortion = 1 + k1 * r2 + k2 * r2**2
        x_undist = x_norm * distortion
        y_undist = y_norm * distortion
        ray_cam = np.array([x_undist, y_undist, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)
        ray_world = R.T @ ray_cam
        origin_world = -R.T @ t
        return origin_world, ray_world

    def intersection_over_plane(origin, direction, plane_z=0.0):
        if abs(direction[2]) < 1e-6:
            return None
        t_int = (plane_z - origin[2]) / direction[2]
        if t_int < 0:
            return None
        return origin + t_int * direction

    skel_3d_world = []
    for skeleton_cam, skeleton_2d in zip(skel_3d_camera, skel_2d):
        if skeleton_cam.shape[0] == 0 or skeleton_cam.shape[1] != 3:
            skel_3d_world.append(skeleton_cam)
            continue
        valid_2d = ~np.isnan(skeleton_2d).any(axis=1)
        if not valid_2d.any():
            skel_3d_world.append(skeleton_cam)
            continue
        lowest_idx = np.where(valid_2d)[0][np.argmax(skeleton_2d[valid_2d, 1])]
        foot_2d = skeleton_2d[lowest_idx, :2]
        k1 = dist_coeffs[0]
        k2 = dist_coeffs[1] if len(dist_coeffs) > 1 else 0.0
        origin, direction = ray_from_xy(foot_2d, K, R, t, k1, k2)
        foot_3d = intersection_over_plane(origin, direction, plane_z=0.0)
        if foot_3d is None:
            skel_3d_world.append(skeleton_cam)
            continue
        skeleton_world = skeleton_cam @ R
        skeleton_world -= skeleton_world[lowest_idx]
        skeleton_world += foot_3d
        skel_3d_world.append(skeleton_world)
    return skel_3d_world


def _parse_submission_for_sequence(submission_file, sequence_name):
    """Parse world-coord skeletons from a submission NPZ.
    Returns (skel_3d_world_frames, True) on success, (None, False) on miss."""
    p = Path(submission_file)
    if not p.exists():
        return None, False
    submission = np.load(p)
    if sequence_name not in submission:
        return None, False
    raw = submission[sequence_name]   # (P, F, 15, 3)
    num_people, num_frames = raw.shape[:2]
    skel_3d_world_frames = []
    for fi in range(num_frames):
        frame_skels = []
        for pi in range(num_people):
            skel = raw[pi, fi]
            if not np.all(skel == 0) and not np.all(np.isnan(skel)):
                frame_skels.append((pi, skel))
        skel_3d_world_frames.append(frame_skels)
    print(f"Loaded {num_frames} frames ({num_people}-person format, 15 joints) "
          f"from {p.name}")
    return skel_3d_world_frames, True


def load_skels_and_boxes(sequence_name, mode, data_dir,
                         submission_file, yolox_dir):
    """Load skeleton + bounding-box data for *mode* ('baseline' or 'yolox').
    Camera calibration and pitch points are NOT loaded here — they are
    unchanged across mode switches.

    Fallback chain
    --------------
    Skeletons : YoloX submission → baseline submission → data/skel_3d/
    Boxes     : YoloX boxes_tracked/ → data/boxes/
    """
    data_path  = Path(data_dir)
    yolox_path = Path(yolox_dir)

    # --- Skeletons ---
    skel_3d_world_frames = None
    use_world_coords = False
    skel_2d = None

    if mode == 'yolox':
        skel_3d_world_frames, use_world_coords = _parse_submission_for_sequence(
            yolox_path / 'submission_full.npz', sequence_name)
        if not use_world_coords:
            print(f"  YoloX submission missing '{sequence_name}' – falling back to baseline")

    if not use_world_coords:
        skel_3d_world_frames, use_world_coords = _parse_submission_for_sequence(
            submission_file, sequence_name)

    if not use_world_coords:
        skel_3d_path = data_path / 'skel_3d' / f'{sequence_name}.npy'
        skel_3d_world_frames = np.load(skel_3d_path, allow_pickle=True)
        skel_2d_path = data_path / 'skel_2d' / f'{sequence_name}.npy'
        skel_2d = np.load(skel_2d_path, allow_pickle=True)

    # --- Bounding boxes ---
    if mode == 'yolox':
        boxes_path = yolox_path / 'boxes_tracked' / f'{sequence_name}.npy'
        if not boxes_path.exists():
            print(f"  YoloX boxes missing for '{sequence_name}' – falling back to baseline")
            boxes_path = data_path / 'boxes' / f'{sequence_name}.npy'
    else:
        boxes_path = data_path / 'boxes' / f'{sequence_name}.npy'

    boxes = np.load(boxes_path) if boxes_path.exists() else None

    return skel_3d_world_frames, skel_2d, use_world_coords, boxes


def load_data(sequence_name, data_dir='data',
              submission_file='outputs/submission_full.npz',
              yolox_dir='YoloX+DeepEIOU_Data',
              mode='baseline'):
    """Load skeleton, camera, and pitch-point data for a sequence."""
    data_path = Path(data_dir)

    skel_3d_world_frames, skel_2d, use_world_coords, boxes = load_skels_and_boxes(
        sequence_name, mode, data_dir, submission_file, yolox_dir)

    # Prefer per-frame estimated calibration from the camera tracker output
    cal_path = Path('outputs') / 'calibration' / f'{sequence_name}.npz'
    if cal_path.exists():
        cam = np.load(cal_path)
        print(f"Loaded per-frame calibration from {cal_path}  "
              f"(R: {cam['R'].shape}, t: {cam['t'].shape})")
    else:
        cam = np.load(data_path / 'cameras' / f'{sequence_name}.npz')
        print(f"Using static calibration from data/cameras/  "
              f"(R: {cam['R'].shape}, t: {cam['t'].shape})")
    K, dist_coeffs, R, t = cam['K'], cam['k'], cam['R'], cam['t']

    pitch_points = []
    with open(data_path / 'pitch_points.txt') as f:
        for line in f:
            x, y, z = line.strip().split()
            pitch_points.append([float(x), float(y), float(z)])
    pitch_points = np.array(pitch_points)

    return skel_3d_world_frames, skel_2d, K, dist_coeffs, R, t, pitch_points, use_world_coords, boxes



# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

class VTK3DVisualizer:
    """VTK-based interactive 3D skeleton viewer.

    Matches the look and feel of the matplotlib visualize_3d_interactive.py.
    """

    PLAY_INTERVAL_MS = 33   # ~30 FPS

    def _init_skeleton_topology(self):
        if self.skel_3d_world[0]:
            first = self.skel_3d_world[0][0]
            skel = first[1] if isinstance(first, tuple) else first
            self.num_joints = skel.shape[0]
        else:
            self.num_joints = 15
        self.connections = (SKELETON_CONNECTIONS_15
                            if self.num_joints == 15
                            else SKELETON_CONNECTIONS_25)

    def _update_camera_geometry(self):
        """Recompute cam_world_pos, field_bounds, field_center, elev/azim
        from the currently loaded R/t/pitch_points."""
        R0 = self.R[0] if self.R.ndim > 2 else self.R
        t0 = self.t[0] if self.t.ndim > 1 else self.t
        self.cam_world_pos = -R0.T @ t0

        x_min = self.pitch_points[:, 0].min()
        x_max = self.pitch_points[:, 0].max()
        y_min = self.pitch_points[:, 1].min()
        y_max = self.pitch_points[:, 1].max()
        self.field_bounds = (x_min, x_max, y_min, y_max)
        self.field_center = np.array([(x_min + x_max) / 2,
                                      (y_min + y_max) / 2, 0.0])
        d = self.cam_world_pos - self.field_center
        horiz = np.sqrt(d[0]**2 + d[1]**2)
        self._default_elev_rad = np.arctan2(d[2], horiz)
        self._default_azim_rad = np.arctan2(d[0], -d[1])

    def __init__(self, sequence_name, data_dir='data',
                 submission_file='outputs/submission_full.npz',
                 yolox_dir='YoloX+DeepEIOU_Data'):
        # Store paths so _switch_sequence can reuse them
        self._data_dir = data_dir
        self._submission_file = submission_file
        self._yolox_dir = yolox_dir
        self._data_mode = 'baseline'   # 'baseline' or 'yolox'; toggle with Y

        # Discover all sequences (union of data/cameras + outputs/calibration)
        self.sequences = sorted(
            p.stem for p in Path(data_dir, 'cameras').glob('*.npz'))
        if sequence_name not in self.sequences:
            self.sequences.insert(0, sequence_name)
        self.sequence_idx = self.sequences.index(sequence_name)
        self.sequence_name = sequence_name

        print(f"Loading data for sequence: {sequence_name}")
        (self.skel_3d_world, self.skel_2d, self.K, self.dist_coeffs,
         self.R, self.t, self.pitch_points,
         self.use_world_coords, self.boxes) = load_data(
             sequence_name, data_dir, submission_file,
             yolox_dir, self._data_mode)

        self.num_frames = len(self.skel_3d_world)
        self.current_frame = 0
        self.playing = False
        self.timer_id = None
        self._camera_follow = False   # start in free overview; 'o' → broadcast cam follow

        # Video overlay (broadcast mode only, toggle with 'v')
        self._video_on   = False
        # Bounding-box overlay (broadcast mode only, toggle with 'b')
        self._bbox_on    = False
        self._update_images_dir(sequence_name, data_dir)

        # Skeleton topology
        self._init_skeleton_topology()

        # Real broadcast-camera world position + field geometry
        self._update_camera_geometry()

        # Build scene
        self._setup_vtk()
        self._build_lighting()
        self._build_field()
        self._build_camera_indicator()
        self._build_player_actors()
        self._build_bbox_actors()
        self._build_hud()
        self._build_slider()
        self._setup_camera()

        # First frame
        self._update_frame()

        # Recompute clipping range now that skeletons are visible
        self.renderer.ResetCameraClippingRange()

        print("\nControls:")
        print("  Mouse left drag          : Rotate")
        print("  Mouse middle / Shift+LMB : Pan")
        print("  Mouse right / Scroll     : Zoom")
        print("  Space                    : Play / Pause")
        print("  ← / →                   : ±1 frame")
        print("  ↑ / ↓                   : ±10 frames")
        print("  [ / ]                   : Previous / next sequence")
        print("  o                        : Toggle broadcast-camera follow")
        print("  r                        : Reset to overview camera")
        print("  q / Escape               : Quit")
        print(f"\nTotal frames: {self.num_frames}")

        R = self.R[0] if self.R.ndim > 2 else self.R
        t_ = self.t[0] if self.t.ndim > 1 else self.t
        cc = -R.T @ t_
        print(f"\nCamera world pos : X={cc[0]:.2f}  Y={cc[1]:.2f}  Z={cc[2]:.2f}")
        x_min, x_max, y_min, y_max = self.field_bounds
        print(f"Field X range    : {x_min:.1f} → {x_max:.1f}")
        print(f"Field Y range    : {y_min:.1f} → {y_max:.1f}")

    # ------------------------------------------------------------------
    # VTK window / interactor
    # ------------------------------------------------------------------

    def _setup_vtk(self):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.53, 0.81, 0.98)  # light-blue sky

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)

        # Windowed fullscreen: fill primary screen, keep OS decorations
        # so the window manager grants keyboard focus correctly
        sw, sh = _primary_screen_size()
        self.render_window.SetSize(sw, sh)
        self.render_window.SetPosition(0, 0)
        self.render_window.SetWindowName(
            f"FIFA 3D Skeleton Viewer – {self.sequence_name}")

        self._screen_w   = sw
        self._screen_h   = sh
        self._font_scale = sh / 1080.0    # >1 on 4K, <1 on small displays

        # Background texture pipeline (video frames shown in broadcast mode)
        self._bg_reader  = vtk.vtkJPEGReader()
        self._bg_texture = vtk.vtkTexture()
        self._bg_texture.SetInputConnection(self._bg_reader.GetOutputPort())
        self._bg_texture.InterpolateOn()
        self.renderer.SetBackgroundTexture(self._bg_texture)
        # TexturedBackground starts OFF; toggled per-frame by _update_video_bg

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        self.interactor.AddObserver('KeyPressEvent', self._on_key_press)
        self.interactor.AddObserver('TimerEvent', self._on_timer)

    # ------------------------------------------------------------------
    # Lighting  (flat ambient-dominant, similar to matplotlib's look)
    # ------------------------------------------------------------------

    def _build_lighting(self):
        self.renderer.RemoveAllLights()

        # Key light from the camera direction
        key = vtk.vtkLight()
        key.SetLightTypeToHeadlight()
        key.SetIntensity(0.7)
        key.SetDiffuseColor(1, 1, 1)
        key.SetSpecularColor(0.5, 0.5, 0.5)
        self.renderer.AddLight(key)

        # Fill light from below to reduce harsh shadows
        fill = vtk.vtkLight()
        fill.SetLightTypeToSceneLight()
        fill.SetPosition(0, 0, -50)
        fill.SetFocalPoint(0, 0, 0)
        fill.SetIntensity(0.4)
        fill.SetDiffuseColor(1, 1, 1)
        self.renderer.AddLight(fill)

    # ------------------------------------------------------------------
    # Soccer field
    # ------------------------------------------------------------------

    def _build_field(self):
        x_min, x_max, y_min, y_max = self.field_bounds
        margin = 15.0

        def flat_plane(ox, oy, oz, px, py, pz, qx, qy, qz,
                       color, opacity):
            src = vtk.vtkPlaneSource()
            src.SetOrigin(ox, oy, oz)
            src.SetPoint1(px, py, pz)
            src.SetPoint2(qx, qy, qz)
            src.Update()
            m = vtk.vtkPolyDataMapper()
            m.SetInputConnection(src.GetOutputPort())
            a = vtk.vtkActor()
            a.SetMapper(m)
            a.GetProperty().SetColor(*color)
            a.GetProperty().SetOpacity(opacity)
            a.GetProperty().SetAmbient(1.0)   # flat – no lighting
            a.GetProperty().SetDiffuse(0.0)
            a.GetProperty().BackfaceCullingOff()
            return a

        # Outer ground with margin (dark green)
        outer = flat_plane(
            x_min - margin, y_min - margin, -0.02,
            x_max + margin, y_min - margin, -0.02,
            x_min - margin, y_max + margin, -0.02,
            (0.13, 0.55, 0.13), 0.9)
        self.renderer.AddActor(outer)

        # Bright playing surface (lime green)
        inner = flat_plane(
            x_min, y_min, 0.00,
            x_max, y_min, 0.00,
            x_min, y_max, 0.00,
            (0.20, 0.80, 0.20), 0.85)
        self.renderer.AddActor(inner)

        # Pitch-point markers (white dots)
        pts_vtk = vtk.vtkPoints()
        for p in self.pitch_points:
            pts_vtk.InsertNextPoint(p[0], p[1], 0.02)

        poly = vtk.vtkPolyData()
        poly.SetPoints(pts_vtk)

        verts = vtk.vtkCellArray()
        for i in range(len(self.pitch_points)):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        poly.SetVerts(verts)

        m = vtk.vtkPolyDataMapper()
        m.SetInputData(poly)
        pitch_a = vtk.vtkActor()
        pitch_a.SetMapper(m)
        pitch_a.GetProperty().SetColor(1, 1, 1)
        pitch_a.GetProperty().SetPointSize(6)
        pitch_a.GetProperty().SetAmbient(1.0)
        pitch_a.GetProperty().SetDiffuse(0.0)
        self.renderer.AddActor(pitch_a)

        # Store refs for opacity control (video mode)
        self._field_actors = [outer, inner, pitch_a]
        self._field_base_opacities = [0.9, 0.85, 1.0]

    def _set_field_opacity(self, scale: float):
        """Scale field actor opacities by *scale* (1.0 = normal, 0.15 = semi-transparent)."""
        for actor, base in zip(self._field_actors, self._field_base_opacities):
            actor.GetProperty().SetOpacity(base * scale)

    # ------------------------------------------------------------------
    # Per-frame real-camera helpers
    # ------------------------------------------------------------------

    def _get_frame_camera(self, frame_idx: int):
        """Return (world_pos, look_dir, up_dir) for the real broadcast camera
        at *frame_idx*.  Handles both static (R.ndim==2) and per-frame R/t."""
        fi = min(frame_idx, len(self.R) - 1) if self.R.ndim > 2 else 0
        R = self.R[fi] if self.R.ndim > 2 else self.R
        fi_t = min(frame_idx, len(self.t) - 1) if self.t.ndim > 1 else 0
        t = self.t[fi_t].ravel() if self.t.ndim > 1 else self.t.ravel()

        cam_pos  = -R.T @ t
        look_dir =  R.T @ np.array([0.0, 0.0,  1.0])  # camera +Z in world
        up_dir   =  R.T @ np.array([0.0, -1.0, 0.0])  # camera -Y → world up
        return cam_pos, look_dir, up_dir

    def _get_frame_K(self, frame_idx: int) -> np.ndarray:
        """Return the intrinsic matrix K for *frame_idx* (per-frame zoom/pan)."""
        fi = min(frame_idx, len(self.K) - 1) if self.K.ndim > 2 else 0
        return self.K[fi] if self.K.ndim > 2 else self.K

    def _frustum_corners(self, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                         depth: float = 55.0):
        """Compute camera world pos + 4 far frustum corners from K, R, t."""
        cam_pos = -R.T @ t.ravel()
        cx, cy = K[0, 2], K[1, 2]
        # approximate image extent from principal point
        W, H = 2.0 * cx, 2.0 * cy
        img_corners = [(0.0, 0.0), (W, 0.0), (W, H), (0.0, H)]
        far = []
        for u, v in img_corners:
            ray_cam = np.array([(u - K[0, 2]) / K[0, 0],
                                (v - K[1, 2]) / K[1, 1], 1.0])
            ray_cam /= np.linalg.norm(ray_cam)
            ray_world = R.T @ ray_cam
            far.append(cam_pos + ray_world * depth)
        return cam_pos, far

    # ------------------------------------------------------------------
    # Camera indicator: orange dot at camera + animated FOV frustum
    # The frustum 4 corner-rays are driven by K[frame] so they visibly
    # "zoom in / pan" even though the camera world position is static.
    # ------------------------------------------------------------------

    def _build_camera_indicator(self):
        cx, cy, cz = self.cam_world_pos

        # Pre-allocate frustum polydata: 5 points = cam + 4 far corners
        # 8 lines: 4 apex→corner + 4 corner→corner (perimeter)
        self._frustum_pts = vtk.vtkPoints()
        self._frustum_pts.SetNumberOfPoints(5)
        for i in range(5):
            self._frustum_pts.SetPoint(i, cx, cy, cz)

        cells = vtk.vtkCellArray()
        # apex → each corner
        for c in range(4):
            ln = vtk.vtkLine()
            ln.GetPointIds().SetId(0, 0)        # apex = pt 0
            ln.GetPointIds().SetId(1, c + 1)    # corner = pt 1..4
            cells.InsertNextCell(ln)
        # perimeter of the far rectangle
        for c in range(4):
            ln = vtk.vtkLine()
            ln.GetPointIds().SetId(0, c + 1)
            ln.GetPointIds().SetId(1, (c + 1) % 4 + 1)
            cells.InsertNextCell(ln)

        frustum_poly = vtk.vtkPolyData()
        frustum_poly.SetPoints(self._frustum_pts)
        frustum_poly.SetLines(cells)
        self._frustum_poly = frustum_poly

        fm = vtk.vtkPolyDataMapper()
        fm.SetInputData(frustum_poly)
        fa = vtk.vtkActor()
        fa.SetMapper(fm)
        fa.GetProperty().SetColor(1.0, 0.5, 0.0)
        fa.GetProperty().SetOpacity(0.55)
        fa.GetProperty().SetLineWidth(2.0)
        fa.GetProperty().SetAmbient(1.0)
        fa.GetProperty().SetDiffuse(0.0)
        self.renderer.AddActor(fa)
        self._frustum_actor = fa

    # ------------------------------------------------------------------
    # Player skeleton actors (pre-allocated for all 21 slots)
    # ------------------------------------------------------------------

    def _build_player_actors(self):
        """Allocate one set of VTK actors per player slot (reused every frame)."""
        self.players = []
        n = self.num_joints
        fc = self.field_center  # fallback position for invalid joints

        for pid in range(MAX_PLAYERS):
            color = PLAYER_COLORS[pid % len(PLAYER_COLORS)]

            # Shared point storage for both bones and joints
            pts = vtk.vtkPoints()
            pts.SetNumberOfPoints(n)
            for i in range(n):
                pts.SetPoint(i, fc[0], fc[1], fc[2])

            # ---- Bone tubes ----
            bone_poly = vtk.vtkPolyData()
            bone_poly.SetPoints(pts)
            bone_cells = vtk.vtkCellArray()
            for j1, j2 in self.connections:
                cell = vtk.vtkLine()
                cell.GetPointIds().SetId(0, j1)
                cell.GetPointIds().SetId(1, j2)
                bone_cells.InsertNextCell(cell)
            bone_poly.SetLines(bone_cells)

            tube = vtk.vtkTubeFilter()
            tube.SetInputData(bone_poly)
            tube.SetRadius(0.04)
            tube.SetNumberOfSides(10)
            tube.CappingOn()

            bm = vtk.vtkPolyDataMapper()
            bm.SetInputConnection(tube.GetOutputPort())
            ba = vtk.vtkActor()
            ba.SetMapper(bm)
            ba.GetProperty().SetColor(*color)
            ba.GetProperty().SetAmbient(0.5)
            ba.GetProperty().SetDiffuse(0.7)
            ba.GetProperty().SetSpecular(0.2)
            ba.GetProperty().SetSpecularPower(20)
            ba.VisibilityOff()
            self.renderer.AddActor(ba)

            # ---- Joint spheres (rendered-as-spheres point cloud) ----
            joint_poly = vtk.vtkPolyData()
            joint_poly.SetPoints(pts)   # same vtkPoints – shared
            vc = vtk.vtkCellArray()
            for i in range(n):
                vc.InsertNextCell(1)
                vc.InsertCellPoint(i)
            joint_poly.SetVerts(vc)

            jm = vtk.vtkPolyDataMapper()
            jm.SetInputData(joint_poly)
            ja = vtk.vtkActor()
            ja.SetMapper(jm)
            ja.GetProperty().SetColor(*color)
            ja.GetProperty().SetPointSize(7)
            ja.GetProperty().RenderPointsAsSpheresOn()
            ja.GetProperty().SetAmbient(0.5)
            ja.GetProperty().SetDiffuse(0.7)
            ja.VisibilityOff()
            self.renderer.AddActor(ja)

            # ---- Billboard text label ----
            try:
                label = vtk.vtkBillboardTextActor3D()
                label.SetInput(f'P{pid}')
                label.GetTextProperty().SetFontSize(max(10, int(16 * self._font_scale)))
                label.GetTextProperty().SetColor(*color)
                label.GetTextProperty().SetBold(True)
                label.GetTextProperty().ShadowOn()
                label.SetPosition(fc[0], fc[1], fc[2])
                label.VisibilityOff()
                self.renderer.AddActor(label)
            except AttributeError:
                label = None

            self.players.append({
                'pts': pts,
                'bone_poly': bone_poly,
                'joint_poly': joint_poly,
                'bone_actor': ba,
                'joint_actor': ja,
                'label': label,
            })

    # ------------------------------------------------------------------
    # HUD (2-D screen-space text overlay)
    # ------------------------------------------------------------------
    # Bounding-box 2-D overlay  (broadcast mode only)
    # ------------------------------------------------------------------
    # Box data is (F, 21, 4) in image pixels [x1,y1,x2,y2], NaN = absent.
    # We map pixels → normalised-viewport coords so they sit on the video bg.
    #   norm_x = px / 1920,   norm_y = 1 - py / 1080  (VTK Y is up)
    # ------------------------------------------------------------------

    IMG_W_F = 1920.0
    IMG_H_F = 1080.0

    def _build_bbox_actors(self):
        """Pre-allocate one 2-D rectangle actor per player slot."""
        self._bbox_actors = []
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToNormalizedViewport()

        for pid in range(MAX_PLAYERS):
            color = PLAYER_COLORS[pid % len(PLAYER_COLORS)]

            # 4 corner points + close the rectangle (5 points, 4 line segments)
            pts = vtk.vtkPoints()
            pts.SetNumberOfPoints(4)
            for i in range(4):
                pts.SetPoint(i, 0.0, 0.0, 0.0)

            cells = vtk.vtkCellArray()
            for i in range(4):
                cells.InsertNextCell(2)
                cells.InsertCellPoint(i)
                cells.InsertCellPoint((i + 1) % 4)

            poly = vtk.vtkPolyData()
            poly.SetPoints(pts)
            poly.SetLines(cells)

            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputData(poly)
            mapper.SetTransformCoordinate(coord)

            actor = vtk.vtkActor2D()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(2.5)
            actor.VisibilityOff()
            self.renderer.AddViewProp(actor)

            self._bbox_actors.append({'actor': actor, 'pts': pts, 'poly': poly})

    def _update_bbox_overlay(self, frame_idx: int):
        """Show / hide bounding-box rectangles for this frame."""
        show = self._camera_follow and self._bbox_on and self.boxes is not None

        if not show:
            for d in self._bbox_actors:
                d['actor'].VisibilityOff()
            return

        fi = min(frame_idx, len(self.boxes) - 1)
        frame_boxes = self.boxes[fi]   # (21, 4)

        for pid, d in enumerate(self._bbox_actors):
            box = frame_boxes[pid] if pid < len(frame_boxes) else None
            if box is None or np.any(np.isnan(box)):
                d['actor'].VisibilityOff()
                continue

            x1, y1, x2, y2 = box
            # Map image pixels → normalised viewport (VTK Y up)
            nx1 = x1 / self.IMG_W_F
            nx2 = x2 / self.IMG_W_F
            ny1 = 1.0 - y2 / self.IMG_H_F   # bottom of rect in VTK coords
            ny2 = 1.0 - y1 / self.IMG_H_F   # top of rect

            pts = d['pts']
            pts.SetPoint(0, nx1, ny1, 0)
            pts.SetPoint(1, nx2, ny1, 0)
            pts.SetPoint(2, nx2, ny2, 0)
            pts.SetPoint(3, nx1, ny2, 0)
            pts.Modified()
            d['poly'].Modified()
            d['actor'].VisibilityOn()

    # ------------------------------------------------------------------

    def _build_hud(self):
        fs = self._font_scale
        sz  = lambda base: max(10, int(base * fs))  # scaled font size

        def norm_actor(base_size, bold=False, color=(0, 0, 0),
                       mono=False, nx=0.01, ny=0.96):
            a = vtk.vtkTextActor()
            tp = a.GetTextProperty()
            tp.SetFontSize(sz(base_size))
            tp.SetColor(*color)
            if bold:
                tp.SetBold(True)
            if mono:
                tp.SetFontFamilyToCourier()
            a.GetPositionCoordinate() \
             .SetCoordinateSystemToNormalizedDisplay()
            a.GetPositionCoordinate().SetValue(nx, ny)
            return a

        # Title — top-left
        self.title_actor = norm_actor(18, bold=True, ny=0.965)
        self.renderer.AddViewProp(self.title_actor)

        # Controls hint — just above the slider (slider sits at ~0.03)
        hint = norm_actor(12, color=(0.15, 0.15, 0.15), ny=0.065)
        hint.SetInput(
            "Space: Play/Pause  |  ←/→: frame  |  ↑/↓: ±10  |  [/]: sequence  "
            "|  Y: Baseline/YoloX  |  O: broadcast cam  |  V: video  |  B: bbox  "
            "|  R: reset  |  Q: quit")
        self.renderer.AddViewProp(hint)

        # Camera parameter panel — top-right
        self.cam_info_actor = norm_actor(
            13, mono=True, color=(0.05, 0.05, 0.35), nx=0.72, ny=0.75)
        self.renderer.AddViewProp(self.cam_info_actor)

        # Sequence list panel — left side
        self.seq_panel_actor = norm_actor(
            13, mono=True, color=(0.05, 0.25, 0.05), nx=0.01, ny=0.35)
        self._update_sequence_panel()
        self.renderer.AddViewProp(self.seq_panel_actor)

    def _update_sequence_panel(self):
        lines = ["── SEQUENCES ──  [ ] to switch"]
        for i, seq in enumerate(self.sequences):
            marker = "►" if i == self.sequence_idx else " "
            lines.append(f"{marker} {seq}")
        self.seq_panel_actor.SetInput("\n".join(lines))

    # ------------------------------------------------------------------
    # Frame scrubber slider
    # ------------------------------------------------------------------

    def _build_slider(self):
        rep = vtk.vtkSliderRepresentation2D()
        rep.SetMinimumValue(0)
        rep.SetMaximumValue(max(1, self.num_frames - 1))
        rep.SetValue(0)
        rep.SetTitleText("Frame")
        rep.GetSliderProperty().SetColor(1.0, 0.5, 0.0)
        rep.GetTitleProperty().SetColor(0, 0, 0)
        rep.GetLabelProperty().SetColor(0, 0, 0)
        rep.GetSelectedProperty().SetColor(1, 0.8, 0)
        rep.GetTubeProperty().SetColor(0.4, 0.4, 0.4)
        rep.GetCapProperty().SetColor(0.4, 0.4, 0.4)
        rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint1Coordinate().SetValue(0.10, 0.03)
        rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint2Coordinate().SetValue(0.90, 0.03)
        rep.SetSliderLength(0.012)
        rep.SetSliderWidth(0.012)

        self.slider_widget = vtk.vtkSliderWidget()
        self.slider_widget.SetInteractor(self.interactor)
        self.slider_widget.SetRepresentation(rep)
        self.slider_widget.SetAnimationModeToAnimate()
        self.slider_widget.EnabledOn()
        self.slider_widget.AddObserver('InteractionEvent', self._on_slider)
        self._slider_rep = rep

    # ------------------------------------------------------------------
    # Camera setup
    # ------------------------------------------------------------------

    def _setup_camera(self):
        """
        Default view = real broadcast camera position/orientation from R and t.
        Also pre-computes an overview position (accessible via 'o' key).
        """
        # ---- Overview position (stored for 'o' key) ----
        elev = self._default_elev_rad
        azim = self._default_azim_rad
        x_min, x_max, y_min, y_max = self.field_bounds
        field_diag = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        dist = field_diag * 0.85
        ov_x = self.field_center[0] + dist * np.cos(elev) * np.sin(azim)
        ov_y = self.field_center[1] - dist * np.cos(elev) * np.cos(azim)
        ov_z = self.field_center[2] + dist * np.sin(elev)
        self._overview_pos   = (ov_x, ov_y, ov_z)
        self._overview_focal = tuple(self.field_center)
        self._overview_up    = (0.0, 0.0, 1.0)

        # ---- Default: overview (free navigation) ----
        self._reset_to_overview()

    # ------------------------------------------------------------------
    # Per-frame real-camera indicator + optional camera follow
    # ------------------------------------------------------------------

    def _update_camera_indicator(self, frame_idx: int):
        """Recompute the 4 frustum corner rays using K[frame] (zoom/pan changes)."""
        R_fi = min(frame_idx, len(self.R) - 1) if self.R.ndim > 2 else 0
        R = self.R[R_fi] if self.R.ndim > 2 else self.R
        fi_t = min(frame_idx, len(self.t) - 1) if self.t.ndim > 1 else 0
        t = self.t[fi_t].ravel() if self.t.ndim > 1 else self.t.ravel()
        K = self._get_frame_K(frame_idx)

        cam_pos, far_corners = self._frustum_corners(K, R, t)
        self._frustum_pts.SetPoint(0, *cam_pos)
        for i, fc in enumerate(far_corners):
            self._frustum_pts.SetPoint(i + 1, *fc)
        self._frustum_pts.Modified()
        self._frustum_poly.Modified()

    def _update_cam_info_hud(self, frame_idx: int):
        """Refresh the top-right camera parameter panel for *frame_idx*."""
        K  = self._get_frame_K(frame_idx)
        fi_k = min(frame_idx, len(self.dist_coeffs) - 1) \
               if self.dist_coeffs.ndim > 1 else 0
        dist = self.dist_coeffs[fi_k] if self.dist_coeffs.ndim > 1 \
               else self.dist_coeffs

        fx, fy  = K[0, 0], K[1, 1]
        cx, cy  = K[0, 2], K[1, 2]
        img_w   = 2.0 * cx          # approx from principal point
        img_h   = 2.0 * cy
        fov_h   = np.degrees(2.0 * np.arctan(img_w / (2.0 * fx)))
        fov_v   = np.degrees(2.0 * np.arctan(img_h / (2.0 * fy)))

        # Extrinsics (position static, orientation per-frame)
        cam_pos, look_dir, _ = self._get_frame_camera(frame_idx)
        horiz   = np.sqrt(cam_pos[0]**2 + cam_pos[1]**2)
        elev    = np.degrees(np.arctan2(cam_pos[2], horiz))
        dist_fc = np.linalg.norm(cam_pos - self.field_center)

        # Ground intersection: where the optical axis hits Z = 0
        if abs(look_dir[2]) > 1e-6:
            t_gnd  = -cam_pos[2] / look_dir[2]
            gnd    = cam_pos + look_dir * t_gnd
            gnd_str = f"({gnd[0]:+.1f}, {gnd[1]:+.1f})"
        else:
            gnd_str = "parallel (no hit)"

        lines = [
            "─── CAMERA PARAMETERS ───",
            "",
            "INTRINSICS  (per-frame)",
            f"  fx        {fx:>10.1f} px",
            f"  fy        {fy:>10.1f} px",
            f"  cx        {cx:>10.1f} px",
            f"  cy        {cy:>10.1f} px",
            f"  img size  {img_w:.0f} x {img_h:.0f} px",
            f"  FOV horiz {fov_h:>8.2f} deg",
            f"  FOV vert  {fov_v:>8.2f} deg",
            f"  k1        {dist[0]:>10.5f}",
            f"  k2        {dist[1]:>10.5f}",
            "",
            "EXTRINSICS  (pos=static, R=per-frame)",
            f"  pos X     {cam_pos[0]:>10.3f} m",
            f"  pos Y     {cam_pos[1]:>10.3f} m",
            f"  pos Z     {cam_pos[2]:>10.3f} m",
            f"  elevation {elev:>8.2f} deg",
            f"  dist/ctr  {dist_fc:>8.2f} m",
            "",
            "LOOK DIRECTION  (world)",
            f"  dx        {look_dir[0]:>+10.4f}",
            f"  dy        {look_dir[1]:>+10.4f}",
            f"  dz        {look_dir[2]:>+10.4f}",
            f"  gnd hit   {gnd_str}",
        ]
        self.cam_info_actor.SetInput("\n".join(lines))

    def _update_images_dir(self, sequence_name: str, data_dir: str):
        """Point to the image folder for a sequence and check it exists."""
        img_dir = Path(data_dir) / 'images' / sequence_name
        self._images_dir  = img_dir
        self._has_images  = img_dir.is_dir() and bool(list(img_dir.glob('*.jpg')))

    def _update_video_bg(self, frame_idx: int):
        """Show the raw video frame as the renderer background texture.
        Only active when in broadcast-camera mode AND video is toggled on.
        When active the field is dimmed so the video pitch shows through."""
        if self._camera_follow and self._video_on and self._has_images:
            path = str(self._images_dir / f'{frame_idx:05d}.jpg')
            if Path(path).exists():
                self._bg_reader.SetFileName(path)
                self._bg_reader.Modified()
                self.renderer.TexturedBackgroundOn()
                self._set_field_opacity(0.15)   # mostly transparent over video
                return
        self.renderer.TexturedBackgroundOff()
        self._set_field_opacity(1.0)            # restore full opacity

    def _apply_camera_follow(self, frame_idx: int):
        """Snap VTK viewport to the real broadcast camera (position + FOV from K).

        Camera model notes
        ------------------
        * Position / orientation come from the per-frame R and t.
        * Vertical FOV is computed from the ACTUAL image height (1080 px) and fy.
        * Principal-point offset (cx, cy ≠ image-centre) is applied via
          SetWindowCenter so the 3D projection matches the real lens axis.
        * Aspect ratio is locked to 1920 / 1080 so the VTK projection matrix
          exactly matches the real camera, regardless of screen resolution.
        * Radial distortion (k1, k2) cannot be modelled by VTK's pinhole camera.
          Players near the image edges will have a small residual misalignment.
        """
        cam_pos, look_dir, up_dir = self._get_frame_camera(frame_idx)
        K = self._get_frame_K(frame_idx)

        vtk_cam = self.renderer.GetActiveCamera()
        vtk_cam.SetPosition(*cam_pos)
        vtk_cam.SetFocalPoint(*(cam_pos + look_dir * 30.0))
        vtk_cam.SetViewUp(*up_dir)

        # --- FOV: use true image height, not 2*cy ---
        IMG_W, IMG_H = 1920.0, 1080.0
        fy = K[1, 1]
        fov_y = float(np.degrees(2.0 * np.arctan(IMG_H / (2.0 * fy))))
        vtk_cam.SetViewAngle(fov_y)

        # --- Principal-point offset (shifts optical axis in viewport) ---
        cx, cy = K[0, 2], K[1, 2]
        # VTK WindowCenter: +x shifts view right, +y shifts view up
        # A pp displaced left of centre (cx < W/2) means the projection axis
        # is left of the image centre → shift the view window right to compensate.
        wc_x = (IMG_W / 2.0 - cx) / (IMG_W / 2.0)
        wc_y = -(IMG_H / 2.0 - cy) / (IMG_H / 2.0)
        vtk_cam.SetWindowCenter(wc_x, wc_y)

        # --- Lock aspect ratio to 1920/1080 ---
        self.renderer.SetAspect(IMG_W / IMG_H, 1.0)

        self.renderer.ResetCameraClippingRange()

    # ------------------------------------------------------------------
    # Per-frame skeleton update
    # ------------------------------------------------------------------

    def _update_player(self, slot: int, player_id: int,
                       skeleton: np.ndarray):
        """Move a player's pre-allocated geometry to the new skeleton pose."""
        data = self.players[slot]
        pts = data['pts']
        valid = ~np.isnan(skeleton).any(axis=1)

        # Fallback position for NaN joints: mean of valid joints (avoids
        # z=-2000 stretching artifacts in the tube filter)
        if valid.any():
            center = skeleton[valid].mean(axis=0)
        else:
            center = self.field_center.copy()

        for i in range(self.num_joints):
            if valid[i]:
                pts.SetPoint(i, skeleton[i, 0], skeleton[i, 1], skeleton[i, 2])
            else:
                pts.SetPoint(i, center[0], center[1], center[2])

        pts.Modified()
        data['bone_poly'].Modified()
        data['joint_poly'].Modified()

        # Label at head (nose joint, index 0), offset 0.3 m above
        if data['label'] is not None:
            lx, ly, lz = (skeleton[0] if valid[0] else center)
            data['label'].SetPosition(lx, ly, lz + 0.35)
            data['label'].SetInput(f'P{player_id}')
            data['label'].VisibilityOn()

        data['bone_actor'].VisibilityOn()
        data['joint_actor'].VisibilityOn()

    def _hide_player(self, slot: int):
        d = self.players[slot]
        d['bone_actor'].VisibilityOff()
        d['joint_actor'].VisibilityOff()
        if d['label'] is not None:
            d['label'].VisibilityOff()

    def _update_frame(self):
        frame_skels = self.skel_3d_world[self.current_frame]
        used: set = set()

        if frame_skels and isinstance(frame_skels[0], tuple):
            for person_idx, skel in frame_skels:
                slot = person_idx % MAX_PLAYERS
                self._update_player(slot, person_idx, skel)
                used.add(slot)
            n_players = len(frame_skels)
        else:
            for i, skel in enumerate(frame_skels):
                if skel is not None and skel.shape[0] > 0:
                    slot = i % MAX_PLAYERS
                    self._update_player(slot, i, skel)
                    used.add(slot)
            n_players = sum(1 for s in frame_skels
                            if s is not None and s.shape[0] > 0)

        for slot in range(MAX_PLAYERS):
            if slot not in used:
                self._hide_player(slot)

        # Move orange camera indicator / frustum (hidden in broadcast mode —
        # the VTK camera IS the broadcast camera so the frustum would just
        # radiate away from the viewer and be misleading)
        self._update_camera_indicator(self.current_frame)
        if self._camera_follow:
            self._frustum_actor.VisibilityOff()
        else:
            self._frustum_actor.VisibilityOn()

        # Live camera parameter panel
        self._update_cam_info_hud(self.current_frame)

        # Camera follow mode: VTK viewport tracks the real broadcast camera
        if self._camera_follow:
            self._apply_camera_follow(self.current_frame)
            video_label  = "  VIDEO ON" if self._video_on else ""
            follow_label = f"  [BROADCAST CAM{video_label}]"
        else:
            follow_label = "  [OVERVIEW]"

        # Update video background and bbox overlay (after _apply_camera_follow)
        self._update_video_bg(self.current_frame)
        self._update_bbox_overlay(self.current_frame)

        status = "▶" if self.playing else "⏸"
        mode_label = "  [YoloX]" if self._data_mode == 'yolox' else "  [Baseline]"
        self.title_actor.SetInput(
            f"{self.sequence_name}   Frame {self.current_frame} / "
            f"{self.num_frames - 1}   Players: {n_players}   {status}"
            f"{mode_label}{follow_label}")

        # Sync slider (suppress re-entrant callback via guard)
        self._in_slider_update = True
        self._slider_rep.SetValue(self.current_frame)
        self._in_slider_update = False

        self.render_window.Render()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    # Sequence switching
    # ------------------------------------------------------------------

    def _switch_sequence(self, name: str):
        """Reload all data for *name* without recreating the VTK window."""
        if self.playing:
            self._toggle_play()

        print(f"\nSwitching to sequence: {name}")
        self.sequence_name = name
        self.sequence_idx = self.sequences.index(name)

        (self.skel_3d_world, self.skel_2d, self.K, self.dist_coeffs,
         self.R, self.t, self.pitch_points,
         self.use_world_coords, self.boxes) = load_data(
             name, self._data_dir, self._submission_file,
             self._yolox_dir, self._data_mode)

        self.num_frames = len(self.skel_3d_world)
        self.current_frame = 0
        self._init_skeleton_topology()
        self._update_camera_geometry()
        self._update_images_dir(name, self._data_dir)
        self._video_on = False                        # reset video on sequence change

        # Update slider range
        self._slider_rep.SetMaximumValue(max(1, self.num_frames - 1))
        self._slider_rep.SetValue(0)

        # Hide all player actors (fresh start)
        for slot in range(MAX_PLAYERS):
            self._hide_player(slot)

        # Recalculate overview camera for the new camera position
        self._setup_camera()
        self._camera_follow = False

        # Update sequence list and frustum, then render frame 0
        self._update_sequence_panel()
        self._update_frame()
        self.renderer.ResetCameraClippingRange()
        self.render_window.SetWindowName(
            f"FIFA 3D Skeleton Viewer – {name}")
        print(f"Loaded {self.num_frames} frames.")

    # ------------------------------------------------------------------
    # Data-mode switching (Baseline ↔ YoloX)
    # ------------------------------------------------------------------

    def _switch_data_mode(self):
        """Toggle between 'baseline' and 'yolox' data; reload skeletons+boxes only."""
        self._data_mode = 'yolox' if self._data_mode == 'baseline' else 'baseline'
        print(f"\nSwitching data mode → {self._data_mode.upper()}")
        (self.skel_3d_world, self.skel_2d,
         self.use_world_coords, self.boxes) = load_skels_and_boxes(
             self.sequence_name, self._data_mode,
             self._data_dir, self._submission_file, self._yolox_dir)
        self.num_frames = len(self.skel_3d_world)
        self.current_frame = min(self.current_frame, self.num_frames - 1)
        self._init_skeleton_topology()
        self._slider_rep.SetMaximumValue(max(1, self.num_frames - 1))
        for slot in range(MAX_PLAYERS):
            self._hide_player(slot)
        self._update_frame()

    # ------------------------------------------------------------------

    def _on_key_press(self, obj, event):
        key = self.interactor.GetKeySym()
        if key == 'space':
            self._toggle_play()
        elif key == 'Left':
            self.current_frame = max(0, self.current_frame - 1)
            self._update_frame()
        elif key == 'Right':
            self.current_frame = min(self.num_frames - 1, self.current_frame + 1)
            self._update_frame()
        elif key == 'Up':
            self.current_frame = min(self.num_frames - 1, self.current_frame + 10)
            self._update_frame()
        elif key == 'Down':
            self.current_frame = max(0, self.current_frame - 10)
            self._update_frame()
        elif key == 'bracketleft':    # [ → previous sequence
            idx = (self.sequence_idx - 1) % len(self.sequences)
            self._switch_sequence(self.sequences[idx])
        elif key == 'bracketright':   # ] → next sequence
            idx = (self.sequence_idx + 1) % len(self.sequences)
            self._switch_sequence(self.sequences[idx])
        elif key in ('o', 'O'):
            self._camera_follow = not self._camera_follow
            if self._camera_follow:
                self._apply_camera_follow(self.current_frame)
            else:
                self._reset_to_overview()
            self._update_frame()
        elif key in ('r', 'R'):
            # Reset to overview (free navigation)
            self._camera_follow = False
            self._reset_to_overview()
            self._update_frame()
        elif key in ('v', 'V'):
            if self._camera_follow:
                self._video_on = not self._video_on
                self._update_frame()
        elif key in ('b', 'B'):
            if self._camera_follow:
                self._bbox_on = not self._bbox_on
                self._update_frame()
        elif key in ('y', 'Y'):
            self._switch_data_mode()
        elif key in ('q', 'Q', 'Escape'):
            self._quit()

    def _on_timer(self, obj, event):
        if self.playing:
            self.current_frame = (self.current_frame + 1) % self.num_frames
            self._update_frame()

    def _on_slider(self, obj, event):
        if getattr(self, '_in_slider_update', False):
            return
        val = int(round(self._slider_rep.GetValue()))
        if val != self.current_frame:
            self.current_frame = np.clip(val, 0, self.num_frames - 1)
            self._update_frame()

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _toggle_play(self):
        if self.playing:
            self.playing = False
            if self.timer_id is not None:
                self.interactor.DestroyTimer(self.timer_id)
                self.timer_id = None
        else:
            self.playing = True
            self.timer_id = self.interactor.CreateRepeatingTimer(
                self.PLAY_INTERVAL_MS)
        self._update_frame()

    def _reset_to_overview(self):
        """Switch VTK camera to the field overview position."""
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(*self._overview_pos)
        cam.SetFocalPoint(*self._overview_focal)
        cam.SetViewUp(*self._overview_up)
        cam.SetViewAngle(30.0)        # restore default VTK FOV
        cam.SetWindowCenter(0.0, 0.0) # clear principal-point offset
        self.renderer.SetAspect(1.0, 1.0)  # restore free aspect ratio
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()

    def _quit(self):
        if self.timer_id is not None:
            self.interactor.DestroyTimer(self.timer_id)
        self.render_window.Finalize()
        self.interactor.TerminateApp()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def show(self):
        """Open the interactive window (blocks until closed)."""
        self.render_window.Render()
        self.interactor.Initialize()
        self.interactor.Start()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Interactive VTK 3D skeleton visualization')
    parser.add_argument('--sequence', '-s', type=str, default='ARG_CRO_225412', 
                        help='Sequence name, e.g. ARG_CRO_225412')
    parser.add_argument('--data_dir', '-d', type=str, default='data',
                        help='Path to the data directory')
    parser.add_argument('--submission', type=str,
                        default='output/submission_full.npz',
                        help='Path to baseline submission NPZ (world-coord predictions)')
    parser.add_argument('--yolox_dir', type=str,
                        default='YoloX+DeepEIOU_Data',
                        help='Path to YoloX+DeepEIOU data directory')
    parser.add_argument('--frame', '-f', type=int, default=0,
                        help='Starting frame index (default: 0)')
    args = parser.parse_args()

    viz = VTK3DVisualizer(args.sequence, args.data_dir, args.submission,
                          args.yolox_dir)
    viz.current_frame = args.frame
    viz._update_frame()
    viz.show()


if __name__ == '__main__':
    main()
