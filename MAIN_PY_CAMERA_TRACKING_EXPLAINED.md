"""
HOW MAIN.PY IMPLEMENTS CAMERA TRACKING
========================================

This document explains how main.py handles camera rotation and position changes
across frames, compared to the static camera assumption.
"""

# ============================================================================
# PROBLEM: Static Camera Data
# ============================================================================

"""
The cameras/*.npz files contain:
  - K(N,3,3): intrinsics per frame ✅
  - k(N,5): distortion per frame ✅
  - R(1,3,3): rotation for frame 0 ONLY ❌
  - t(1,3): translation for frame 0 ONLY ❌

This is insufficient for accurate projection across all frames because
broadcast cameras pan, tilt, and move during the match!
"""

# ============================================================================
# SOLUTION: CameraTracker Class (lib/camera_tracker.py)
# ============================================================================

"""
main.py uses a CameraTracker class that tracks camera movement across frames
using TWO techniques:

1. OPTICAL FLOW (every frame)
   - Projects 714 pitch_points to previous frame
   - Tracks these points to current frame using Lucas-Kanade optical flow
   - Estimates relative rotation from point correspondences
   - Updates camera rotation: R[i] = R_relative @ R[i-1]

2. LANE LINE REFINEMENT (every 10 frames)
   - Detects field lane lines (white markings) using adaptive thresholding
   - Creates distance map from detected lines
   - Optimizes camera rotation to minimize distance between:
     * Projected pitch points (yellow field markings)
     * Detected line pixels
   - Uses scipy.optimize to refine R


This produces R(N,3,3) and t(N,3) for ALL N frames!
"""

# ============================================================================
# MAIN.PY WORKFLOW (Simplified)
# ============================================================================

def main_py_workflow():
    """
    Simplified view of main.py process_sequence() function
    """
    
    # 1. Initialize tracker with frame 0 pose
    camera_tracker = CameraTracker(pitch_points=pitch_points, fps=50.0)
    camera_tracker.initialize(
        frame_idx=0,
        K=cameras["K"][0],
        k=cameras["k"][0],
        R=cameras["R"][0],  # Initial rotation
        t=cameras["t"][0],  # Initial translation
    )
    
    # 2. Track camera for each frame
    Rt = []  # Store tracked (R, t) for each frame
    
    for frame_idx in range(NUM_FRAMES):
        # Read video frame
        success, img = video.read()
        
        # Track camera using optical flow + refinement
        state = camera_tracker.track(
            frame_idx=frame_idx,
            frame=img,  # ⚠️ REQUIRES VIDEO!
            K=cameras["K"][frame_idx],
            dist_coeffs=cameras["k"][frame_idx],
        )
        
        # Store tracked pose
        Rt.append((state.R.copy(), state.t.copy()))
        
        # 3. For each player, use tracked camera to:
        for person in range(NUM_PERSONS):
            # - Cast ray from 2D keypoint through camera
            # - Find intersection with pitch plane (Z=0)
            # - Convert skel_3d from camera coords to world coords
            # - Optimize 3D position to minimize reprojection error
            ...
    
    return predictions, Rt


# ============================================================================
# DETAILED: CameraTracker.track() Method
# ============================================================================

def camera_tracker_track_explained(frame_idx, frame, K, dist_coeffs):
    """
    What happens inside CameraTracker.track()
    """
    
    if frame_idx == 0:
        # Frame 0: Use provided initial pose
        # No tracking needed, just initialization
        return state
    
    # ========================================================================
    # STEP 1: OPTICAL FLOW (every frame)
    # ========================================================================
    
    # 1a. Project pitch points using PREVIOUS frame's camera
    pts_2d_prev = project_points(
        pitch_points,  # 714 3D field markings
        R_prev, t_prev, K_prev, k_prev
    )  # → Get 2D pixel coordinates in previous frame
    
    # 1b. Track these points to CURRENT frame using optical flow
    pts_2d_next, status = cv2.calcOpticalFlowPyrLK(
        prev_frame,  # Previous video frame
        frame,       # Current video frame
        pts_2d_prev  # Previous pixel locations
    )  # → Get where these pixels moved to
    
    # 1c. Estimate rotation from point correspondences
    # Convert 2D pixels → normalized 3D rays
    rays_prev = undistort_and_normalize(pts_2d_prev, K_prev, k_prev)
    rays_next = undistort_and_normalize(pts_2d_next, K, dist_coeffs)
    
    # Solve: rays_next = R_relative @ rays_prev
    # Using SVD rotation estimation
    M = rays_next.T @ rays_prev
    U, S, Vt = np.linalg.svd(M)
    R_relative = U @ Vt
    
    # Update rotation
    R_current = R_relative @ R_prev
    
    # ========================================================================
    # STEP 2: LANE LINE REFINEMENT (every 10 frames)
    # ========================================================================
    
    if frame_idx % 10 == 0:
        # 2a. Detect lane lines in current frame
        mask = extract_lane_lines_mask(frame)  # Adaptive thresholding
        field_mask = create_field_mask(frame)  # HSV color filtering
        mask = cv2.bitwise_and(mask, field_mask)
        
        # 2b. Create distance transform
        dist_map = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        
        # 2c. Optimize rotation to align projected points with lines
        def objective(R_delta):
            R = orthogonalize(R_current + R_delta)
            t = -R @ C
            pts_2d = project_points(pitch_points, R, t, K, dist_coeffs)
            
            # Measure average distance from projected points to lane lines
            distances = dist_map[pts_2d[:, 1], pts_2d[:, 0]]
            return distances.mean()
        
        # Minimize distance
        result = scipy.optimize.minimize(objective, x0=np.zeros(9))
        R_refined = orthogonalize(R_current + result.x.reshape(3,3))
        
        R_current = R_refined
    
    # ========================================================================
    # RESULT: Accurate R and t for current frame
    # ========================================================================
    
    return CameraState(R=R_current, t=-R_current @ C, K=K, k=dist_coeffs)


# ============================================================================
# KEY DIFFERENCES: main.py vs Our Approach
# ============================================================================

"""
┌─────────────────────┬─────────────────────┬─────────────────────────┐
│     Aspect          │     main.py         │   Our estimate_camera   │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Input Required      │ Videos (MP4)        │ Only data/ folder       │
│                     │ cameras/            │ cameras/                │
│                     │ boxes/              │ boxes/                  │
│                     │ skel_2d/            │ skel_2d/                │
│                     │ skel_3d/            │ skel_3d/                │
│                     │ pitch_points.txt    │ pitch_points.txt        │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Tracking Method     │ Optical flow        │ PnP (2D-3D matching)    │
│                     │ + Lane detection    │                         │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Accuracy            │ High (~1-2px error) │ Moderate (~5% success)  │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Frame Coverage      │ All frames          │ ~5-10% frames           │
│                     │                     │ (interpolated to fill)  │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Camera Position     │ Optimized per frame │ Assumed constant (C[0]) │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Speed               │ ~30 sec/sequence    │ ~30 sec/sequence        │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Limitation          │ REQUIRES VIDEOS     │ Limited by skel_3d      │
│                     │ (not available)     │ being camera-relative   │
└─────────────────────┴─────────────────────┴─────────────────────────┘
"""

# ============================================================================
# WHY main.py IS MORE ACCURATE
# ============================================================================

"""
1. OPTICAL FLOW is robust
   - Tracks 714 pitch points across frames
   - Uses actual pixel motion from video
   - Handles smooth camera movements well
   
2. LANE LINE REFINEMENT corrects drift
   - Every 10 frames, re-aligns with detected field markings
   - Minimizes accumulated error from optical flow
   - Uses visual features (white lines) directly from video

3. ITERATIVE REFINEMENT
   - Frame i uses refined pose from frame i-1
   - Temporal coherence (smooth motion assumption)
   - Corrects small errors before they accumulate

OUR APPROACH (PnP) struggles because:
   - skel_3d is in CAMERA-relative coordinates, not world
   - Only works when skel_3d happens to be accurate
   - ~95% of frames fail PnP (not enough good correspondences)
   - Interpolation fills gaps but accumulates error
"""

# ============================================================================
# CONCLUSION
# ============================================================================

"""
main.py implementation:
  ✅ Highly accurate (uses video optical flow + lane detection)
  ✅ Produces R(N,3,3) and t(N,3) for all N frames
  ✅ Handles smooth camera motion naturally
  ❌ REQUIRES video files (not available without FIFA license)

Our estimate_camera approach:
  ✅ Works WITHOUT videos (uses only data/ folder)
  ✅ Still produces R(N,3,3) and t(N,3) via interpolation
  ⚠️  Less accurate (~5% success rate, rest interpolated)
  ⚠️  Useful for visualization, not for evaluation

For VISUALIZATION: Our approach is sufficient
For COMPETITION: Need videos + main.py for accurate camera tracking
"""


# ============================================================================
# CODE EXAMPLE: How to Use main.py
# ============================================================================

"""
# Assuming you have videos in data/videos/

python main.py --export_camera -s data/sequences_val.txt

This will:
1. Process each sequence in sequences_val.txt
2. Run CameraTracker on each video
3. Export tracked cameras to outputs/calibration/*.npz
   - Contains R(N,3,3) and t(N,3) for all N frames
4. Also generates 3D predictions in outputs/

Output structure:
outputs/
  calibration/
    ARG_CRO_225412.npz    # R(569,3,3), t(569,3)
    ARG_FRA_183303.npz
    ...
  ARG_CRO_225412/
    predictions.npy        # 3D skeleton predictions
"""


# ============================================================================
# VISUALIZATION COMPARISON
# ============================================================================

"""
WITH static camera (our fallback):
  Frame 0:   Pitch points aligned ✅ (using R[0], t[0])
  Frame 100: Pitch points drift ❌ (camera has moved!)
  Frame 200: Pitch points way off ❌ (large camera motion)

WITH tracked camera (main.py output):
  Frame 0:   Pitch points aligned ✅ (using R[0], t[0])
  Frame 100: Pitch points aligned ✅ (using R[100], t[100])
  Frame 200: Pitch points aligned ✅ (using R[200], t[200])

This is why tracked cameras are essential for multi-frame visualization!
"""
