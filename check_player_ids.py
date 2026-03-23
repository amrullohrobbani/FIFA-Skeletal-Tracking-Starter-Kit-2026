"""
Check how player IDs are assigned across frames in the submission data.
This demonstrates why player IDs/colors change between frames.
"""
import numpy as np

# Load submission data
submission_data = np.load('outputs/submission_full.npz')
seq_data = submission_data['ARG_CRO_225412']  # Shape: (21, 569, 15, 3)

print("Submission data shape:", seq_data.shape)
print("Format: (21 person slots, 569 frames, 15 joints, 3 coords)\n")

# Check which person slots have valid data in different frames
frames_to_check = [0, 1, 2, 50, 100, 200]

for frame_idx in frames_to_check:
    valid_people = []
    for person_idx in range(21):
        skel = seq_data[person_idx, frame_idx]
        # Check if person is valid (not all zeros/nans)
        if not np.all(skel == 0) and not np.all(np.isnan(skel)):
            # Get pelvis position (joint 7 in 15-joint format)
            pelvis = skel[7]
            valid_people.append((person_idx, pelvis))
    
    print(f"Frame {frame_idx}:")
    print(f"  Valid person slots in submission: {[p[0] for p in valid_people]}")
    print(f"  Total valid: {len(valid_people)}")
    if len(valid_people) > 0:
        print(f"  Sample positions:")
        for i, (orig_idx, pos) in enumerate(valid_people[:3]):
            print(f"    Submission slot {orig_idx} → Display ID {i} → Position: X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f}")
    print()

print("\n" + "="*70)
print("PROBLEM:")
print("="*70)
print("The current system filters out empty slots per frame, which loses")
print("the original person_idx from the submission array.")
print()
print("Example:")
print("  Frame 0: Submission slots [0, 2, 5, 7, 10] → Display IDs [0, 1, 2, 3, 4]")
print("  Frame 1: Submission slots [1, 2, 6, 8, 10] → Display IDs [0, 1, 2, 3, 4]")
print()
print("Person in submission slot 10 appears in both frames:")
print("  - Gets color/label 'P4' in both frames (good!)")
print()
print("But person in submission slot 0 (Frame 0) is different from")
print("person in submission slot 1 (Frame 1):")
print("  - Both get color/label 'P0' (BAD - should have different IDs!)")
print()
print("SOLUTION:")
print("  Preserve the original submission person_idx (0-20) as the display ID")
print("  instead of using the filtered list index.")
