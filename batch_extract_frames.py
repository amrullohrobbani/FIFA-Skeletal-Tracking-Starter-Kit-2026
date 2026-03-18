"""
Batch extract frames from all videos in data/videos/
Creates images in data/images/ folder with same structure as expected by main.py
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def extract_frames(video_path, output_folder, save_as_png=False):
    """
    Extract frames from a video file
    
    Args:
        video_path: Path to input video file
        output_folder: Path to output folder for frames
        save_as_png: Save as PNG instead of JPG
    """
    # Create the output directory if it doesn't exist
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    
    # Get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    with tqdm(total=total_frames, desc=f"Extracting {video_path.stem}", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit loop if no frame is returned (end of video)
            
            # Save each frame as an image file
            if save_as_png:
                output_filename = output_folder / f'{frame_count:05d}.png'
            else:
                output_filename = output_folder / f'{frame_count:05d}.jpg'
            
            # Convert BGR to RGB and save with PIL (more efficient)
            Image.fromarray(frame[..., ::-1]).save(str(output_filename), optimize=True, quality=95)
            frame_count += 1
            pbar.update(1)

    cap.release()
    print(f'✅ {frame_count} frames extracted and saved in "{output_folder}"')
    return frame_count


def batch_extract_frames(video_dir, output_dir, save_as_png=False, sequences_file=None):
    """
    Extract frames from all videos in a directory
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save extracted frames
        save_as_png: Save as PNG instead of JPG
        sequences_file: Optional file with list of sequences to process
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    
    # Get list of sequences to process
    if sequences_file:
        with open(sequences_file, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
    else:
        # Process all videos in directory
        sequences = [v.stem for v in video_dir.glob("*.mp4")]
    
    print(f"Found {len(sequences)} videos to process")
    print("="*80)
    
    results = []
    for idx, sequence in enumerate(sequences, 1):
        video_path = video_dir / f"{sequence}.mp4"
        
        if not video_path.exists():
            print(f"⚠️  Video not found: {video_path}")
            continue
        
        output_folder = output_dir / sequence
        
        print(f"\n[{idx}/{len(sequences)}] Processing: {sequence}")
        print(f"  Video: {video_path}")
        print(f"  Output: {output_folder}")
        
        try:
            frame_count = extract_frames(video_path, output_folder, save_as_png)
            results.append((sequence, frame_count, "✅ Success"))
        except Exception as e:
            print(f"❌ Error processing {sequence}: {e}")
            results.append((sequence, 0, f"❌ Error: {e}"))
    
    # Print summary
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    for sequence, frame_count, status in results:
        print(f"{sequence:30s} {frame_count:6d} frames  {status}")
    
    total_frames = sum(r[1] for r in results)
    successful = sum(1 for r in results if r[2].startswith("✅"))
    print("="*80)
    print(f"Total: {successful}/{len(sequences)} videos, {total_frames:,} frames extracted")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch extract frames from videos")
    parser.add_argument("--video_dir", type=str, default="data/videos",
                       help="Directory containing video files")
    parser.add_argument("--output_dir", type=str, default="data/images",
                       help="Directory to save extracted frames")
    parser.add_argument("--png", action="store_true",
                       help="Save frames as PNG instead of JPG")
    parser.add_argument("--sequences", type=str,
                       help="File with list of sequences to process (e.g., data/sequences_val.txt)")
    
    args = parser.parse_args()
    
    batch_extract_frames(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        save_as_png=args.png,
        sequences_file=args.sequences
    )
