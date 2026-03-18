"""
FIFA Skeletal Tracking Challenge - Evaluation Metrics

This script implements the official evaluation metrics:
- Global MPJPE: Mean Per Joint Position Error (absolute)
- Local MPJPE: Mean Per Joint Position Error relative to root joint

Final Score = Global MPJPE + 5 × Local MPJPE

NOTE: Ground truth 3D poses are NOT provided in this repo.
      You can only get official scores by submitting to Codabench.
      
This script is provided for:
- Understanding the metric computation
- Testing on external datasets that have GT
- Validating your implementation before submission
"""

import numpy as np
from pathlib import Path
from typing import Dict
import argparse


def compute_global_mpjpe(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Compute Global MPJPE (Mean Per Joint Position Error).
    
    Measures absolute error between predicted and ground truth joint positions.
    
    Args:
        pred: Predicted 3D poses, shape (N_players, N_frames, 15, 3)
        gt: Ground truth 3D poses, shape (N_players, N_frames, 15, 3)
        mask: Valid joint mask, shape (N_players, N_frames, 15) or None
              True = valid joint, False = ignore (e.g., occluded, out of frame)
    
    Returns:
        Global MPJPE in millimeters (mm)
    """
    # Compute Euclidean distance per joint
    errors = np.sqrt(np.sum((pred - gt) ** 2, axis=-1))  # (N_players, N_frames, 15)
    
    if mask is not None:
        # Only compute error for valid joints
        valid_errors = errors[mask]
        if len(valid_errors) == 0:
            return 0.0
        return np.mean(valid_errors)
    else:
        return np.mean(errors)


def compute_local_mpjpe(pred: np.ndarray, gt: np.ndarray, 
                        root_joint_idx: int = 8, 
                        mask: np.ndarray = None) -> float:
    """
    Compute Local MPJPE (Mean Per Joint Position Error relative to root).
    
    Measures error after centering both prediction and GT at the root joint.
    No scaling or rotation is applied (unlike PA-MPJPE).
    
    Args:
        pred: Predicted 3D poses, shape (N_players, N_frames, 15, 3)
        gt: Ground truth 3D poses, shape (N_players, N_frames, 15, 3)
        root_joint_idx: Index of root joint (default: 8 = Pelvis in BODY25)
        mask: Valid joint mask, shape (N_players, N_frames, 15) or None
    
    Returns:
        Local MPJPE in millimeters (mm)
    """
    # Center at root joint
    pred_root = pred[:, :, root_joint_idx:root_joint_idx+1, :]  # Keep dims for broadcasting
    gt_root = gt[:, :, root_joint_idx:root_joint_idx+1, :]
    
    pred_centered = pred - pred_root
    gt_centered = gt - gt_root
    
    # Compute Euclidean distance per joint
    errors = np.sqrt(np.sum((pred_centered - gt_centered) ** 2, axis=-1))
    
    if mask is not None:
        valid_errors = errors[mask]
        if len(valid_errors) == 0:
            return 0.0
        return np.mean(valid_errors)
    else:
        return np.mean(errors)


def compute_final_score(global_mpjpe: float, local_mpjpe: float) -> float:
    """
    Compute final challenge score.
    
    Final Score = Global MPJPE + 5 × Local MPJPE
    
    The weighting factor 5 ensures both terms contribute meaningfully,
    as Local MPJPE is typically an order of magnitude smaller.
    """
    return global_mpjpe + 5.0 * local_mpjpe


def evaluate_sequence(pred: np.ndarray, gt: np.ndarray, 
                      sequence_name: str,
                      mask: np.ndarray = None) -> Dict[str, float]:
    """
    Evaluate a single sequence.
    
    Args:
        pred: Predicted 3D poses, shape (N_players, N_frames, 15, 3)
        gt: Ground truth 3D poses, shape (N_players, N_frames, 15, 3)
        sequence_name: Name of the sequence
        mask: Valid joint mask
    
    Returns:
        Dictionary with metrics
    """
    assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"
    
    global_mpjpe = compute_global_mpjpe(pred, gt, mask)
    local_mpjpe = compute_local_mpjpe(pred, gt, root_joint_idx=8, mask=mask)
    final_score = compute_final_score(global_mpjpe, local_mpjpe)
    
    return {
        'sequence': sequence_name,
        'global_mpjpe': global_mpjpe,
        'local_mpjpe': local_mpjpe,
        'final_score': final_score,
        'n_valid_joints': np.sum(mask) if mask is not None else pred.size // 3
    }


def evaluate_submission(predictions_path: Path, 
                       ground_truth_path: Path,
                       sequences_file: Path = None) -> Dict[str, float]:
    """
    Evaluate entire submission.
    
    Args:
        predictions_path: Path to predictions NPZ file (e.g., submission_full.npz)
        ground_truth_path: Path to ground truth NPZ file
        sequences_file: Optional file listing which sequences to evaluate
    
    Returns:
        Dictionary with overall metrics
    """
    # Load predictions and ground truth
    predictions = np.load(predictions_path)
    ground_truth = np.load(ground_truth_path)
    
    # Get sequence names
    if sequences_file is not None:
        with open(sequences_file) as f:
            sequences = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        sequences = list(predictions.keys())
    
    # Evaluate each sequence
    results = []
    total_valid_joints = 0
    weighted_global_mpjpe = 0.0
    weighted_local_mpjpe = 0.0
    
    print("\n" + "="*80)
    print(f"{'Sequence':<20} {'Global MPJPE':>15} {'Local MPJPE':>15} {'Final Score':>15}")
    print("="*80)
    
    for seq_name in sequences:
        if seq_name not in predictions:
            print(f"Warning: {seq_name} not found in predictions, skipping...")
            continue
        if seq_name not in ground_truth:
            print(f"Warning: {seq_name} not found in ground truth, skipping...")
            continue
        
        pred = predictions[seq_name]
        gt = ground_truth[seq_name]
        
        # Create mask for valid joints (not NaN)
        mask = ~np.isnan(gt).any(axis=-1)  # (N_players, N_frames, 15)
        
        metrics = evaluate_sequence(pred, gt, seq_name, mask)
        results.append(metrics)
        
        # Accumulate weighted metrics
        n_valid = metrics['n_valid_joints']
        total_valid_joints += n_valid
        weighted_global_mpjpe += metrics['global_mpjpe'] * n_valid
        weighted_local_mpjpe += metrics['local_mpjpe'] * n_valid
        
        print(f"{seq_name:<20} {metrics['global_mpjpe']:>15.2f} "
              f"{metrics['local_mpjpe']:>15.2f} {metrics['final_score']:>15.2f}")
    
    # Compute overall metrics (weighted by number of valid joints)
    if total_valid_joints > 0:
        overall_global = weighted_global_mpjpe / total_valid_joints
        overall_local = weighted_local_mpjpe / total_valid_joints
        overall_final = compute_final_score(overall_global, overall_local)
    else:
        overall_global = overall_local = overall_final = 0.0
    
    print("="*80)
    print(f"{'OVERALL':<20} {overall_global:>15.2f} "
          f"{overall_local:>15.2f} {overall_final:>15.2f}")
    print("="*80)
    print(f"\nFinal Challenge Score: {overall_final:.2f} mm")
    print(f"  - Global MPJPE:  {overall_global:.2f} mm")
    print(f"  - Local MPJPE:   {overall_local:.2f} mm (×5 weight)")
    print(f"  - Total valid joints evaluated: {total_valid_joints:,}\n")
    
    return {
        'overall_global_mpjpe': overall_global,
        'overall_local_mpjpe': overall_local,
        'overall_final_score': overall_final,
        'per_sequence': results,
        'total_valid_joints': total_valid_joints
    }


def create_dummy_ground_truth(predictions_path: Path, output_path: Path):
    """
    Create dummy ground truth for testing (adds random noise to predictions).
    
    THIS IS ONLY FOR TESTING THE EVALUATION SCRIPT!
    Real ground truth is held by competition organizers.
    """
    predictions = np.load(predictions_path)
    
    dummy_gt = {}
    for seq_name, pred in predictions.items():
        # Add 50mm random noise to create "ground truth"
        noise = np.random.randn(*pred.shape) * 50.0
        dummy_gt[seq_name] = pred + noise
    
    np.savez_compressed(output_path, **dummy_gt)
    print(f"Created dummy ground truth at: {output_path}")
    print("NOTE: This is FAKE data for testing only!")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FIFA Skeletal Tracking predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with ground truth (if you have it)
  python evaluate_metrics.py --predictions outputs/submission_full.npz \\
                             --ground_truth data/ground_truth.npz

  # Create dummy GT for testing the script
  python evaluate_metrics.py --predictions outputs/submission_full.npz \\
                             --create_dummy_gt
        """
    )
    
    parser.add_argument('--predictions', type=Path, required=True,
                       help='Path to predictions NPZ file')
    parser.add_argument('--ground_truth', type=Path,
                       help='Path to ground truth NPZ file')
    parser.add_argument('--sequences', type=Path,
                       help='File listing sequences to evaluate (e.g., sequences_val.txt)')
    parser.add_argument('--create_dummy_gt', action='store_true',
                       help='Create dummy ground truth for testing (adds noise to predictions)')
    
    args = parser.parse_args()
    
    if not args.predictions.exists():
        print(f"Error: Predictions file not found: {args.predictions}")
        return
    
    if args.create_dummy_gt:
        # Create dummy ground truth for testing
        dummy_gt_path = args.predictions.parent / "dummy_ground_truth.npz"
        create_dummy_ground_truth(args.predictions, dummy_gt_path)
        print("\nNow you can test evaluation with:")
        print(f"python evaluate_metrics.py --predictions {args.predictions} \\")
        print(f"                           --ground_truth {dummy_gt_path}")
        return
    
    if args.ground_truth is None:
        print("\n" + "="*80)
        print("ERROR: Ground truth not provided!")
        print("="*80)
        print("\nThis repo does NOT include ground truth 3D poses.")
        print("You can only get official scores by submitting to Codabench:")
        print("  - Validation: https://www.codabench.org/competitions/11681/")
        print("  - Challenge:  https://www.codabench.org/competitions/11682/")
        print("\nTo test this script with dummy data, run:")
        print(f"  python evaluate_metrics.py --predictions {args.predictions} --create_dummy_gt")
        return
    
    if not args.ground_truth.exists():
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        return
    
    # Evaluate submission
    results = evaluate_submission(
        predictions_path=args.predictions,
        ground_truth_path=args.ground_truth,
        sequences_file=args.sequences
    )
    
    # Save results to JSON
    import json
    results_path = args.predictions.parent / "evaluation_results.json"
    
    # Convert numpy types to Python types for JSON serialization
    json_results = {
        'overall_global_mpjpe': float(results['overall_global_mpjpe']),
        'overall_local_mpjpe': float(results['overall_local_mpjpe']),
        'overall_final_score': float(results['overall_final_score']),
        'total_valid_joints': int(results['total_valid_joints']),
        'per_sequence': [
            {k: float(v) if isinstance(v, (np.floating, float)) else v 
             for k, v in seq.items()}
            for seq in results['per_sequence']
        ]
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Detailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
