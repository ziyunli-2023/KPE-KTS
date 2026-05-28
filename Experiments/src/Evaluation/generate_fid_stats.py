"""
Generate FID reference statistics from CelebA test set.
This script creates multiple .npz files containing InceptionV3 feature statistics
for non-overlapping subsets of the test set.

Usage:
    cd Experiments/src/Evaluation
    python generate_fid_stats.py --size 32 --num_per_stats 10000 --num_stats 2 --seed 42
"""

import os
import csv
import random
import shutil
import subprocess
import argparse
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
import numpy as np
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3


def load_test_images_from_partition(partition_file):
    """
    Read partition file and return list of test image filenames.

    Args:
        partition_file: Path to list_eval_partition.csv

    Returns:
        List of test image filenames (partition == 2)
    """
    test_images = []

    if not os.path.exists(partition_file):
        raise FileNotFoundError(f"Partition file not found: {partition_file}")

    print(f"Loading test images from: {partition_file}")
    with open(partition_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['partition'] == '2':  # Test images
                test_images.append(row['image_id'])

    print(f"Found {len(test_images)} test images")
    return test_images


def process_and_save_images(image_names, input_dir, output_dir, size):
    """
    Load, preprocess, and save images to output directory.

    Args:
        image_names: List of image filenames to process
        input_dir: Directory containing original jpg images
        output_dir: Directory to save processed images
        size: Target image size
    """
    # Define transformation (MUST match preprocess_celeba.py for consistency!)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale like training data
        transforms.ToTensor(),  # Converts to [0, 1] range
    ])

    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(image_names)} images...")
    for idx, img_name in enumerate(tqdm(image_names)):
        img_path = os.path.join(input_dir, img_name)

        try:
            # Load image
            img = Image.open(img_path).convert('RGB')

            # Apply transformations
            img_tensor = transform(img)

            # InceptionV3 expects 3-channel RGB images
            # Convert 1-channel grayscale to 3-channel by repeating
            if img_tensor.shape[0] == 1:
                img_tensor = img_tensor.repeat(3, 1, 1)

            # Save as PNG
            output_path = os.path.join(output_dir, f'{idx:06d}.png')
            vutils.save_image(img_tensor, output_path)

        except Exception as e:
            print(f"\nError processing {img_name}: {e}")
            continue

    print(f"Saved {len(image_names)} images to {output_dir}")


def compute_fid_stats(image_dir, output_stats_file, device='cuda:0'):
    """
    Compute FID statistics using pytorch_fid.

    Args:
        image_dir: Directory containing images
        output_stats_file: Path to save statistics (.npz file)
        device: Device to use for computation
    """
    print(f"\nComputing FID statistics...")
    print(f"  Input: {image_dir}")
    print(f"  Output: {output_stats_file}")
    print(f"  Device: {device}")

    try:
        # Set device
        if device.startswith('cuda'):
            device_obj = torch.device(device)
        else:
            device_obj = torch.device('cpu')

        # Get list of image files
        print("Getting list of image files...")
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        image_files = []
        for fname in sorted(os.listdir(image_dir)):
            if os.path.splitext(fname.lower())[1] in image_extensions:
                image_files.append(os.path.join(image_dir, fname))

        print(f"Found {len(image_files)} image files")

        if len(image_files) == 0:
            raise ValueError(f"No image files found in {image_dir}")

        # Load InceptionV3 model
        print("Loading InceptionV3 model...")
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(device_obj)
        model.eval()

        # Compute statistics
        print("Computing statistics from images...")
        mu, sigma = fid_score.calculate_activation_statistics(
            image_files,
            model,
            batch_size=50,
            dims=2048,
            device=device_obj,
            num_workers=1
        )

        # Save statistics
        print(f"Saving statistics to {output_stats_file}...")
        np.savez_compressed(output_stats_file, mu=mu, sigma=sigma)

        print("Statistics computed successfully!")
        print(f"  Mean shape: {mu.shape}")
        print(f"  Covariance shape: {sigma.shape}")

    except Exception as e:
        print(f"Error computing statistics: {e}")
        import traceback
        traceback.print_exc()
        raise


def generate_fid_stats(args):
    """
    Main function to generate FID reference statistics.
    """
    print("="*80)
    print("Generating FID Reference Statistics")
    print("="*80)
    print(f"Configuration:")
    print(f"  Image size: {args.size}x{args.size}")
    print(f"  Images per stats file: {args.num_per_stats}")
    print(f"  Number of stats files: {args.num_stats}")
    print(f"  Random seed: {args.seed}")
    print(f"  Device: {args.device}")
    print("="*80)

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"\nRandom seed set to: {args.seed}")

    # Load test images from partition file
    test_images = load_test_images_from_partition(args.partition_file)

    # Check if we have enough images for at least the first batch
    min_needed = args.num_per_stats
    if min_needed > len(test_images):
        raise ValueError(
            f"Not enough test images! Need at least {min_needed} "
            f"for the first batch, but only have {len(test_images)}"
        )

    # Calculate how many images will be in the last batch
    if args.num_stats > 1:
        remaining = len(test_images) - (args.num_stats - 1) * args.num_per_stats
        if remaining < 0:
            raise ValueError(
                f"Not enough test images! Need at least {(args.num_stats - 1) * args.num_per_stats} "
                f"for {args.num_stats - 1} batches, but only have {len(test_images)}"
            )
        print(f"\nBatch distribution:")
        for i in range(args.num_stats - 1):
            print(f"  Batch {i+1}: {args.num_per_stats} images")
        print(f"  Batch {args.num_stats}: {remaining} images (all remaining)")
    else:
        print(f"\nBatch distribution:")
        print(f"  Batch 1: {min(args.num_per_stats, len(test_images))} images")

    # Randomly shuffle test images
    random.shuffle(test_images)
    print(f"\nShuffled {len(test_images)} test images")

    # Create output directory for stats
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate each stats file
    for i in range(args.num_stats):
        print(f"\n{'='*80}")
        print(f"Generating stats{i+1}.npz ({i+1}/{args.num_stats})")
        print(f"{'='*80}")

        # Select images for this batch
        start_idx = i * args.num_per_stats

        # For the last batch, use all remaining images
        if i == args.num_stats - 1:
            batch_images = test_images[start_idx:]
            end_idx = len(test_images)
        else:
            end_idx = start_idx + args.num_per_stats
            batch_images = test_images[start_idx:end_idx]

        print(f"Selected images {start_idx+1} to {end_idx} from shuffled test set ({len(batch_images)} images)")

        # Create temporary directory for processed images
        temp_dir = os.path.join(args.output_dir, f'temp_images_{i+1}')
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        try:
            # Process and save images
            process_and_save_images(batch_images, args.input_dir, temp_dir, args.size)

            # Compute and save FID statistics
            output_stats = os.path.join(args.output_dir, f'stats{i+1}.npz')
            compute_fid_stats(temp_dir, output_stats, args.device)

            # Check file size
            if os.path.exists(output_stats):
                size_mb = os.path.getsize(output_stats) / (1024**2)
                print(f"Stats file size: {size_mb:.2f} MB")

        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                print(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir)

    print(f"\n{'='*80}")
    print("All stats files generated successfully!")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    # List generated files
    print("\nGenerated files:")
    for i in range(args.num_stats):
        stats_file = os.path.join(args.output_dir, f'stats{i+1}.npz')
        if os.path.exists(stats_file):
            size_mb = os.path.getsize(stats_file) / (1024**2)
            print(f"  - stats{i+1}.npz ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate FID reference statistics from CelebA test set'
    )

    parser.add_argument('--size', type=int, default=32,
                        help='Target image size (default: 32)')
    parser.add_argument('--num_per_stats', type=int, default=10000,
                        help='Number of images per stats file (default: 10000)')
    parser.add_argument('--num_stats', type=int, default=2,
                        help='Number of stats files to generate (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--input_dir', type=str,
                        default='/cephyr/users/ziyunl/Alvis/code/Why-Diffusion-Models-Don-t-Memorize/Experiments/Data/img_align_celeba',
                        help='Directory containing jpg images')
    parser.add_argument('--partition_file', type=str,
                        default='/cephyr/users/ziyunl/Alvis/code/Why-Diffusion-Models-Don-t-Memorize/Experiments/Data/list_eval_partition.csv',
                        help='Path to partition file')
    parser.add_argument('--output_dir', type=str,
                        default='../../Saves/FID_ref/',
                        help='Directory to save stats files (default: ../../Saves/FID_ref/)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not os.path.exists(args.partition_file):
        raise FileNotFoundError(f"Partition file not found: {args.partition_file}")

    # Generate stats
    generate_fid_stats(args)


if __name__ == "__main__":
    main()
