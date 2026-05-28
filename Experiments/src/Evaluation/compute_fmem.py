"""
Compute fraction collapsed (memorization metric) for Flow Matching models.

The metric uses k-NN gap ratio analysis to estimate the fraction of generated
samples that have collapsed onto a training-set neighbour. This script consumes
samples written by `generate.py` (with or without KTS).
"""

import torch
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import warnings

# Add Utils to path
sys.path.insert(1, '../Utils/')      # In case we run from Experiments/Evaluation
import cfg

warnings.filterwarnings("ignore")


def bootstrap_mean_se(data, threshold, n_bootstrap=1000, random_state=None):
    """
    Compute bootstrap estimate of the mean and its standard error for values below a threshold.

    Parameters:
    - data: 1D array-like of values.
    - threshold: numeric threshold; only values < threshold are considered.
    - n_bootstrap: number of bootstrap samples.
    - random_state: seed for reproducibility.

    Returns:
    - mean_est: bootstrap estimate of the mean.
    - se_est: bootstrap estimate of the standard error of the mean.
    - lower: lower bound of 95% confidence interval.
    - upper: upper bound of 95% confidence interval.
    """
    # Prepare RNG
    rng = np.random.default_rng(random_state)
    
    # Generate bootstrap samples
    means = np.empty(n_bootstrap)
    n_data = len(data)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n_data, replace=True)
        collapsed = np.where(sample < threshold)[0]
        means[i] = len(collapsed) / len(sample)
    
    # Compute estimates
    mean_est = means.mean()
    se_est = means.std(ddof=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return mean_est, se_est, lower, upper


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute fraction collapsed (memorization metric) for Flow Matching models."
    )
    
    # Model configuration arguments
    parser.add_argument("-n", "--num", help="Number of training data", type=int, required=True)
    parser.add_argument("-i", "--index", help="Index for the dataset (0 or 1)", type=int, required=True)
    parser.add_argument("-s", "--img_size", help="Size of the images to use", type=int, required=True)
    parser.add_argument("-LR", "--learning_rate", help="Learning rate for optimization", type=float, required=True)
    parser.add_argument("-O", "--optim", help="Optimisation type (SGD_Momentum or Adam)", type=str, required=True)
    parser.add_argument("-W", "--nbase", help="Number of base filters", type=int, required=True)
    parser.add_argument("-B", "--batch_size", help="Batch size used to train the model", type=int, required=True)
    parser.add_argument("-D", "--dataset", help="Dataset used to train the model", type=str, required=True)
    
    # Analysis parameters
    parser.add_argument("-Ns", "--Nsamples", help="Number of sample batches to analyze", type=int, default=100)
    parser.add_argument("--batch_sample_size", help="Size of each sample batch", type=int, default=100)
    parser.add_argument("--gap_threshold", help="Gap ratio threshold for collapsed samples", type=float, default=1/3)
    parser.add_argument("--device", help="Device to use (cuda:0, cpu)", type=str, default='cuda:0')
    parser.add_argument('--solver', type=str, default='euler',
                        choices=['euler', 'heun', 'rk45'],
                        help='ODE solver used for flow matching (default: euler)')
    parser.add_argument('--n_steps', type=int, default=100,
                        help='Number of integration steps used in sampling (default: 100)')
    parser.add_argument('--kts', action='store_true',
                        help='Use KTS samples')
    parser.add_argument('--alpha_0', type=float, default=0.0,
                        help='KTS alpha_0 parameter')
    parser.add_argument('--beta_0', type=float, default=0.0,
                        help='KTS beta_0 parameter')
    parser.add_argument('--tau_split', type=float, default=0.6,
                        help='KTS tau_split parameter (default: 0.6)')
    parser.add_argument('--early_schedule', type=str, default='linear',
                        choices=['linear', 'constant', 'exponential'],
                        help='KTS early phase schedule form (default: linear)')
    parser.add_argument('--late_schedule', type=str, default='exponential',
                        choices=['exponential', 'linear', 'constant'],
                        help='KTS late phase schedule form (default: exponential)')
    parser.add_argument('--time_schedule', type=str, default='uniform',
                        choices=['uniform', 'cosine'],
                        help='Time schedule used during sampling (default: uniform)')
    parser.add_argument('--checkpoint', type=int, default=None,
                        help='Only evaluate this specific checkpoint ID (default: all checkpoints)')

    return parser.parse_args()

def compute_fraction_mem(training_times, train_images, type_model, config, file_fc,
                             nsamples, sample_size, gap_threshold, solver, n_steps, kts_suffix=''):
    """Compute fraction collapsed for all training times."""
    N = np.prod(config.IMG_SHAPE)
    X = train_images.reshape(-1, N).float().to(config.DEVICE)

    pbar = tqdm(training_times)
    for tau in pbar:
        # Load generated images and compute k-nearest neighbors
        k = min(2, len(train_images))

        # Load first file to determine actual batch size
        path_save = config.path_save + type_model + '/Samples-' + solver + '-step-' + str(n_steps) + kts_suffix + '/' + '{:d}/'.format(tau)
        path = path_save + 'generated'
        file_a = path + '/samples_a_0'
        images_a = torch.load(file_a)
        actual_batch_size = len(images_a)

        distances_tensor_all = torch.zeros(nsamples * actual_batch_size, k)
        knn_tensor_all = torch.zeros(nsamples * actual_batch_size, k)

        for i in range(nsamples):
            path_save = config.path_save + type_model + '/Samples-' + solver + '-step-' + str(n_steps) + kts_suffix + '/' + '{:d}/'.format(tau)
            path = path_save + 'generated'
            file_a = path + '/samples_a_{:d}'.format(i)
            
            try:
                images_a = torch.load(file_a)
            except FileNotFoundError:
                print(f"Warning: File not found: {file_a}")
                continue
            
            i1, i2 = i * actual_batch_size, (i + 1) * actual_batch_size
            
            # Compute distances to training set
            s = images_a.reshape(-1, 1, N).to(config.DEVICE)
            dist = torch.norm(s - X, dim=2, p=2)
            knn = dist.topk(k, dim=1, largest=False)
            
            distances_tensor_all[i1:i2, :] = knn[0].cpu()
            knn_tensor_all[i1:i2, :] = knn[1].cpu()
        
        # Compute gap ratios
        gap_ratio = distances_tensor_all[:, 0] / distances_tensor_all[:, 1]
        
        # Compute fraction collapsed with bootstrap confidence intervals
        collapsed_samples = np.where(gap_ratio < gap_threshold)[0]
        fraction_mem = len(collapsed_samples) / len(gap_ratio)
        
        if len(collapsed_samples) > 0:
            fraction_mem, std_frac, lower, upper = bootstrap_mean_se(
                gap_ratio.numpy(), gap_threshold
            )
        else:
            std_frac = 0.0
            lower = 0.0
            upper = 0.0

        pbar.set_description(f'Fmem = {fraction_mem*100:.2f}% ± {std_frac*100:.2f}')

        # Write results to file
        with open(file_fc, "a") as myfile:
            myfile.write(f"\n{tau:d}\t{fraction_mem*100:.3f}\t{std_frac*100:.5f}\t"
                        f"{lower*100:.5f}\t{upper*100:.5f}")


def main():
    """Main function to compute fraction collapsed."""
    # Parse arguments
    args = parse_arguments()
    print("Arguments:", args)
    
    # Load configuration
    config = cfg.load_config(args.dataset)
    config.IMG_SHAPE = (1, args.img_size, args.img_size)
    config.n_images = args.num
    config.BATCH_SIZE = min(args.batch_size, config.n_images)
    config.OPTIM = args.optim
    config.LR = args.learning_rate
    config.DEVICE = args.device

    # Model directory uses the '_flow' suffix written by the training scripts.
    type_model = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}_flow/'.format(
        config.DATASET, args.img_size, config.n_images, args.nbase,
        config.OPTIM, config.BATCH_SIZE, config.LR, args.index
    )
    
    # Create KTS suffix if KTS is enabled
    if args.kts:
        kts_suffix = '-kts-a{:.2f}-b{:.2f}-t{:.1f}'.format(args.alpha_0, args.beta_0, args.tau_split)
        if args.early_schedule != 'linear' or args.late_schedule != 'exponential':
            kts_suffix += '-es-{}-ls-{}'.format(args.early_schedule, args.late_schedule)
    else:
        kts_suffix = ''
    sched_suffix = '-sched-cosine' if args.time_schedule == 'cosine' else ''

    # Create output directory and file
    path_file = config.path_save + type_model + 'Memorization/'
    file_fc = path_file + f'fraction_memorized_{args.solver}{args.n_steps}{kts_suffix}{sched_suffix}.txt'
    if os.path.exists(file_fc):     # Remove existing file
        os.remove(file_fc)
    os.makedirs(path_file, exist_ok=True)

    # Define training times to analyze
    training_times = cfg.get_training_times()
    if args.checkpoint is not None:
        training_times = [args.checkpoint]

    # Output sample folder name
    samples_folder = 'Samples-' + args.solver + '-step-' + str(args.n_steps) + kts_suffix + sched_suffix + '/'
    print(f"\nReading samples from folder: {samples_folder}")
    print(f"Full sample path: {config.path_save + type_model + samples_folder}")
    if args.kts:
        print(f"KTS parameters: alpha_0={args.alpha_0}, beta_0={args.beta_0}, tau_split={args.tau_split}")

    print(f"\nComputing memorization fraction for {len(training_times)} checkpoints...")
    print(f"Model: {type_model}")
    print(f"Output file: {file_fc}")

    # Load training data
    train_images, _ = cfg.load_training_data(config, args.index)
    train_images = train_images[:config.n_images, :, :, :].to(config.DEVICE)

    # Compute fraction collapsed for each checkpoint
    compute_fraction_mem(
        training_times=training_times,
        train_images=train_images,
        type_model=type_model,
        config=config,
        file_fc=file_fc,
        nsamples=args.Nsamples,
        sample_size=args.batch_sample_size,
        gap_threshold=args.gap_threshold,
        solver=args.solver,
        n_steps=args.n_steps,
        kts_suffix=kts_suffix + sched_suffix
    )

    print("Memorization fraction computation completed!")


if __name__ == "__main__":
    main()