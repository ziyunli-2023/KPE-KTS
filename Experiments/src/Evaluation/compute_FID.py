"""
Compute FID (Fréchet Inception Distance) for Flow Matching models.

Generated samples are detransformed, written to a temporary directory, and
scored against precomputed reference statistics (`Saves/FID_ref/stats{i}.npz`)
via `python -m pytorch_fid`.
"""

import torch
import os
import sys
import argparse
from tqdm import tqdm
import warnings
import torchvision
import subprocess
import shutil

# Add Utils to path
sys.path.insert(1, '../Utils/')      # In case we run from Experiments/Evaluation
import cfg

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute FID (Fréchet Inception Distance) for Flow Matching models."
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
    parser.add_argument("-istat", "--id_stat", help="Index of the reference statistics (1 to 5)", type=int, required=True)
    
    # Analysis parameters
    parser.add_argument("--N1", help="Starting batch index", type=int, default=0)
    parser.add_argument("--N2", help="Ending batch index", type=int, default=100)
    parser.add_argument("--batch_size_samples", help="Size of each sample batch", type=int, default=100)
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


def detransform_images(images, config):
    """Detransform images from normalized to original scale."""
    t = images.clone()
    mean = torch.tensor(config.mean, dtype=images.dtype, 
                        device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(config.std[0], dtype=images.dtype,
                       device=images.device).view(1, -1, 1, 1)
    return t * std + mean


def compute_fid_for_checkpoint(tau, type_model, config, path_stats_testset,
                             N1, N2, batch_size_samples, file_FID, solver, n_steps, kts_suffix=''):
    """Compute FID for a specific training checkpoint."""
    # Save directory for temporary images
    file_img_gen = config.path_save + type_model + 'FID/{:d}/'.format(tau)
    os.makedirs(file_img_gen, exist_ok=True)

    try:
        # Load generated images for the current training time
        for i in range(N1, N2):
            path_save = config.path_save + type_model + '/Samples-' + solver + '-step-' + str(n_steps) + kts_suffix + '/' + '{:d}/'.format(tau)
            path = path_save + 'generated'
            file_a = path + '/samples_a_{:d}'.format(i)
            
            # Load generated samples
            images_a = torch.load(file_a)
            
            # Detransform data to original scale
            t = detransform_images(images_a, config)

            # Convert grayscale (1-channel) to RGB (3-channel) for InceptionV3
            # This ensures consistency with FID reference statistics
            if t.shape[1] == 1:
                t = t.repeat(1, 3, 1, 1)

            # Save images as PNG files
            for (index_im, x) in enumerate(t):
                torchvision.utils.save_image(x, file_img_gen + '{:d}.png'.format(index_im + i*batch_size_samples))
        
        # Compute FID using pytorch_fid
        args = '{:s} {:s} --device cuda:{:d}'.format(path_stats_testset,
                                                     file_img_gen,
                                                     int(config.DEVICE[-1]))
        cmd = 'python -m pytorch_fid {:s}'.format(args)
        p = subprocess.check_output(cmd, shell=True, text=True)
        fid = float(p.split(' ')[2][0:-2])
        
        # Save result
        with open(file_FID, "a") as myfile:
            myfile.write("\n{:d}\t{:.3f}".format(tau, fid))
        
    except Exception as e:
        print(f"Error computing FID for checkpoint {tau}: {e}")
        fid = -1.000
        with open(file_FID, "a") as myfile:
            myfile.write("\n{:d}\t{:.3f}".format(tau, fid))
        print('Skipping...')
    
    finally:
        # Clean up temporary images directory
        if os.path.exists(file_img_gen):
            shutil.rmtree(file_img_gen)
    
    return fid


def compute_fid_all_checkpoints(training_times, type_model, config, args, solver, n_steps, kts_suffix=''):
    """Compute FID for all training checkpoints."""
    # Setup paths and files
    path_stats_testset = config.path_save + 'FID_ref/stats{:d}.npz'.format(args.id_stat)
    path_file = config.path_save + type_model + 'FID/'
    file_FID = path_file + f'FID_{args.id_stat}_{solver}{n_steps}{kts_suffix}.txt'
    if os.path.exists(file_FID):     # Remove existing file
        os.remove(file_FID)
    os.makedirs(path_file, exist_ok=True)

    print(f"Computing FID for {len(training_times)} checkpoints...")
    print(f"Model: {type_model}")
    print(f"Reference statistics: {path_stats_testset}")
    print(f"Output file: {file_FID}")

    pbar = tqdm(training_times)
    for tau in pbar:
        fid = compute_fid_for_checkpoint(
            tau=tau,
            type_model=type_model,
            config=config,
            path_stats_testset=path_stats_testset,
            N1=args.N1,
            N2=args.N2,
            batch_size_samples=args.batch_size_samples,
            file_FID=file_FID,
            solver=solver,
            n_steps=n_steps,
            kts_suffix=kts_suffix
        )
        pbar.set_description(f'FID = {fid:.3f}')


def main():
    """Main function to compute FID scores."""
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

    # Load training data (for consistency, though not used in FID computation)
    train_images, _ = cfg.load_training_data(config, args.index)
    train_images = train_images[:config.n_images, :, :, :].to(config.DEVICE)

    # Compute FID for all checkpoints
    compute_fid_all_checkpoints(
        training_times=training_times,
        type_model=type_model,
        config=config,
        args=args,
        solver=args.solver,
        n_steps=args.n_steps,
        kts_suffix=kts_suffix + sched_suffix
    )
    
    print("FID computation completed!")


if __name__ == "__main__":
    main()