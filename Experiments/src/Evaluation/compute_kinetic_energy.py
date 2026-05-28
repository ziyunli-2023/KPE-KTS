"""
Compute kinetic energy E = (1/2) ∫₀¹ ||v_θ(x(t), t)||² dt for flow matching models.

This script analyzes trained flow matching models to compute the kinetic energy
along ODE trajectories as a function of training steps.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm
import warnings

# Add Utils to path
sys.path.insert(1, '../Utils/')
import FlowMatching as fm
import cfg
import loader
from Unet import UNet

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute kinetic energy for flow matching models."
    )

    # Model configuration arguments
    parser.add_argument("-D", "--dataset", help="Dataset used to train the model",
                        type=str, required=True, choices=['CelebA'])
    parser.add_argument("-n", "--num", help="Number of training data", type=int, required=True)
    parser.add_argument("-i", "--index", help="Index for the dataset", type=int, default=0)
    parser.add_argument("-s", "--img_size", help="Size of the images", type=int, default=32)
    parser.add_argument("-LR", "--learning_rate", help="Learning rate", type=float, required=True)
    parser.add_argument("-O", "--optim", help="Optimizer (SGD_Momentum or Adam)", type=str, required=True)
    parser.add_argument("-W", "--nbase", help="Number of U-Net base filters", type=int, required=True)
    parser.add_argument("-B", "--batch_size", help="Batch size used to train", type=int, required=True)

    # Kinetic energy computation parameters
    parser.add_argument("--n_samples", help="Number of trajectories to sample for averaging",
                        type=int, default=100)
    parser.add_argument("--n_integration_steps", help="Number of steps for ODE integration",
                        type=int, default=100)
    parser.add_argument("--device", help="Device to use", type=str, default='cuda:0')
    parser.add_argument("--solver", type=str, default='heun',
                        choices=['euler', 'heun'],
                        help='ODE solver: euler or heun (default: heun)')

    # KTS parameters
    parser.add_argument("--alpha_0", help="KTS launch intensity", type=float, default=0.0)
    parser.add_argument("--beta_0", help="KTS soft-landing damping", type=float, default=0.0)
    parser.add_argument("--tau_split", help="KTS phase transition point", type=float, default=0.6)

    return parser.parse_args()


def load_model(model_path, config, args):
    """Load model from checkpoint."""
    model = UNet(
        input_channels=config.IMG_SHAPE[0],
        output_channels=config.IMG_SHAPE[0],
        base_channels=args.nbase,
        base_channels_multiples=(1, 2, 4),
        apply_attention=(False, True, True),
        dropout_rate=0.1,
    ).to(config.DEVICE)

    # Load checkpoint
    if os.path.exists(model_path):
        # Use loader.load_model to handle DataParallel models
        model = loader.load_model(model, model_path, verbose=False)
        model.eval()
        return model
    else:
        return None


@torch.no_grad()
def compute_kinetic_energy_single_trajectory(model, x_init, config, flow_config,
                                             n_steps=100, solver='euler',
                                             alpha_0=0.0, beta_0=0.0, tau_split=0.6):
    """
    Compute kinetic energy E = (1/2) ∫₀¹ ||v_θ(x(t), t)||² dt for a single trajectory.

    With KTS enabled (alpha_0 or beta_0 > 0), computes E = (1/2) ∫₀¹ η²(t)||v_θ(x(t), t)||² dt
    where η(t) is the KTS scaling factor.

    Parameters:
    -----------
    model : nn.Module
        Trained flow matching model
    x_init : torch.Tensor
        Initial noise sample, shape [C, H, W]
    config : TrainingConfig
        Training configuration
    flow_config : FlowMatchingConfig
        Flow matching configuration
    n_steps : int
        Number of integration steps
    solver : str
        ODE solver: 'euler' or 'heun'
    alpha_0 : float
        KTS launch intensity (default: 0.0)
    beta_0 : float
        KTS soft-landing damping (default: 0.0)
    tau_split : float
        KTS phase transition point (default: 0.6)

    Returns:
    --------
    total_ke : float
        Total kinetic energy over full trajectory
    early_ke : float
        Kinetic energy in early phase [0, tau_split]
    late_ke : float
        Kinetic energy in late phase [tau_split, 1]
    """
    x = x_init.clone().unsqueeze(0)  # Add batch dimension
    model.eval()

    # Time discretization
    dt_discrete = (config.TIMESTEPS - 1) / n_steps
    dt_normalized = 1.0 / n_steps

    # Arrays to store velocity norms for early and late phases
    early_velocity_norms_squared = []
    late_velocity_norms_squared = []

    # KTS exponential damping rate
    k = 3.0

    # Determine split index
    split_idx = int(tau_split * n_steps)

    if solver == 'euler':
        for i in range(n_steps):
            t_discrete = int(i * dt_discrete)
            t_tensor = torch.ones(1, dtype=torch.long, device=config.DEVICE) * t_discrete

            # Current normalized time in [0, 1]
            t_normalized = i / n_steps

            # Compute KTS gain η(t)
            if t_normalized < tau_split:
                # Launch phase: linear decay from α₀ to 0
                alpha_t = alpha_0 * (1.0 - t_normalized / tau_split)
                eta = 1.0 + alpha_t
            else:
                # Soft-landing phase: exponential growth of damping
                t_relative = (t_normalized - tau_split) / (1.0 - tau_split)
                beta_t = beta_0 * (np.exp(k * t_relative) - 1.0)
                eta = 1.0 - beta_t

            # Compute velocity at current position
            v_t = model(x, t_tensor)

            # Store ||η(t) * v_t||² (squared L2 norm with KTS scaling)
            velocity_norm_sq = (eta * eta) * (v_t ** 2).sum().item()

            # Assign to early or late phase
            if i < split_idx:
                early_velocity_norms_squared.append(velocity_norm_sq)
            else:
                late_velocity_norms_squared.append(velocity_norm_sq)

            # Euler step with KTS scaling
            x = x + eta * dt_normalized * v_t

    elif solver == 'heun':
        for i in range(n_steps):
            t_discrete = int(i * dt_discrete)
            t_tensor = torch.ones(1, dtype=torch.long, device=config.DEVICE) * t_discrete

            # Current normalized time in [0, 1]
            t_normalized = i / n_steps

            # Compute KTS gain η(t) at current time
            if t_normalized < tau_split:
                alpha_t = alpha_0 * (1.0 - t_normalized / tau_split)
                eta_t = 1.0 + alpha_t
            else:
                t_relative = (t_normalized - tau_split) / (1.0 - tau_split)
                beta_t = beta_0 * (np.exp(k * t_relative) - 1.0)
                eta_t = 1.0 - beta_t

            # Predictor: evaluate velocity at current point
            v_t = model(x, t_tensor)

            # Store ||η(t) * v_t||² at current point
            velocity_norm_sq = (eta_t * eta_t) * (v_t ** 2).sum().item()

            # Assign to early or late phase
            if i < split_idx:
                early_velocity_norms_squared.append(velocity_norm_sq)
            else:
                late_velocity_norms_squared.append(velocity_norm_sq)

            # Predictor step
            x_temp = x + eta_t * dt_normalized * v_t

            # Compute η(t) at next time for corrector step
            t_next_normalized = (i + 1) / n_steps
            if t_next_normalized < tau_split:
                alpha_t_next = alpha_0 * (1.0 - t_next_normalized / tau_split)
                eta_t_next = 1.0 + alpha_t_next
            else:
                t_relative_next = (t_next_normalized - tau_split) / (1.0 - tau_split)
                beta_t_next = beta_0 * (np.exp(k * t_relative_next) - 1.0)
                eta_t_next = 1.0 - beta_t_next

            # Corrector: evaluate velocity at predicted position
            t_next_discrete = min(int((i + 1) * dt_discrete), config.TIMESTEPS - 1)
            t_next_tensor = torch.ones(1, dtype=torch.long, device=config.DEVICE) * t_next_discrete
            v_t_next = model(x_temp, t_next_tensor)

            # Average eta for corrector step
            eta_avg = 0.5 * (eta_t + eta_t_next)

            # Corrector step
            x = x + eta_avg * dt_normalized * (v_t + v_t_next) * 0.5

    # Numerical integration using trapezoidal rule
    # E = (1/2) * ∫ ||v||² dt ≈ (1/2) * dt * Σ ||v_i||²
    early_ke = 0.5 * dt_normalized * sum(early_velocity_norms_squared)
    late_ke = 0.5 * dt_normalized * sum(late_velocity_norms_squared)
    total_ke = early_ke + late_ke

    return total_ke, early_ke, late_ke


def compute_kinetic_energy_for_checkpoint(model, config, flow_config, args,
                                         alpha_0=0.0, beta_0=0.0, tau_split=0.6):
    """
    Compute average kinetic energy over multiple trajectories for a single checkpoint.

    Parameters:
    -----------
    model : nn.Module
        Trained model
    config : TrainingConfig
        Configuration
    flow_config : FlowMatchingConfig
        Flow matching config
    args : argparse.Namespace
        Command line arguments
    alpha_0 : float
        KTS launch intensity (default: 0.0)
    beta_0 : float
        KTS soft-landing damping (default: 0.0)
    tau_split : float
        KTS phase transition point (default: 0.6)

    Returns:
    --------
    mean_ke : float
        Mean total kinetic energy
    std_ke : float
        Standard deviation of total kinetic energy
    mean_early_ke : float
        Mean early phase kinetic energy
    std_early_ke : float
        Standard deviation of early phase kinetic energy
    mean_late_ke : float
        Mean late phase kinetic energy
    std_late_ke : float
        Standard deviation of late phase kinetic energy
    """
    total_kinetic_energies = []
    early_kinetic_energies = []
    late_kinetic_energies = []

    for _ in range(args.n_samples):
        # Generate initial noise [C, H, W]
        x_init = torch.randn(config.IMG_SHAPE[0], config.IMG_SHAPE[1],
                             config.IMG_SHAPE[2]).to(config.DEVICE)

        # Compute kinetic energy for this trajectory
        total_ke, early_ke, late_ke = compute_kinetic_energy_single_trajectory(
            model=model,
            x_init=x_init,
            config=config,
            flow_config=flow_config,
            n_steps=args.n_integration_steps,
            solver=args.solver,
            alpha_0=alpha_0,
            beta_0=beta_0,
            tau_split=tau_split
        )
        total_kinetic_energies.append(total_ke)
        early_kinetic_energies.append(early_ke)
        late_kinetic_energies.append(late_ke)

    # Compute statistics
    mean_ke = np.mean(total_kinetic_energies)
    std_ke = np.std(total_kinetic_energies)
    mean_early_ke = np.mean(early_kinetic_energies)
    std_early_ke = np.std(early_kinetic_energies)
    mean_late_ke = np.mean(late_kinetic_energies)
    std_late_ke = np.std(late_kinetic_energies)

    return mean_ke, std_ke, mean_early_ke, std_early_ke, mean_late_ke, std_late_ke


def compute_kinetic_energy_all_checkpoints(training_times, model_dir, config,
                                          flow_config, args,
                                          alpha_0=0.0, beta_0=0.0, tau_split=0.6):
    """
    Compute kinetic energy for all training checkpoints.

    Parameters:
    -----------
    training_times : np.ndarray
        Array of training step numbers where checkpoints are saved
    model_dir : str
        Directory containing model checkpoints
    config : TrainingConfig
        Configuration
    flow_config : FlowMatchingConfig
        Flow matching configuration
    args : argparse.Namespace
        Command line arguments
    alpha_0 : float
        KTS launch intensity (default: 0.0)
    beta_0 : float
        KTS soft-landing damping (default: 0.0)
    tau_split : float
        KTS phase transition point (default: 0.6)

    Returns:
    --------
    results : dict
        Dictionary with 'training_steps', 'mean_ke', 'std_ke',
        'mean_early_ke', 'std_early_ke', 'mean_late_ke', 'std_late_ke'
    """
    results = {
        'training_steps': [],
        'mean_ke': [],
        'std_ke': [],
        'mean_early_ke': [],
        'std_early_ke': [],
        'mean_late_ke': [],
        'std_late_ke': []
    }

    print(f"Computing kinetic energy for {len(training_times)} checkpoints...")
    print(f"Model directory: {model_dir}")
    print(f"Number of samples per checkpoint: {args.n_samples}")
    print(f"Integration steps: {args.n_integration_steps}")
    print(f"Solver: {args.solver}")
    print(f"KTS parameters: alpha_0={alpha_0}, beta_0={beta_0}, tau_split={tau_split}")

    pbar = tqdm(training_times)
    for tau in pbar:
        model_path = config.path_save + model_dir + f'Models/Model_{tau}'

        # Load model
        model = load_model(model_path, config, args)

        if model is None:
            print(f"\nWarning: Model checkpoint at step {tau} not found, skipping...")
            continue

        # Compute kinetic energy
        mean_ke, std_ke, mean_early_ke, std_early_ke, mean_late_ke, std_late_ke = \
            compute_kinetic_energy_for_checkpoint(
                model=model,
                config=config,
                flow_config=flow_config,
                args=args,
                alpha_0=alpha_0,
                beta_0=beta_0,
                tau_split=tau_split
            )

        # Store results
        results['training_steps'].append(tau)
        results['mean_ke'].append(mean_ke)
        results['std_ke'].append(std_ke)
        results['mean_early_ke'].append(mean_early_ke)
        results['std_early_ke'].append(std_early_ke)
        results['mean_late_ke'].append(mean_late_ke)
        results['std_late_ke'].append(std_late_ke)

        pbar.set_description(f'KE = {mean_ke:.6f} ± {std_ke:.6f}')

    # Convert to numpy arrays
    results['training_steps'] = np.array(results['training_steps'])
    results['mean_ke'] = np.array(results['mean_ke'])
    results['std_ke'] = np.array(results['std_ke'])
    results['mean_early_ke'] = np.array(results['mean_early_ke'])
    results['std_early_ke'] = np.array(results['std_early_ke'])
    results['mean_late_ke'] = np.array(results['mean_late_ke'])
    results['std_late_ke'] = np.array(results['std_late_ke'])

    return results


def save_and_plot_results(results, model_dir, config, args):
    """Save results and create plots."""
    # Create output directory with KTS parameters
    output_dir = config.path_save + model_dir + f'KineticEnergy/kinetic_energy-a{args.alpha_0}-b{args.beta_0}-t{args.tau_split}/'
    os.makedirs(output_dir, exist_ok=True)

    # Save raw data
    output_file = output_dir + 'kinetic_energy.npz'
    np.savez(output_file,
             training_steps=results['training_steps'],
             mean_ke=results['mean_ke'],
             std_ke=results['std_ke'],
             mean_early_ke=results['mean_early_ke'],
             std_early_ke=results['std_early_ke'],
             mean_late_ke=results['mean_late_ke'],
             std_late_ke=results['std_late_ke'])
    print(f"\nResults saved to: {output_file}")

    # Save text file
    txt_file = output_dir + 'kinetic_energy.txt'
    with open(txt_file, 'w') as f:
        f.write("# Training_Step\tMean_KE\tStd_KE\tMean_Early_KE\tStd_Early_KE\tMean_Late_KE\tStd_Late_KE\n")
        for i in range(len(results['training_steps'])):
            f.write(f"{results['training_steps'][i]}\t"
                    f"{results['mean_ke'][i]:.6f}\t{results['std_ke'][i]:.6f}\t"
                    f"{results['mean_early_ke'][i]:.6f}\t{results['std_early_ke'][i]:.6f}\t"
                    f"{results['mean_late_ke'][i]:.6f}\t{results['std_late_ke'][i]:.6f}\n")
    print(f"Text results saved to: {txt_file}")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot total kinetic energy with error bars
    ax.errorbar(results['training_steps'], results['mean_ke'],
                yerr=results['std_ke'], fmt='o-', capsize=3,
                label='Total KE', linewidth=2, markersize=4, color='blue')

    # Plot early phase kinetic energy
    ax.errorbar(results['training_steps'], results['mean_early_ke'],
                yerr=results['std_early_ke'], fmt='s--', capsize=3,
                label=f'Early KE (t < {args.tau_split})', linewidth=1.5, markersize=3, color='green')

    # Plot late phase kinetic energy
    ax.errorbar(results['training_steps'], results['mean_late_ke'],
                yerr=results['std_late_ke'], fmt='^--', capsize=3,
                label=f'Late KE (t >= {args.tau_split})', linewidth=1.5, markersize=3, color='red')

    ax.set_xlabel('Training Steps', fontsize=14)
    ax.set_ylabel(r'Kinetic Energy $E = \frac{1}{2}\int \|\eta(t) v_\theta(x(t), t)\|^2 dt$',
                  fontsize=14)
    ax.set_title(f'Kinetic Energy vs Training Steps (α₀={args.alpha_0}, β₀={args.beta_0})', fontsize=16)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    # Save figure
    fig_file = output_dir + 'kinetic_energy.pdf'
    fig.savefig(fig_file, bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {fig_file}")

    # Also save as PNG for easy viewing
    png_file = output_dir + 'kinetic_energy.png'
    fig.savefig(png_file, bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {png_file}")

    plt.close(fig)


def main():
    """Main function to compute kinetic energy."""
    # Parse arguments
    args = parse_arguments()
    print("Arguments:", args)

    # Load configuration
    config = cfg.load_config(args.dataset)

    # Update configuration based on arguments
    config.IMG_SHAPE = (1, args.img_size, args.img_size)

    config.n_images = args.num
    config.BATCH_SIZE = min(args.batch_size, config.n_images)
    config.OPTIM = args.optim
    config.LR = args.learning_rate
    config.DEVICE = args.device

    # Initialize flow matching config
    flow_config = fm.FlowMatchingConfig(
        n_steps=config.TIMESTEPS,
        img_shape=config.IMG_SHAPE,
        device=config.DEVICE
    )

    # Build model directory path (with _flow suffix)
    model_dir = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}_flow/'.format(
        config.DATASET, args.img_size, config.n_images, args.nbase,
        config.OPTIM, config.BATCH_SIZE, config.LR, args.index
    )

    # Get training checkpoint times
    training_times = cfg.get_training_times()

    # Compute kinetic energy for all checkpoints
    results = compute_kinetic_energy_all_checkpoints(
        training_times=training_times,
        model_dir=model_dir,
        config=config,
        flow_config=flow_config,
        args=args,
        alpha_0=args.alpha_0,
        beta_0=args.beta_0,
        tau_split=args.tau_split
    )

    # Save and plot results
    save_and_plot_results(results, model_dir, config, args)

    print("\nKinetic energy computation completed!")


if __name__ == "__main__":
    main()
