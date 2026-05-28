"""
Generate samples from a trained Flow Matching model (optionally with KTS).

The Kinetic Trajectory Shaping (KTS) release supports Flow Matching only.
Samples are written under each model's `Samples-<solver>-step-<N>[...]/`
directory at every training checkpoint produced by `cfg.get_training_times()`.
"""

import os
import sys
import argparse

import torch

sys.path.insert(1, '../Utils/')      # In case we run from Experiments/src/Generation
import Unet
import cfg
import loader
import FlowMatching


# ====================================================================
# Argument parsing
# ====================================================================
parser = argparse.ArgumentParser("Generation of samples from trained Flow Matching models.")

parser.add_argument("-n", "--num", help="Number of training data", type=int)
parser.add_argument("-i", "--index", help="Index for the dataset (0 or 1)", type=int)
parser.add_argument("-s", "--img_size", help="Size of the images used to train", type=int)
parser.add_argument("-LR", "--learning_rate", help="Learning rate for optimization", type=float)
parser.add_argument("-O", "--optim", help="Optimisation type (SGD_Momentum or Adam)", type=str)
parser.add_argument("-W", "--nbase", help="Number of base filters", type=str)
parser.add_argument("-t", "--time", help="Flow timestep", type=int)
parser.add_argument("-B", "--batch_size", type=int,
                    help="Batch size used to train the model")
parser.add_argument('-D', '--dataset', type=str,
                    help='Dataset used to train the model.')
parser.add_argument('-Ns', '--Nsamples', type=int,
                    help='Number of samples to generate (should be multiple of 100).')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Device used to load and apply the model.')
parser.add_argument('--solver', type=str, default='euler',
                    choices=['euler', 'heun', 'rk45'],
                    help='ODE solver for flow matching (default: euler)')
parser.add_argument('--n_steps', type=int, default=100,
                    help='Number of integration steps for sampling (default: 100)')
parser.add_argument('--kts', action='store_true',
                    help='Enable Kinetic Trajectory Shaping (KTS)')
parser.add_argument('--alpha_0', type=float, default=0.0,
                    help='KTS launch intensity (default: 0.0)')
parser.add_argument('--beta_0', type=float, default=0.0,
                    help='KTS soft-landing damping (default: 0.0)')
parser.add_argument('--tau_split', type=float, default=0.6,
                    help='KTS phase transition point (default: 0.6)')
parser.add_argument('--early_schedule', type=str, default='linear',
                    choices=['linear', 'constant', 'exponential'],
                    help='KTS early phase schedule form (default: linear)')
parser.add_argument('--late_schedule', type=str, default='exponential',
                    choices=['exponential', 'linear', 'constant'],
                    help='KTS late phase schedule form (default: exponential)')
parser.add_argument('--time_schedule', type=str, default='uniform',
                    choices=['uniform', 'cosine'],
                    help='Time discretization schedule for ODE sampling (default: uniform)')
parser.add_argument('--checkpoint', type=int, default=None,
                    help='Only generate samples for this specific checkpoint ID (default: all checkpoints)')

args = parser.parse_args()
print(args)

DATASET = args.dataset
config = cfg.load_config(DATASET)   # Load base config for this dataset
n_base = int(args.nbase)
config.DEVICE = args.device
config.n_images = int(args.num)
Nsamples = int(args.Nsamples)
size = int(args.img_size)
config.OPTIM = args.optim
config.BATCH_SIZE = int(args.batch_size)
config.LR = float(args.learning_rate)
index = int(args.index)

if Nsamples % 100 != 0:
    raise TypeError('Nsamples should be a multiple of 100.')


# ====================================================================
# Flow Matching configuration and model directory
# ====================================================================
fm = FlowMatching.FlowMatchingConfig(
    n_steps=config.TIMESTEPS,
    img_shape=config.IMG_SHAPE,
    device=config.DEVICE,
)
# Model path for flow matching (uses the '_flow' suffix from training)
type_model = '{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}_flow/'.format(
    config.DATASET, size, config.n_images, n_base,
    config.OPTIM, config.BATCH_SIZE, config.LR, index,
)

model = Unet.UNet(
    input_channels=config.IMG_SHAPE[0],
    output_channels=config.IMG_SHAPE[0],
    base_channels=n_base,
    base_channels_multiples=(1, 2, 4),
    apply_attention=(False, True, True),
    dropout_rate=0.1,
)
model.to(config.DEVICE)

print('Generating {:d} samples'.format(Nsamples))


# ====================================================================
# Sampling loop over training checkpoints
# ====================================================================
batch_gen = 10000
Ns = Nsamples // batch_gen

training_times = cfg.get_training_times()
if args.checkpoint is not None:
    training_times = [args.checkpoint]


def _select_sampler(solver, use_kts):
    if use_kts:
        return {
            'euler': FlowMatching.sample_flow_euler_kts,
            'heun':  FlowMatching.sample_flow_heun_kts,
            'rk45':  FlowMatching.sample_flow_rk45_kts,
        }[solver]
    return {
        'euler': FlowMatching.sample_flow_euler,
        'heun':  FlowMatching.sample_flow_heun,
        'rk45':  FlowMatching.sample_flow_rk45,
    }[solver]


for (j, checkpoint_id) in enumerate(training_times):
    print(r'Training time = {:d} ({:d}/{:d})'.format(checkpoint_id, j, len(training_times)))

    # Load the model checkpoint
    model_suffix = '/Model_{:d}'.format(checkpoint_id)
    path_model = config.path_save + type_model + '/Models/' + model_suffix
    try:
        model = loader.load_model(model, path_model)
    except Exception:
        raise NameError('The checkpoint does not exist: {:s}'.format(path_model))

    sampler = _select_sampler(args.solver, args.kts)

    for i in range(Ns):
        # Build the output suffix (KTS + time-schedule annotations)
        if args.kts:
            kts_suffix = '-kts-a{:.2f}-b{:.2f}-t{:.1f}'.format(
                args.alpha_0, args.beta_0, args.tau_split,
            )
            if args.early_schedule != 'linear' or args.late_schedule != 'exponential':
                kts_suffix += '-es-{}-ls-{}'.format(args.early_schedule, args.late_schedule)
        else:
            kts_suffix = ''
        sched_suffix = '-sched-cosine' if args.time_schedule == 'cosine' else ''

        path_save = (
            config.path_save + type_model
            + '/Samples-' + args.solver + '-step-' + str(args.n_steps)
            + kts_suffix + sched_suffix + '/'
            + '{:d}/'.format(checkpoint_id)
        )
        os.makedirs(path_save, exist_ok=True)

        print('Sample {:d}/{:d}'.format(i, Ns))

        sampler_kwargs = dict(
            n_images=batch_gen,
            config=config,
            fm=fm,
            dim=4,
            n_steps=args.n_steps,
        )
        if args.kts:
            sampler_kwargs.update(
                alpha_0=args.alpha_0,
                beta_0=args.beta_0,
                tau_split=args.tau_split,
                early_schedule=args.early_schedule,
                late_schedule=args.late_schedule,
            )
        # The RK45 solver does not accept a time_schedule argument
        if args.solver != 'rk45':
            sampler_kwargs['time_schedule'] = args.time_schedule

        samples_gen, samples_init = sampler(model, **sampler_kwargs)

        # Save initial noise (t=0)
        path = path_save + str(config.TIMESTEPS)
        os.makedirs(path, exist_ok=True)
        torch.save(samples_init, path + '/samples_a_{:d}'.format(i))

        # Save the generated samples (t=1)
        path = path_save + 'generated'
        os.makedirs(path, exist_ok=True)
        torch.save(samples_gen, path + '/samples_a_{:d}'.format(i))

print('Done!')
