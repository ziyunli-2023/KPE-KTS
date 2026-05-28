import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
import Plot
from cfg import TrainingConfig


def _make_time_grid(n_steps, time_schedule='uniform'):
    """Return list of n_steps+1 normalized time points in [0, 1].

    'uniform' : t_i = i / n_steps  (linear spacing)
    'cosine'  : t_i = 0.5 * (1 - cos(pi * i / n_steps))
    """
    if time_schedule == 'cosine':
        return [0.5 * (1.0 - np.cos(np.pi * i / n_steps)) for i in range(n_steps + 1)]
    else:
        return [i / n_steps for i in range(n_steps + 1)]

# ====================================================================
# Flow Matching Configuration Class
# ====================================================================
class FlowMatchingConfig:
    '''
    FlowMatchingConfig: Class containing information related to the
    flow matching process (number of steps, device, etc.).
    Flow matching uses simple linear interpolation: x_t = (1-t)*x0 + t*x1.
    '''
    def __init__(self, n_steps=1000, img_shape=(3, 32, 32), device='cpu'):
        self.n_steps = n_steps
        self.img_shape = img_shape
        self.device = device
        # No beta schedules or alpha computations needed for CFM
        # Flow matching uses simple linear interpolation: x_t = (1-t)*x0 + t*x1


# ====================================================================
# Flow Matching Functions
# ====================================================================
def forward_flow(x0, x1, timesteps, config):
    '''
    Apply forward flow process: x_t = (1-t)*x0 + t*x1

    Parameters:
    -----------
    x0 : torch.Tensor
        Source samples (noise), shape [B, ...]
    x1 : torch.Tensor
        Target samples (data), shape [B, ...]
    timesteps : torch.Tensor
        Time values in discrete units [0, TIMESTEPS-1], shape [B]
    config : TrainingConfig
        Configuration object with TIMESTEPS attribute

    Returns:
    --------
    x_t : torch.Tensor
        Interpolated samples at time t
    velocity : torch.Tensor
        True velocity v_t = x1 - x0 (what model should predict)
    '''
    dim = len(x0.shape)

    # Normalize discrete timesteps to continuous time in [0, 1]
    # timesteps are in {0, 1, ..., TIMESTEPS-1}
    t_normalized = timesteps.float() / (config.TIMESTEPS - 1)

    # Reshape t for broadcasting based on tensor dimension
    if dim == 4:  # [B, C, H, W]
        t = t_normalized.reshape(-1, 1, 1, 1)
    elif dim == 3:  # [B, C, N]
        t = t_normalized.reshape(-1, 1, 1)
    elif dim == 2:  # [B, N]
        t = t_normalized.reshape(-1, 1)

    # Linear interpolation: x_t = (1-t)*x0 + t*x1
    x_t = (1 - t) * x0 + t * x1

    # Velocity field for conditional flow matching
    # v_t = dx/dt = x1 - x0 (constant velocity for linear path)
    velocity = x1 - x0

    return x_t, velocity


# ====================================================================
# Training Functions
# ====================================================================
def train_one_batch_flow(X, model, optimizer, loss_fn,
                         config=TrainingConfig(),
                         fm=FlowMatchingConfig()):
    '''
    Train one batch using Conditional Flow Matching.

    Key differences from diffusion:
    1. Sample x0 ~ N(0,1) and x1 = X (data)
    2. Sample random timesteps t ~ Uniform[0, TIMESTEPS-1]
    3. Compute x_t = (1-t)*x0 + t*x1
    4. Model predicts velocity: v_pred = model(x_t, t)
    5. Loss: MSE(v_pred, x1 - x0)

    Parameters:
    -----------
    X : torch.Tensor
        Batch of data samples
    model : nn.Module
        Neural network that predicts velocity
    optimizer : torch.optim.Optimizer
        Optimizer for model parameters
    loss_fn : callable
        Loss function (typically MSELoss)
    config : TrainingConfig
        Training configuration
    fm : FlowMatchingConfig
        Flow matching configuration

    Returns:
    --------
    loss : float
        Loss value for this batch
    x_t : torch.Tensor
        Interpolated samples (for visualization if needed)
    '''
    model.train()

    # Sample source noise x0 ~ N(0, 1)
    x0 = torch.randn_like(X)

    # Target is the data
    x1 = X

    # Sample random timesteps
    # Flow matching uses t ∈ [0, TIMESTEPS-1] (includes 0, unlike diffusion)
    if config.mode == 'normal':
        ts = torch.randint(low=0, high=config.TIMESTEPS,
                          size=(X.shape[0],), device=config.DEVICE)
    elif config.mode == 'fixed_time':
        # Fixed time training mode
        ts = torch.ones((X.shape[0],), dtype=torch.long,
                       device=config.DEVICE) * config.time_step

    # Apply forward flow to get x_t and true velocity
    x_t, velocity_true = forward_flow(x0, x1, ts, config)
    x_t = x_t.to(config.DEVICE)

    # Model predicts velocity at (x_t, t)
    velocity_pred = model(x_t.float(), ts)

    # Loss: MSE between predicted and true velocity
    loss = loss_fn(velocity_pred, velocity_true)

    # Update parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach().item(), x_t


def train_flow(model, trainloader, optimizer, config, fm, loss_fn,
               sweep=1., times_save=[], offset=0, suffix='', generate=False):
    '''
    Main training loop for flow matching.

    Parameters:
    -----------
    model : nn.Module
        Neural network model
    trainloader : torch.utils.data.DataLoader
        Data loader for training data
    optimizer : torch.optim.Optimizer
        Optimizer
    config : TrainingConfig
        Training configuration
    fm : FlowMatchingConfig
        Flow matching configuration
    loss_fn : callable
        Loss function
    sweep : float
        Unused parameter (kept for API compatibility)
    times_save : list
        List of training steps at which to save checkpoints
    offset : int
        Starting step number (for resuming training)
    suffix : str
        Directory suffix for saving (e.g., 'CelebA32_1024_32_Adam_512_0.0001_index0_flow/')
    generate : bool
        Whether to generate samples during training for visual inspection
    '''
    n_steps = offset    # Number of SGD steps
    k_steps = 100       # Number of steps before printing

    bar = trange(config.N_STEPS, leave=True, position=0)
    bar.update(offset)

    while n_steps < config.N_STEPS:
        for i, X in enumerate(trainloader):
            X = X.to(config.DEVICE)

            shallSave = n_steps in times_save
            if n_steps >= config.N_STEPS:
                shallSave = 1

            if shallSave == 1:
                # Save model checkpoint
                p = config.path_save + suffix + 'Models/' + 'Model_{:d}'.format(n_steps)
                torch.save(model.state_dict(), p)

                if generate:
                    # Sample a small batch and save it to check quality visually
                    if len(X.shape) == 4:  # For images, assumes [B, C, H, W]
                        samples, samples_init = sample_flow_heun(model, 16, config, fm, dim=4, n_steps=100)
                        fig = Plot.imshow(samples.cpu(), config.mean, config.std)
                        fig.savefig(config.path_save + suffix + 'Images/' + 'Sample_{:d}.pdf'.format(n_steps),
                                   bbox_inches='tight')
                        plt.close('all')

            loss, _ = train_one_batch_flow(X, model, optimizer, loss_fn, config, fm)
            n_steps += 1            # Update number of steps

            # Update the progress bar (every k steps)
            if n_steps % k_steps == 0:
                bar.set_description(f'loss: {loss:.5f}, n_steps: {n_steps:d}')
                bar.update(k_steps)

            # If we performed all the steps, exit
            if n_steps >= config.N_STEPS:
                break

    return


# ====================================================================
# ODE Solvers for Sampling
# ====================================================================

@torch.no_grad()
def sample_flow_euler(model, n_images=25, config=TrainingConfig(),
                     fm=FlowMatchingConfig(), dim=3, n_steps=100,
                     time_schedule='uniform'):
    '''
    Euler method for solving flow ODE: dx/dt = v_theta(x, t)

    This is the simplest first-order ODE solver. Fast but requires
    more steps for good accuracy.

    Parameters:
    -----------
    model : nn.Module
        Trained velocity prediction network
    n_images : int
        Number of samples to generate
    config : TrainingConfig
        Configuration object with IMG_SHAPE, TIMESTEPS, DEVICE
    fm : FlowMatchingConfig
        Flow matching configuration
    dim : int
        Dimensionality: 2 for [B, N], 3 for [B, C, N], 4 for [B, C, H, W]
    n_steps : int
        Number of integration steps (default: 100)

    Returns:
    --------
    x : torch.Tensor
        Generated samples at t=1
    x_init : torch.Tensor
        Initial noise samples at t=0
    '''
    # Generate initial noise x_0 ~ N(0, 1)
    if dim == 4:  # [B, C, H, W]
        x_init = torch.randn(n_images, config.IMG_SHAPE[0],
                            config.IMG_SHAPE[1], config.IMG_SHAPE[2]).to(config.DEVICE)
    elif dim == 3:  # [B, C, N]
        x_init = torch.randn(n_images, config.IMG_SHAPE[0],
                            config.IMG_SHAPE[1]).to(config.DEVICE)
    elif dim == 2:  # [B, N]
        x_init = torch.randn(n_images, config.IMG_SHAPE[1]).to(config.DEVICE)

    x = x_init.clone()
    model.eval()

    times = _make_time_grid(n_steps, time_schedule)

    for i in range(n_steps):
        t_normalized = times[i]
        dt = times[i + 1] - times[i]
        t_discrete = int(t_normalized * (config.TIMESTEPS - 1))
        t_tensor = torch.ones(n_images, dtype=torch.long,
                             device=config.DEVICE) * t_discrete

        v_t = model(x, t_tensor)
        x = x + dt * v_t

    return x, x_init


@torch.no_grad()
def sample_flow_heun(model, n_images=25, config=TrainingConfig(),
                    fm=FlowMatchingConfig(), dim=3, n_steps=100,
                    time_schedule='uniform'):
    '''
    Heun's method (2nd order Runge-Kutta) for flow ODE.

    More accurate than Euler method. This is the RECOMMENDED solver
    as it provides a good balance between quality and speed.

    Algorithm:
    1. Predictor: x_temp = x + dt * v(x, t)
    2. Corrector: x_new = x + dt/2 * (v(x, t) + v(x_temp, t+dt))

    Parameters:
    -----------
    model : nn.Module
        Trained velocity prediction network
    n_images : int
        Number of samples to generate
    config : TrainingConfig
        Configuration object
    fm : FlowMatchingConfig
        Flow matching configuration
    dim : int
        Dimensionality of tensor
    n_steps : int
        Number of integration steps (default: 100)

    Returns:
    --------
    x : torch.Tensor
        Generated samples
    x_init : torch.Tensor
        Initial noise
    '''
    # Generate initial noise
    if dim == 4:
        x_init = torch.randn(n_images, config.IMG_SHAPE[0],
                            config.IMG_SHAPE[1], config.IMG_SHAPE[2]).to(config.DEVICE)
    elif dim == 3:
        x_init = torch.randn(n_images, config.IMG_SHAPE[0],
                            config.IMG_SHAPE[1]).to(config.DEVICE)
    elif dim == 2:
        x_init = torch.randn(n_images, config.IMG_SHAPE[1]).to(config.DEVICE)

    x = x_init.clone()
    model.eval()

    times = _make_time_grid(n_steps, time_schedule)

    for i in range(n_steps):
        t_normalized = times[i]
        t_next_normalized = times[i + 1]
        dt = t_next_normalized - t_normalized
        t_discrete = int(t_normalized * (config.TIMESTEPS - 1))
        t_tensor = torch.ones(n_images, dtype=torch.long,
                             device=config.DEVICE) * t_discrete

        v_t = model(x, t_tensor)
        x_temp = x + dt * v_t

        t_next_discrete = min(int(t_next_normalized * (config.TIMESTEPS - 1)), config.TIMESTEPS - 1)
        t_next_tensor = torch.ones(n_images, dtype=torch.long,
                                   device=config.DEVICE) * t_next_discrete
        v_t_next = model(x_temp, t_next_tensor)

        x = x + dt * 0.5 * (v_t + v_t_next)

    return x, x_init


@torch.no_grad()
def sample_flow_rk45(model, n_images=25, config=TrainingConfig(),
                    fm=FlowMatchingConfig(), dim=3, n_steps=100):
    '''
    Runge-Kutta 45 (4th order with 5th order error estimation) using torchdiffeq.

    Highest accuracy solver with adaptive step size. Can use very few steps
    (as low as 10-20) for high quality results, but each step is expensive.

    Requires: pip install torchdiffeq

    Parameters:
    -----------
    model : nn.Module
        Trained velocity prediction network
    n_images : int
        Number of samples to generate
    config : TrainingConfig
        Configuration object
    fm : FlowMatchingConfig
        Flow matching configuration
    dim : int
        Dimensionality of tensor
    n_steps : int
        Number of evaluation points (default: 100)
        Note: Actual integration steps are adaptive

    Returns:
    --------
    x : torch.Tensor
        Generated samples at t=1
    x_init : torch.Tensor
        Initial noise at t=0
    '''
    try:
        from torchdiffeq import odeint
    except ImportError:
        raise ImportError(
            "torchdiffeq not installed. Install it with: pip install torchdiffeq\n"
            "Or use sample_flow_euler() or sample_flow_heun() instead."
        )

    # Generate initial noise
    if dim == 4:
        x_init = torch.randn(n_images, config.IMG_SHAPE[0],
                            config.IMG_SHAPE[1], config.IMG_SHAPE[2]).to(config.DEVICE)
    elif dim == 3:
        x_init = torch.randn(n_images, config.IMG_SHAPE[0],
                            config.IMG_SHAPE[1]).to(config.DEVICE)
    elif dim == 2:
        x_init = torch.randn(n_images, config.IMG_SHAPE[1]).to(config.DEVICE)

    model.eval()

    # Define ODE function: dx/dt = v_theta(x, t)
    def ode_func(t, x):
        '''
        ODE right-hand side: dx/dt = v(x, t)

        Parameters:
        -----------
        t : torch.Tensor
            Scalar time in [0, 1]
        x : torch.Tensor
            Current state

        Returns:
        --------
        v : torch.Tensor
            Velocity at (x, t)
        '''
        # Convert continuous time t ∈ [0,1] to discrete timestep
        t_discrete = int(t.item() * (config.TIMESTEPS - 1))
        t_discrete = max(0, min(t_discrete, config.TIMESTEPS - 1))  # Clamp to valid range

        t_tensor = torch.ones(x.shape[0], dtype=torch.long,
                             device=config.DEVICE) * t_discrete
        return model(x, t_tensor)

    # Solve ODE from t=0 to t=1 using adaptive RK45
    t_span = torch.linspace(0, 1, n_steps).to(config.DEVICE)
    solution = odeint(ode_func, x_init, t_span, method='dopri5', rtol=1e-5, atol=1e-5)

    # Return final state (at t=1) and initial state (at t=0)
    return solution[-1], x_init


# ====================================================================
# KTS (Kinetic Trajectory Shaping) Samplers
# ====================================================================

def _kts_eta(t, alpha_0, beta_0, tau_split,
             early_schedule='linear', late_schedule='exponential', k=3.0):
    """Compute KTS gain η(t) for a single time point t ∈ [0, 1].

    Early phase (t < tau_split):
        linear      : α(t) = α₀ · (1 - t/τ)
        constant    : α(t) = α₀
        exponential : α(t) = α₀ · [exp(k·(1-t/τ)) - 1] / [exp(k) - 1]
        η = 1 + α(t)

    Late phase (t ≥ tau_split):
        exponential : β(t) = β₀ · [exp(k·t_rel) - 1],  t_rel=(t-τ)/(1-τ)
        linear      : β(t) = β₀ · t_rel
        constant    : β(t) = β₀
        η = 1 - β(t)
    """
    if t < tau_split:
        if early_schedule == 'constant':
            alpha_t = alpha_0
        elif early_schedule == 'exponential':
            t_rel = 1.0 - t / max(tau_split, 1e-8)
            alpha_t = alpha_0 * (np.exp(k * t_rel) - 1.0) / (np.exp(k) - 1.0)
        else:  # 'linear' (default / ours)
            alpha_t = alpha_0 * (1.0 - t / max(tau_split, 1e-8))
        return 1.0 + alpha_t
    else:
        t_rel = (t - tau_split) / max(1.0 - tau_split, 1e-8)
        if late_schedule == 'linear':
            beta_t = beta_0 * t_rel
        elif late_schedule == 'constant':
            beta_t = beta_0
        else:  # 'exponential' (default / ours)
            beta_t = beta_0 * (np.exp(k * t_rel) - 1.0)
        return 1.0 - beta_t

@torch.no_grad()
def sample_flow_euler_kts(model, n_images=25, config=TrainingConfig(),
                         fm=FlowMatchingConfig(), dim=3, n_steps=100,
                         alpha_0=0.0, beta_0=0.0, tau_split=0.6,
                         time_schedule='uniform',
                         early_schedule='linear', late_schedule='exponential'):
    '''
    Euler method with Kinetic Trajectory Shaping (KTS) for flow ODE.

    KTS modifies the velocity field using time-dependent scaling η(t):
    - Launch phase (t < τ_split): η = 1 + α(t) to increase exploration
    - Soft-landing phase (t ≥ τ_split): η = 1 - β(t) to reduce memorization

    Parameters:
    -----------
    model : nn.Module
        Trained velocity prediction network
    n_images : int
        Number of samples to generate
    config : TrainingConfig
        Configuration object with IMG_SHAPE, TIMESTEPS, DEVICE
    fm : FlowMatchingConfig
        Flow matching configuration
    dim : int
        Dimensionality: 2 for [B, N], 3 for [B, C, N], 4 for [B, C, H, W]
    n_steps : int
        Number of integration steps (default: 100)
    alpha_0 : float
        KTS launch intensity (default: 0.0, range: 0.0-0.5)
    beta_0 : float
        KTS soft-landing damping (default: 0.0, range: 0.0-0.5)
    tau_split : float
        KTS phase transition point (default: 0.6, range: 0.0-1.0)

    Returns:
    --------
    x : torch.Tensor
        Generated samples at t=1
    x_init : torch.Tensor
        Initial noise samples at t=0
    '''
    # Generate initial noise x_0 ~ N(0, 1)
    if dim == 4:  # [B, C, H, W]
        x_init = torch.randn(n_images, config.IMG_SHAPE[0],
                            config.IMG_SHAPE[1], config.IMG_SHAPE[2]).to(config.DEVICE)
    elif dim == 3:  # [B, C, N]
        x_init = torch.randn(n_images, config.IMG_SHAPE[0],
                            config.IMG_SHAPE[1]).to(config.DEVICE)
    elif dim == 2:  # [B, N]
        x_init = torch.randn(n_images, config.IMG_SHAPE[1]).to(config.DEVICE)

    x = x_init.clone()
    model.eval()

    times = _make_time_grid(n_steps, time_schedule)

    for i in range(n_steps):
        t_normalized = times[i]
        dt = times[i + 1] - times[i]
        t_discrete = int(t_normalized * (config.TIMESTEPS - 1))
        t_tensor = torch.ones(n_images, dtype=torch.long,
                             device=config.DEVICE) * t_discrete

        eta = _kts_eta(t_normalized, alpha_0, beta_0, tau_split,
                       early_schedule, late_schedule)

        v_t = model(x, t_tensor)
        x = x + eta * dt * v_t

    return x, x_init


@torch.no_grad()
def sample_flow_heun_kts(model, n_images=25, config=TrainingConfig(),
                        fm=FlowMatchingConfig(), dim=3, n_steps=100,
                        alpha_0=0.0, beta_0=0.0, tau_split=0.6,
                        time_schedule='uniform',
                        early_schedule='linear', late_schedule='exponential'):
    '''
    Heun's method with Kinetic Trajectory Shaping (KTS) for flow ODE.

    KTS modifies the velocity field using time-dependent scaling η(t):
    - Launch phase (t < τ_split): η = 1 + α(t) to increase exploration
    - Soft-landing phase (t ≥ τ_split): η = 1 - β(t) to reduce memorization

    This is the RECOMMENDED KTS solver as it provides better accuracy than Euler.

    Algorithm:
    1. Predictor: x_temp = x + η(t) * dt * v(x, t)
    2. Corrector: x_new = x + η_avg * dt/2 * (v(x, t) + v(x_temp, t+dt))

    Parameters:
    -----------
    model : nn.Module
        Trained velocity prediction network
    n_images : int
        Number of samples to generate
    config : TrainingConfig
        Configuration object
    fm : FlowMatchingConfig
        Flow matching configuration
    dim : int
        Dimensionality of tensor
    n_steps : int
        Number of integration steps (default: 100)
    alpha_0 : float
        KTS launch intensity (default: 0.0, range: 0.0-0.5)
    beta_0 : float
        KTS soft-landing damping (default: 0.0, range: 0.0-0.5)
    tau_split : float
        KTS phase transition point (default: 0.6, range: 0.0-1.0)

    Returns:
    --------
    x : torch.Tensor
        Generated samples
    x_init : torch.Tensor
        Initial noise
    '''
    # Generate initial noise
    if dim == 4:
        x_init = torch.randn(n_images, config.IMG_SHAPE[0],
                            config.IMG_SHAPE[1], config.IMG_SHAPE[2]).to(config.DEVICE)
    elif dim == 3:
        x_init = torch.randn(n_images, config.IMG_SHAPE[0],
                            config.IMG_SHAPE[1]).to(config.DEVICE)
    elif dim == 2:
        x_init = torch.randn(n_images, config.IMG_SHAPE[1]).to(config.DEVICE)

    x = x_init.clone()
    model.eval()

    times = _make_time_grid(n_steps, time_schedule)

    for i in range(n_steps):
        t_normalized = times[i]
        t_next_normalized = times[i + 1]
        dt = t_next_normalized - t_normalized
        eta_t      = _kts_eta(t_normalized,      alpha_0, beta_0, tau_split, early_schedule, late_schedule)
        eta_t_next = _kts_eta(t_next_normalized, alpha_0, beta_0, tau_split, early_schedule, late_schedule)
        eta_avg = 0.5 * (eta_t + eta_t_next)

        t_discrete = int(t_normalized * (config.TIMESTEPS - 1))
        t_tensor = torch.ones(n_images, dtype=torch.long,
                             device=config.DEVICE) * t_discrete

        v_t = model(x, t_tensor)
        x_temp = x + eta_t * dt * v_t

        t_next_discrete = min(int(t_next_normalized * (config.TIMESTEPS - 1)), config.TIMESTEPS - 1)
        t_next_tensor = torch.ones(n_images, dtype=torch.long,
                                   device=config.DEVICE) * t_next_discrete
        v_t_next = model(x_temp, t_next_tensor)

        x = x + eta_avg * dt * (v_t + v_t_next) * 0.5

    return x, x_init


@torch.no_grad()
def sample_flow_rk45_kts(model, n_images=25, config=TrainingConfig(),
                         fm=FlowMatchingConfig(), dim=3, n_steps=100,
                         alpha_0=0.0, beta_0=0.0, tau_split=0.6,
                         early_schedule='linear', late_schedule='exponential'):
    '''
    RK45 adaptive ODE solver with Kinetic Trajectory Shaping (KTS).

    Requires: pip install torchdiffeq
    '''
    try:
        from torchdiffeq import odeint
    except ImportError:
        raise ImportError("torchdiffeq is required for RK45: pip install torchdiffeq")

    if dim == 4:
        x_init = torch.randn(n_images, config.IMG_SHAPE[0],
                            config.IMG_SHAPE[1], config.IMG_SHAPE[2]).to(config.DEVICE)
    elif dim == 3:
        x_init = torch.randn(n_images, config.IMG_SHAPE[0],
                            config.IMG_SHAPE[1]).to(config.DEVICE)
    elif dim == 2:
        x_init = torch.randn(n_images, config.IMG_SHAPE[1]).to(config.DEVICE)

    model.eval()

    def ode_func_kts(t, x):
        t_float = t.item() if hasattr(t, 'item') else float(t)
        eta = _kts_eta(t_float, alpha_0, beta_0, tau_split, early_schedule, late_schedule)
        t_discrete = min(int(t_float * (config.TIMESTEPS - 1)), config.TIMESTEPS - 1)
        t_tensor = torch.ones(x.shape[0], dtype=torch.long, device=config.DEVICE) * t_discrete
        return eta * model(x, t_tensor)

    t_span = torch.linspace(0, 1, n_steps).to(config.DEVICE)
    solution = odeint(ode_func_kts, x_init, t_span, method='dopri5', rtol=1e-5, atol=1e-5)

    return solution[-1], x_init
