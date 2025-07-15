import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from omegaconf import OmegaConf
from src.inference.smc import SMCFilter
from src.models.channel_model import DelayDopplerChannel
from src.utils.simulators import generate_symbols, generate_channel_outputs
from src.utils.metrics import compute_ber
from src.utils.plotting import plot_multiple_metrics
from src.utils.logger import init_wandb, log_wandb, save_csv
from src.training.evaluate import evaluate_run


def base_delay_sampler(min_delay, max_delay):
    return lambda: np.random.uniform(min_delay, max_delay)


def channel_factory(cfg):
    return DelayDopplerChannel(
        alpha=cfg.dp_alpha,
        base_delay_sampler=base_delay_sampler(*cfg.delay_range),
        kernel_type=cfg.kernel_type
    )


def run_smc(cfg, num_particles, tx_symbols, rx_signal, train_times, true_gains, noise_var):
    smc = SMCFilter(
        num_particles=num_particles,
        channel_factory=lambda: channel_factory(cfg),
        noise_var=noise_var,
        resample_method=cfg.resample_method
    )
    smc.initialize(train_times)
    est_symbols = []

    for n in range(len(tx_symbols)):
        x_hist = tx_symbols[max(0, n - 10):n + 1]
        smc.step(rx_signal[n], x_hist, time_n=n)
        y_est, _ = smc.estimate_symbol(x_hist, time_n=n)
        est_symbols.append(np.sign(np.real(y_est)))

    # Evaluate and return metrics
    ber, mse = evaluate_run(
        cfg=cfg,
        tx_symbols=tx_symbols,
        est_symbols=est_symbols,
        true_gains=true_gains,
        smc_particles=smc.particles,
        time_points=train_times,
        silent=True
    )
    return ber, mse


if __name__ == '__main__':
    cfg = OmegaConf.load("configs/default.yaml")
    np.random.seed(123)

    num_symbols = cfg.num_symbols
    snr_db = cfg.snr_db
    noise_var = 10 ** (-snr_db / 10) if cfg.noise_var is None else cfg.noise_var
    train_times = np.linspace(0, num_symbols - 1, num_symbols)

    tx_symbols = generate_symbols(num_symbols, modulation=cfg.modulation)
    true_channel = channel_factory(cfg)
    true_channel.sample_paths(train_times)
    rx_signal, true_gains = generate_channel_outputs(true_channel, tx_symbols, noise_var, train_times)

    particle_list = cfg.particle_list
    ber_list, mse_list = [], []

    if cfg.logging.use_wandb:
        init_wandb(
            project=cfg.logging.project_name,
            run_name="complexity-sweep",
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    csv_rows = []
    csv_header = ["Np", "BER", "MSE"]

    for Np in particle_list:
        print(f"Running with {Np} particles...")
        ber, mse = run_smc(cfg, Np, tx_symbols, rx_signal, train_times, true_gains, noise_var)
        ber_list.append(ber)
        mse_list.append(mse)

        print(f"Np = {Np}, BER = {ber:.4f}, MSE = {mse:.4f}")

        if cfg.logging.use_wandb:
            log_wandb({"BER": ber, "MSE": mse, "Np": Np})
        if cfg.logging.use_csv:
            csv_rows.append([Np, ber, mse])

    if cfg.logging.use_csv:
        os.makedirs(cfg.logging.save_dir, exist_ok=True)
        log_path = os.path.join(cfg.logging.save_dir, "complexity-sweep.csv")
        save_csv(csv_rows, log_path, header=csv_header)

    plot_multiple_metrics(
        x_vals=np.array(particle_list),
        metric_dict={"BER": ber_list, "MSE": mse_list},
        param_name="Number of Particles (Np)",
        log_x=True
    )
