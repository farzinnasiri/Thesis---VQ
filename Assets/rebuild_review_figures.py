"""Regenerate revised thesis plots from archived aggregate measurements."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Images'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 220})
MODES = ['closest', 'orthogonal', 'random_uniform', 'farthest']
NAMES = ['Closest', 'Orthogonal', 'Random', 'Farthest']
MODELS = ['llamagen', 'vqgan']
TITLES = ['LlamaGen', 'VQGAN']
MODEL_COLORS = ['#2F70A2', '#BF5934']
COLORS = ['#0072B2', '#009E73', '#E69F00', '#CC79A7']

def save(fig, name):
    fig.savefig(OUT / name, bbox_inches='tight')
    plt.close(fig)

# Reuse paired clean reconstructions archived for the robustness experiment to
# replace the original fish-only reconstruction montage with varied examples.
sample_ids = [
    'ILSVRC2012_val_00025815',
    'ILSVRC2012_val_00009569',
    'ILSVRC2012_val_00041991',
]
fig, axes = plt.subplots(len(sample_ids), 3, figsize=(7.5, 7.5), layout='constrained')
for row, sample_id in enumerate(sample_ids):
    base = ROOT / 'Assets/rq2/qualitative_samples'
    paths = [
        base / 'llamagen' / sample_id / '0_original.png',
        base / 'vqgan' / sample_id / '1_recon_clean.png',
        base / 'llamagen' / sample_id / '1_recon_clean.png',
    ]
    for col, path in enumerate(paths):
        axes[row, col].imshow(Image.open(path).convert('RGB'))
        axes[row, col].axis('off')
for ax, title in zip(axes[0], ['Original', 'VQGAN', 'LlamaGen']):
    ax.set_title(title)
save(fig, 'ch04_fig_reconstruction_triplets.png')

for model in MODELS:
    fig, axes = plt.subplots(4, 4, figsize=(8, 8.7), layout='constrained')
    for sample_index, sample_id in enumerate(sample_ids[:2]):
        base = ROOT / 'Assets/rq2/qualitative_samples' / model / sample_id
        for row_offset, files in enumerate([
            ['0_original.png', '2_input_noise_low.png', '4_input_noise_mid.png', '6_input_noise_high.png'],
            ['1_recon_clean.png', '3_recon_noise_low.png', '5_recon_noise_mid.png', '7_recon_noise_high.png'],
        ]):
            row = 2 * sample_index + row_offset
            for col, filename in enumerate(files):
                ax = axes[row, col]
                ax.imshow(Image.open(base / filename).convert('RGB'))
                ax.set(xticks=[], yticks=[])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            axes[row, 0].set_ylabel('Input' if row_offset == 0 else r'$D(Q(E(x_\sigma)))$')
        axes[2 * sample_index, 0].set_title(sample_id.removeprefix('ILSVRC2012_val_'), fontsize=9, loc='left')
    for col, title in enumerate(['Clean', r'$\sigma=0.1$', r'$\sigma=0.2$', r'$\sigma=0.5$']):
        axes[0, col].set_title(title)
    save(fig, f'ch05_fig_global_noise_qualitative_{model}.png')

# The token-response maps are binary. Remove the old continuous colorbar; the
# thesis caption now gives the two-value legend explicitly.
for model in ['vqgan', 'llamagen']:
    source = ROOT / f'Assets/rq3/figures/ch06_fig_patch_noise_qualitative_{model}.png'
    source_image = Image.open(source)
    source_image.crop((0, 0, min(3860, source_image.width), source_image.height)).save(
        OUT / f'ch06_fig_patch_noise_qualitative_{model}.png'
    )

def sweep(frame, column, ylabel, name):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True, layout='constrained')
    for ax, model, title in zip(axes, MODELS, TITLES):
        for mode, label, color in zip(MODES, NAMES, COLORS):
            d = frame[(frame.model.str.lower() == model) & (frame['mode'] == mode)].sort_values('fraction_label')
            ax.plot(d.fraction_label, d[column], marker='o', label=label, color=color)
        ax.set(title=title, xlabel='Target patch fraction (%)', xticks=[10, 25, 50, 75])
        ax.grid(alpha=.2)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(fontsize=8)
    save(fig, name)

fidelity = pd.read_csv(ROOT / 'Assets/rq4/data/ch07_table_fidelity_summary.csv')
fidelity = fidelity[fidelity.dataset == 'imagenet_val']
locality = pd.read_csv(ROOT / 'Assets/rq4/data/ch07_table_locality_summary.csv')
locality = locality[locality.dataset == 'imagenet_val']
sweep(fidelity, 'lpips_patch_mean', 'Mean patch LPIPS', 'ch07_fig_patch_lpips_imagenet_val.png')
sweep(locality, 'leakage_ratio_median', 'Median outside / inside response', 'ch07_fig_leakage_ratio_imagenet_val.png')

fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout='constrained')
for ax, metric, label in zip(axes, ['psnr_patch_mean', 'ssim_patch_mean', 'lpips_patch_mean'], ['Patch PSNR (dB)', 'Patch SSIM', 'Patch LPIPS']):
    for offset, model, title, color in zip([-.18, .18], MODELS, TITLES, MODEL_COLORS):
        d = fidelity[(fidelity.model.str.lower() == model) & (fidelity.fraction_label == 25)].set_index('mode')
        ax.bar([i+offset for i in range(4)], d.loc[MODES, metric], width=.36, label=title, color=color)
    ax.set_ylabel(label)
    ax.set_xticks(range(4), NAMES, rotation=35, ha='right')
axes[0].legend(fontsize=8)
save(fig, 'ch07_fig_fidelity_patch25_imagenet_val.png')

fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey='row', layout='constrained')
for col, model, title in zip(range(2), MODELS, TITLES):
    for row, metric in enumerate(['inside_change_mean', 'outside_change_mean']):
        for mode, label, color in zip(MODES, NAMES, COLORS):
            d = locality[(locality.model.str.lower() == model) & (locality['mode'] == mode)].sort_values('fraction_label')
            axes[row, col].plot(d.fraction_label, d[metric], marker='o', color=color, label=label)
        axes[row, col].set_xticks([10, 25, 50, 75])
        axes[row, col].grid(alpha=.2)
    axes[0, col].set_title(title)
    axes[1, col].set_xlabel('Target patch fraction (%)')
axes[0, 0].set_ylabel('Mean inside RMSE')
axes[1, 0].set_ylabel('Mean outside RMSE')
axes[0, 1].legend(fontsize=8)
save(fig, 'ch07_fig_decoder_locality_imagenet_val.png')

shift = pd.read_csv(ROOT / 'Assets/rq4/data/ch07_table_dataset_shift_imagenetv2_minus_val.csv')
sweep(shift[shift.metric == 'lpips_patch'], 'delta', 'Mean patch LPIPS: V2 minus ID', 'ch07_fig_dataset_shift_lpips_patch.png')

usage = pd.read_csv(ROOT / 'Assets/rq5/data/ch08_global_code_usage.csv')
order = ['imagenet_v2', 'objectnet', 'imagenet_sketch', 'organamnist', 'bloodmnist', 'rvlcdip']
labels = ['ImageNet-V2', 'ObjectNet*', 'Sketch', 'OrganAMNIST*', 'BloodMNIST', 'RVL-CDIP']
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
for ax, metric, label in zip(axes, ['js_divergence_vs_imagenet', 'perplexity_ratio_vs_imagenet'], ['JSD (nats)', r'$R_P$']):
    for offset, model, color in zip([-.18, .18], TITLES, MODEL_COLORS):
        d = usage[usage.model == model].set_index('dataset')
        ax.bar([i+offset for i in range(6)], d.loc[order, metric], width=.36, label=model, color=color)
    ax.set_xticks(range(6), labels, rotation=40, ha='right')
    ax.set_ylabel(label)
axes[0].legend(fontsize=8)
axes[1].axhline(1, color='grey', lw=.8, linestyle='--')
save(fig, 'ch08_fig_global_shift_summary.png')

# Both checkpoints have K=16384. Scaling by log(K) expresses the entropy
# difference as a fraction of the theoretical positional entropy range.
entropy = pd.read_csv(ROOT / 'Assets/rq5/data/ch08_positional_entropy_summary.csv')
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, layout='constrained')
for ax, field, title in zip(axes, ['mean_entropy_delta', 'border_entropy_delta', 'interior_entropy_delta'], ['All positions', 'Border positions', 'Interior positions']):
    for offset, model, color in zip([-.18, .18], TITLES, MODEL_COLORS):
        d = entropy[entropy.model == model].set_index('dataset')
        ax.bar([i+offset for i in range(6)], d.loc[order, field] / np.log(16384), width=.36, label=model, color=color)
    ax.set_xticks(range(6), labels, rotation=45, ha='right')
    ax.set_title(title)
    ax.axhline(0, color='grey', lw=.8)
axes[0].set_ylabel(r'Mean $\Delta H / \log K$')
axes[0].legend(fontsize=8)
save(fig, 'ch08_fig_normalized_positional_entropy.png')

# Presentation-only redraws from the stored aggregate tables. No inference or
# statistical estimation is performed here.
noise = pd.read_csv(ROOT / 'Assets/rq2/data/ch05_table_token_distribution_summary.csv')
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained')
for model, title, color in zip(MODELS, TITLES, MODEL_COLORS):
    d = noise[noise.model == model].sort_values('sigma')
    axes[0].plot(d.sigma, d.perplexity, 'o-', label=title, color=color)
    axes[1].plot(d.sigma, d.perplexity / d.iloc[0].perplexity, 'o-', label=title, color=color)
for ax in axes:
    ax.set(xlabel=r'Noise standard deviation $\sigma$', xticks=[0, .1, .2, .5])
    ax.grid(alpha=.2)
axes[0].set(ylabel='Perplexity', ylim=(0, 17500))
axes[1].set(ylabel='Perplexity / clean perplexity', ylim=(0, 1.05))
axes[0].legend()
save(fig, 'ch05_fig_perplexity_vs_sigma.png')

encoder = pd.read_csv(ROOT / 'Assets/rq3/data/ch06_table_inside_outside_summary.csv')
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained')
for title, color in zip(TITLES, MODEL_COLORS):
    d = encoder[encoder.model == title].sort_values('sigma')
    axes[0].plot(d.sigma, d.inside_response_mean, 'o-', label=f'{title} inside', color=color)
    axes[0].plot(d.sigma, d.outside_response_mean, 's--', label=f'{title} outside', color=color)
    axes[1].plot(d.sigma, d.leakage_ratio_mean, 'o-', label=title, color=color)
for ax in axes:
    ax.set(xlabel=r'Patch-noise standard deviation $\sigma$', xticks=[.1, .25, .5], ylim=(0, 1.05))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
axes[0].set_ylabel('Mean token-flip fraction')
axes[1].set_ylabel('Mean per-image outside / inside ratio')
save(fig, 'ch06_fig_inside_outside_response.png')

distance = pd.read_csv(ROOT / 'Assets/rq3/data/ch06_table_distance_profile.csv')
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True, layout='constrained')
for ax, model, title in zip(axes, MODELS, TITLES):
    for sigma, color in zip([.1, .25, .5], COLORS):
        d = distance[(distance.model.str.lower() == model) & (distance.sigma == sigma)].sort_values('distance')
        ax.plot(d.distance, d.flip_probability, 'o-', color=color, label=rf'$\sigma={sigma}$')
    ax.set(title=title, xlabel='Manhattan distance from patch (token cells)', ylim=(0, 1.05))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
axes[0].set_ylabel('Token-flip probability')
save(fig, 'ch06_fig_distance_profile_flip.png')
