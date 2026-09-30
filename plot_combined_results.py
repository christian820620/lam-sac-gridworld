"""
plot_combined_results.py

Read `combined_training_data.csv` (created by `parse_training_logs.py`) and
produce poster-quality PNGs for:
 - reward curve (if reward data exists)
 - average path length / episode length per stage or source
 - success rate per stage or source

Saves PNGs to `figures/` directory.

Usage:
  python plot_combined_results.py

"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('.')
IN = ROOT / 'combined_training_data.csv'
OUTDIR = ROOT / 'figures'
OUTDIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8')

def load():
    if not IN.exists():
        print(f"Input file {IN} not found. Run parse_training_logs.py first.")
        return None
    df = pd.read_csv(IN)
    return df


def plot_reward(df: pd.DataFrame):
    # Reward curves: prefer rows with numeric reward
    if 'reward' not in df.columns or df['reward'].dropna().shape[0] == 0:
        print('No reward data found; skipping reward curve.')
        return None

    r = df.copy()
    # If timesteps exist, plot reward vs timesteps aggregated per source_file
    if 'timesteps' in r.columns and r['timesteps'].notna().sum() > 0:
        groups = r.groupby('source_file')
        plt.figure(figsize=(8,4.5))
        for name, g in groups:
            gg = g.sort_values('timesteps')
            x = gg['timesteps']
            y = gg['reward']
            if y.isnull().all():
                continue
            # rolling for smoothness
            y_s = y.fillna(method='ffill').rolling(window=max(1, int(len(y)/20))).mean()
            plt.plot(x, y_s, label=name)
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.title('Reward curve')
        plt.legend(fontsize='small')
        plt.tight_layout()
        out = OUTDIR / 'reward_curve.png'
        plt.savefig(out, dpi=200)
        plt.close()
        print(f'Saved reward curve to {out}')
        return out
    else:
        # Plot reward vs index per source
        plt.figure(figsize=(8,4.5))
        groups = df.groupby('source_file')
        for name, g in groups:
            y = g['reward']
            if y.isnull().all():
                continue
            y_s = y.fillna(method='ffill').rolling(window=max(1, int(len(y)/20))).mean()
            plt.plot(y_s.values, label=name)
        plt.xlabel('Episode index')
        plt.ylabel('Reward')
        plt.title('Reward curve (index)')
        plt.legend(fontsize='small')
        plt.tight_layout()
        out = OUTDIR / 'reward_curve_index.png'
        plt.savefig(out, dpi=200)
        plt.close()
        print(f'Saved reward curve (index) to {out}')
        return out


def plot_path_length(df: pd.DataFrame):
    # Prefer avg_length if available, else episode_length
    key = None
    if 'avg_length' in df.columns and df['avg_length'].notna().sum() > 0:
        key = 'avg_length'
    elif 'episode_length' in df.columns and df['episode_length'].notna().sum() > 0:
        key = 'episode_length'

    if key is None:
        print('No path/episode length data found; skipping path-length plot.')
        return None

    # If stage column exists, group by stage and algo
    if 'stage' in df.columns and df['stage'].notna().sum() > 0:
        # Try numeric stage
        try:
            grouped = df.dropna(subset=[key]).groupby(['stage', 'algo'])[key].mean().unstack(fill_value=np.nan)
            ax = grouped.plot(kind='bar', figsize=(8,4.5))
            ax.set_xlabel('Stage')
            ax.set_ylabel(key)
            ax.set_title(f'{key} by Stage')
            plt.tight_layout()
            out = OUTDIR / 'path_length_by_stage.png'
            plt.savefig(out, dpi=200)
            plt.close()
            print(f'Saved path-length by stage to {out}')
            return out
        except Exception as e:
            print('Failed grouping by stage:', e)

    # Else group by source_file
    grouped = df.dropna(subset=[key]).groupby('source_file')[key].mean().sort_values()
    plt.figure(figsize=(8,4.5))
    grouped.plot(kind='barh')
    plt.xlabel(key)
    plt.title(f'Average {key} by source')
    plt.tight_layout()
    out = OUTDIR / 'path_length_by_source.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'Saved path-length by source to {out}')
    return out


def plot_success(df: pd.DataFrame):
    if 'success' not in df.columns or df['success'].dropna().shape[0] == 0:
        print('No success data found; skipping success plot.')
        return None

    # Normalize success to numeric
    s = df.copy()
    s['success'] = pd.to_numeric(s['success'], errors='coerce')

    if 'stage' in s.columns and s['stage'].notna().sum() > 0:
        grouped = s.groupby('stage')['success'].mean()
        plt.figure(figsize=(6,3.5))
        grouped.plot(kind='bar')
        plt.xlabel('Stage')
        plt.ylabel('Success rate')
        plt.ylim(0,1.05)
        plt.title('Success rate by Stage')
        plt.tight_layout()
        out = OUTDIR / 'success_by_stage.png'
        plt.savefig(out, dpi=200)
        plt.close()
        print(f'Saved success by stage to {out}')
        return out

    grouped = s.groupby('source_file')['success'].mean().sort_values()
    plt.figure(figsize=(8,4.5))
    grouped.plot(kind='barh')
    plt.xlabel('Success rate')
    plt.xlim(0,1.05)
    plt.title('Success rate by source')
    plt.tight_layout()
    out = OUTDIR / 'success_by_source.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'Saved success by source to {out}')
    return out


def plot_collisions(df: pd.DataFrame):
    if 'collisions' not in df.columns or df['collisions'].dropna().shape[0] == 0:
        print('No collisions data; skipping collisions plot.')
        return None
    grouped = df.groupby('source_file')['collisions'].mean().sort_values()
    plt.figure(figsize=(8,4.5))
    grouped.plot(kind='barh')
    plt.xlabel('Average collisions')
    plt.title('Average collisions by source')
    plt.tight_layout()
    out = OUTDIR / 'collisions_by_source.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'Saved collisions by source to {out}')
    return out


def main():
    df = load()
    if df is None:
        return
    outputs = {}
    outputs['reward'] = plot_reward(df)
    outputs['path_length'] = plot_path_length(df)
    outputs['success'] = plot_success(df)
    outputs['collisions'] = plot_collisions(df)

    # Also produce direct SAC vs LAM+SAC comparison plots
    outputs['compare'] = compare_lam_vs_sac(df)

    print('\nGenerated plots:')
    for k,v in outputs.items():
        print(f' - {k}: {v}')


def compare_lam_vs_sac(df: pd.DataFrame):
    """Create comparison plots that directly compare LAM+SAC vs SAC.

    Heuristic tagging: if `algo` already present and non-empty, keep it.
    Otherwise, infer from `source_file` name: filenames containing 'lam' -> LAM_SAC,
    filenames containing 'sac' (but not 'lam') -> SAC, filenames with 'training' or
    'trial' are assumed LAM_SAC (because this repo used training scripts named
    `train_*` for LAM+SAC). This is conservative and printed for user review.
    """
    out_files = []
    df2 = df.copy()
    # ensure algo column exists
    if 'algo' not in df2.columns:
        df2['algo'] = ''

    # Infer missing algo tags from filenames
    for i, row in df2.iterrows():
        a = str(row.get('algo', '') or '').strip()
        if a:
            continue
        src = str(row.get('source_file', '')).lower()
        tag = ''
        if 'lam' in src and 'sac' in src:
            tag = 'LAM_SAC'
        elif 'lam' in src:
            tag = 'LAM_SAC'
        elif any(k in src for k in ('training', 'trial', 'trials', 'train')):
            tag = 'LAM_SAC'
        elif 'sac' in src:
            tag = 'SAC'
        else:
            tag = ''
        df2.at[i, 'algo'] = tag

    # Summary
    tags = pd.Series(df2['algo'].fillna('')).value_counts()
    print('Inferred algo tag counts:')
    print(tags.to_dict())

    # Path-length comparison
    key = None
    if 'avg_length' in df2.columns and df2['avg_length'].notna().sum() > 0:
        key = 'avg_length'
    elif 'episode_length' in df2.columns and df2['episode_length'].notna().sum() > 0:
        key = 'episode_length'

    if key is not None:
        try:
            grp = df2.dropna(subset=[key]).groupby('algo')[key].mean().reindex(['SAC', 'LAM_SAC'])
            plt.figure(figsize=(6,4))
            grp.plot(kind='bar', color=['#4C72B0', '#DD8452'])
            plt.ylabel(key)
            plt.title(f'Average {key}: SAC vs LAM+SAC')
            plt.tight_layout()
            out = OUTDIR / 'compare_path_length_sac_vs_lam_sac.png'
            plt.savefig(out, dpi=300)
            plt.close()
            out_files.append(out)
            print(f'Saved SAC vs LAM+SAC path-length comparison to {out}')
        except Exception as e:
            print('Failed to create path-length comparison:', e)

    # Success comparison
    if 'success' in df2.columns and df2['success'].notna().sum() > 0:
        try:
            s = df2.copy()
            s['success'] = pd.to_numeric(s['success'], errors='coerce')
            grp = s.groupby('algo')['success'].mean().reindex(['SAC', 'LAM_SAC'])
            plt.figure(figsize=(6,4))
            grp.plot(kind='bar', color=['#4C72B0', '#DD8452'])
            plt.ylabel('Success rate')
            plt.ylim(0,1)
            plt.title('Success rate: SAC vs LAM+SAC')
            plt.tight_layout()
            out = OUTDIR / 'compare_success_sac_vs_lam_sac.png'
            plt.savefig(out, dpi=300)
            plt.close()
            out_files.append(out)
            print(f'Saved SAC vs LAM+SAC success comparison to {out}')
        except Exception as e:
            print('Failed to create success comparison:', e)

    if not out_files:
        print('No comparison plots created (missing metrics).')
        return None
    return out_files


if __name__ == '__main__':
    main()
