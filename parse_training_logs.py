"""
parse_training_logs.py

Scans the project directory (excluding common virtualenv and VCS folders) for
logs and training data files (.csv, .json, .txt, .log and any files inside
folders named logs/, training/, runs/, results/, output/). Attempts to parse
each file and normalize metrics into a single CSV `combined_training_data.csv`.

Usage:
  python parse_training_logs.py

This script is conservative: it excludes `venv`, `.venv`, `env`, and `.git` by
default to avoid scanning package or environment files. If you want to include
those, run the script with `--include-venv`.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd

# Aliases for common metric columns (lowercased)
ALIASES = {
    'episode': ['episode', 'ep', 'episode_nr', 'episode_number', 'episode_id'],
    'timesteps': ['timesteps', 'total_timesteps', 'step', 'steps', 'timestep', 'frame', 'frames'],
    'reward': ['reward', 'episode_reward', 'return', 'ep_reward', 'mean_reward', 'rew'],
    'episode_length': ['length', 'episode_length', 'ep_length', 'path_length', 'steps'],
    'avg_length': ['avg_length', 'average_length', 'mean_length'],
    'collisions': ['collision', 'collisions', 'bump', 'bumps', 'collision_count'],
    'success': ['success', 'success_rate', 'succeeded', 'is_success', 'done', 'win'],
    'actor_loss': ['actor_loss', 'policy_loss'],
    'critic_loss': ['critic_loss', 'value_loss'],
    'entropy': ['entropy', 'policy_entropy']
}

# File extensions to consider
EXTS = {'.csv', '.json', '.txt', '.log'}

# Directory names to include (any file inside these folders will be considered)
INCLUDE_DIR_NAMES = {'logs', 'training', 'runs', 'results', 'output'}

# Default directories to exclude to avoid scanning virtualenvs and git metadata
DEFAULT_EXCLUDE_DIRS = {'venv', '.venv', 'env', '.env', '.git', '__pycache__'}


def find_files(root: str, include_venv: bool=False) -> List[str]:
    files = []
    exclude = set(DEFAULT_EXCLUDE_DIRS)
    if include_venv:
        exclude = set()  # include everything

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip excluded dirs entirely
        parts = set(Path(dirpath).parts)
        if parts & exclude:
            # prune traversal by modifying dirnames in place
            dirnames[:] = [d for d in dirnames if d not in exclude]
            # but still continue, skip adding files under excluded dirs
            continue

        # If current path contains any of the include dir names, take all files
        parent_names = set(Path(dirpath).parts)
        take_all = bool(parent_names & INCLUDE_DIR_NAMES)

        for fn in filenames:
            path = os.path.join(dirpath, fn)
            ext = Path(fn).suffix.lower()
            if take_all or ext in EXTS:
                files.append(path)
    return files


def try_read_table(path: str) -> Optional[pd.DataFrame]:
    """Try to read a file as a table into a DataFrame using pandas.

    Tries CSV autodetection first, then JSON. For text files it will attempt
    CSV read with flexible separators, and fallback to line-based parsing.
    """
    p = Path(path)
    ext = p.suffix.lower()

    try:
        if ext == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Common structures: list of dicts, dict of lists, dict
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                # If top-level dict has 'data' or 'results', try that
                for key in ('data', 'results', 'episodes'):
                    if key in data and isinstance(data[key], (list, dict)):
                        return pd.DataFrame(data[key])
                # dict of lists
                try:
                    return pd.DataFrame(data)
                except Exception:
                    return None

        # Try reading as CSV / table with pandas
        # Use engine='python' and sep=None to let pandas sniff the separator
        df = pd.read_csv(path, sep=None, engine='python')
        if df.shape[1] <= 1:
            # maybe whitespace separated or space-padded; try delim_whitespace
            try:
                df2 = pd.read_csv(path, delim_whitespace=True)
                if df2.shape[1] > 1:
                    return df2
            except Exception:
                pass
        return df
    except Exception:
        # Last-resort: try to extract key:value pairs per line -> small DataFrame
        try:
            rows = []
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Try JSON-like line
                    if (line.startswith('{') and line.endswith('}')) or (line.startswith('[') and line.endswith(']')):
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                rows.append(obj)
                                continue
                        except Exception:
                            pass

                    # Try key: value; key=value; csv-like with commas
                    if ':' in line and not ',' in line:
                        parts = [p.strip() for p in line.split(':', 1)]
                        if len(parts) == 2:
                            k, v = parts
                            rows.append({k: v})
                            continue

                    # If comma-separated numbers with header not present, try parsing numeric list
                    # We'll skip complex heuristics here
                if rows:
                    return pd.DataFrame(rows)
        except Exception:
            return None
    return None


def standardize_df(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Map columns to canonical names; don't remove unknown columns.

    Adds `source_file` and attempts to detect `algo` from columns or filename.
    """
    df = df.copy()
    # Lowercase column names for matching
    col_map = {}
    lowered = {c: c.lower() for c in df.columns}

    for canon, alias_list in ALIASES.items():
        for c, lc in lowered.items():
            for alias in alias_list:
                if alias in lc:
                    col_map[c] = canon
                    break
            if c in col_map:
                break

    # Apply renames
    if col_map:
        df = df.rename(columns=col_map)

    # Make sure canonical columns exist (possibly missing)
    for k in ['episode', 'timesteps', 'reward', 'episode_length', 'avg_length', 'collisions', 'success', 'actor_loss', 'critic_loss', 'entropy']:
        if k not in df.columns:
            df[k] = pd.NA

    # Add source_file
    df['source_file'] = filename

    # Detect algo: prefer explicit column if present
    algo = None
    for cand in ('algo', 'algorithm', 'model', 'run_type'):
        if cand in df.columns:
            # take first non-null value
            try:
                val = df[cand].dropna().astype(str).iloc[0]
                algo = val
            except Exception:
                pass
            break

    if not algo:
        # Infer from filename
        fn = filename.lower()
        if 'lam' in fn and 'sac' in fn:
            algo = 'LAM_SAC'
        elif 'lam' in fn:
            algo = 'LAM'
        elif 'sac' in fn:
            algo = 'SAC'
        else:
            algo = ''

    df['algo'] = algo

    # Normalize success to boolean-ish numeric if possible
    if df['success'].dtype == object:
        df['success'] = df['success'].apply(lambda x: _parse_boolish(x))

    # Try converting numeric columns to numeric dtype
    numeric_cols = ['episode', 'timesteps', 'reward', 'episode_length', 'avg_length', 'collisions', 'actor_loss', 'critic_loss', 'entropy']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


def _parse_boolish(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().lower()
    if s in ('1', 'true', 't', 'yes', 'y', 'success', 'succeeded'):
        return 1
    if s in ('0', 'false', 'f', 'no', 'n', 'fail', 'failed'):
        return 0
    # Could be percentage like '0.98' treat >0 as success if ambiguous
    try:
        f = float(s)
        if f >= 0 and f <= 1:
            # interpret as proportion success or probability
            return 1 if f >= 0.5 else 0
        return int(f)
    except Exception:
        return pd.NA


def extract_from_file(path: str) -> Optional[pd.DataFrame]:
    df = try_read_table(path)
    if df is None or df.shape[0] == 0:
        return None
    try:
        std = standardize_df(df, os.path.relpath(path))
        return std
    except Exception as e:
        print(f"Failed to standardize {path}: {e}")
        return None


def collect_and_merge(root: str, include_venv: bool=False) -> pd.DataFrame:
    files = find_files(root, include_venv=include_venv)
    print(f"Found {len(files)} candidate files (filtered).")
    rows = []
    file_count = 0
    for f in files:
        try:
            df = extract_from_file(f)
            if df is not None:
                rows.append(df)
                file_count += 1
        except Exception as e:
            print(f"Error parsing {f}: {e}")
    if not rows:
        print("No parsable log tables found.")
        return pd.DataFrame()

    merged = pd.concat(rows, ignore_index=True, sort=False)
    # Re-order columns: preferred order then the rest
    preferred = ['source_file', 'algo', 'episode', 'timesteps', 'reward', 'episode_length', 'avg_length', 'collisions', 'success', 'actor_loss', 'critic_loss', 'entropy']
    other_cols = [c for c in merged.columns if c not in preferred]
    ordered = preferred + other_cols
    ordered_existing = [c for c in ordered if c in merged.columns]
    merged = merged[ordered_existing]
    return merged


def main():
    parser = argparse.ArgumentParser(description='Scan and merge training logs into one CSV')
    parser.add_argument('--root', default='.', help='Root directory to scan (default: project root)')
    parser.add_argument('--out', default='combined_training_data.csv', help='Output CSV filename')
    parser.add_argument('--include-venv', action='store_true', help='Include venv/.git and similar directories in scan')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    print(f"Scanning {root} (include_venv={args.include_venv})...")
    merged = collect_and_merge(root, include_venv=args.include_venv)
    if merged.shape[0] == 0:
        print('No data to write. Exiting.')
        return
    out_path = os.path.join(root, args.out)
    merged.to_csv(out_path, index=False)
    print(f"Wrote {merged.shape[0]} rows and {merged.shape[1]} columns to {out_path}")


if __name__ == '__main__':
    main()
