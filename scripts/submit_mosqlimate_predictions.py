from __future__ import annotations
import argparse
import os
import re
import subprocess
from pathlib import Path
import pandas as pd
REQUIRED_COLUMNS = ['date', 'pred', 'lower_50', 'upper_50', 'lower_80', 'upper_80', 'lower_90', 'upper_90', 'lower_95', 'upper_95']
EXPECTED_STATES = 26
NESTED_COLUMNS = ['lower_95', 'lower_90', 'lower_80', 'lower_50', 'pred', 'upper_50', 'upper_80', 'upper_90', 'upper_95']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate/upload hard-pulse IMDC 2026 dengue state forecasts.')
    parser.add_argument('--forecast-dir', type=Path, default=Path('outputs/challenge_hardpulse/mandatory_state_forecasts'), help='Directory containing target_i/adm1_XX.csv files.')
    parser.add_argument('--repository', required=True, help='Model repository in "owner/repo" form, e.g. user/3rd_imdc_inst_team.')
    parser.add_argument('--commit', default=None, help='Git commit hash for the submitted model code. Defaults to HEAD.')
    parser.add_argument('--api-key', default=None, help='Mosqlimate API key. Defaults to MOSQLIMATE_API_KEY or API_KEY.')
    parser.add_argument('--description', default='dengue_hardpulse validation forecast', help='Description attached to each uploaded prediction.')
    parser.add_argument('--targets', default='1,2,3,4', help='Comma-separated validation targets to submit.')
    parser.add_argument('--submit', action='store_true', help='Actually upload predictions. Without this, only validate and print.')
    parser.add_argument('--published', action=argparse.BooleanOptionalAction, default=True, help='Whether uploaded predictions should be public.')
    return parser.parse_args()

def current_commit() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception as exc:
        raise SystemExit('Could not infer git commit; pass --commit explicitly') from exc

def target_numbers(value: str) -> list[int]:
    targets = [int(v.strip()) for v in value.split(',') if v.strip()]
    bad = [t for t in targets if t not in (1, 2, 3, 4)]
    if bad:
        raise SystemExit(f'Only validation targets 1..4 are supported, got {bad}')
    return targets

def validate_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'{path}: missing columns {missing}')
    df = df[REQUIRED_COLUMNS].copy()
    dates = pd.to_datetime(df['date'], errors='raise')
    if not (dates.dt.dayofweek == 6).all():
        raise ValueError(f'{path}: all dates must be Sundays')
    if len(dates) > 1 and (not (dates.diff().dropna() == pd.Timedelta(days=7)).all()):
        raise ValueError(f'{path}: dates must be continuous weekly Sundays')
    value_cols = [c for c in REQUIRED_COLUMNS if c != 'date']
    if df[value_cols].isna().any().any():
        raise ValueError(f'{path}: prediction columns contain NaN')
    if (df[value_cols] < 0).any().any():
        raise ValueError(f'{path}: prediction columns must be non-negative')
    nested = df[NESTED_COLUMNS].to_numpy(float)
    if not (nested[:, :-1] <= nested[:, 1:] + 1e-08).all():
        raise ValueError(f'{path}: intervals are not nested')
    df['date'] = dates.dt.strftime('%Y-%m-%d')
    return df

def iter_prediction_files(root: Path, targets: list[int]) -> list[tuple[int, int, Path]]:
    items = []
    for target in targets:
        target_dir = root / f'target_{target}'
        if not target_dir.exists():
            raise FileNotFoundError(f'Missing target directory: {target_dir}')
        files = sorted(target_dir.glob('adm1_*.csv'))
        if len(files) != EXPECTED_STATES:
            raise ValueError(f'{target_dir}: expected {EXPECTED_STATES} state files, found {len(files)}')
        for fpath in files:
            match = re.fullmatch('adm1_(\\d{2})\\.csv', fpath.name)
            if not match:
                raise ValueError(f'Unexpected filename: {fpath}')
            adm_1 = int(match.group(1))
            if adm_1 == 32:
                raise ValueError(f'Espirito Santo must be excluded, found {fpath}')
            items.append((target, adm_1, fpath))
    return items

def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv('MOSQLIMATE_API_KEY') or os.getenv('API_KEY')
    if args.submit and (not api_key):
        raise SystemExit('Pass --api-key or set MOSQLIMATE_API_KEY/API_KEY to submit')
    commit = args.commit or current_commit()
    targets = target_numbers(args.targets)
    items = iter_prediction_files(args.forecast_dir, targets)
    print(f'Repository : {args.repository}')
    print(f'Commit     : {commit}')
    print(f'Forecasts  : {len(items)} files under {args.forecast_dir}')
    print(f"Mode       : {('SUBMIT' if args.submit else 'dry-run')}")
    upload_prediction = None
    if args.submit:
        try:
            from mosqlient import upload_prediction
        except ImportError as exc:
            raise SystemExit('Install mosqlient first: pip install -U mosqlient') from exc
    for target, adm_1, fpath in items:
        prediction = validate_prediction(fpath)
        description = f'{args.description}; validation target {target}; adm1 {adm_1:02d}'
        print(f'target_{target} adm1_{adm_1:02d}: {len(prediction)} rows OK')
        if args.submit:
            result = upload_prediction(api_key=api_key, repository=args.repository, description=description, commit=commit, disease='A90', case_definition='probable', adm_level=1, adm_0='BRA', adm_1=adm_1, adm_2=None, adm_3=None, published=args.published, prediction=prediction)
            print(result)
    print('Done.')
if __name__ == '__main__':
    main()
