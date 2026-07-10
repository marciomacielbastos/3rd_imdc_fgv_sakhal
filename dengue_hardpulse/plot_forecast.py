from __future__ import annotations
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

def _observed_span_label(data: dict) -> str:
    span = data.get('observed_span')
    n_weeks = data.get('n_observed_weeks')
    if span and n_weeks is not None:
        return f'observed: {span} ({int(n_weeks)} weeks)'
    if span:
        return f'observed: {span}'
    return 'observed weeks only'

def _time_observed_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool).any(axis=0).reshape(-1)

def _shade_unobserved_time(ax, time_mask: np.ndarray) -> None:
    in_gap = False
    start = 0
    for idx, observed in enumerate(time_mask):
        if not observed and (not in_gap):
            start = idx
            in_gap = True
        elif observed and in_gap:
            ax.axvspan(start, idx, color='#eee', alpha=0.45, zorder=0)
            in_gap = False
    if in_gap:
        ax.axvspan(start, len(time_mask), color='#eee', alpha=0.45, zorder=0)

def compute_crps(fore: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    fore = np.asarray(fore, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    D = fore.shape[0]
    term1 = np.abs(fore - obs[None, :, :, :]).mean(axis=0)
    term2 = np.zeros_like(term1)
    for d in range(D):
        term2 += np.abs(fore[d] - fore).mean(axis=0)
    term2 = 0.5 * term2 / D
    crps = term1 - term2
    return crps[mask]

def compute_per_city_metrics(fore: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    fore = np.asarray(fore, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    C = fore.shape[1]
    median = np.median(fore, axis=0)
    lo = np.percentile(fore, 5, axis=0)
    hi = np.percentile(fore, 95, axis=0)
    rows = []
    for ci in range(C):
        m = mask[ci]
        if not m.any():
            rows.append({'city_idx': ci, 'n_obs': 0, 'mae': np.nan, 'rmse': np.nan, 'bias': np.nan, 'coverage_90': np.nan, 'crps': np.nan, 'total_obs': 0, 'total_pred': 0, 'inc_mae_100k': np.nan, 'inc_bias_100k': np.nan, 'total_inc_obs_100k': 0, 'total_inc_pred_100k': 0})
            continue
        y_true = obs[ci][m]
        y_pred = median[ci][m]
        y_lo = lo[ci][m]
        y_hi = hi[ci][m]
        crps_vals = np.abs(fore[:, ci, :, :] - obs[ci]).mean(axis=0)[m]
        pairs_sum = np.zeros(m.sum())
        D = fore.shape[0]
        for d in range(D):
            pairs_sum += np.abs(fore[d, ci][m] - fore[:, ci][np.ix_(range(D), *np.where(m))].mean(axis=0))
        crps_city = (crps_vals - 0.5 * pairs_sum / D).mean()
        rows.append({'city_idx': ci, 'n_obs': int(m.sum()), 'mae': float(np.abs(y_true - y_pred).mean()), 'rmse': float(np.sqrt(((y_true - y_pred) ** 2).mean())), 'bias': float((y_pred - y_true).mean()), 'coverage_90': float(((y_true >= y_lo) & (y_true <= y_hi)).mean()), 'crps': float(crps_city), 'total_obs': float(y_true.sum()), 'total_pred': float(y_pred.sum())})
    return pd.DataFrame(rows)

def print_skill_metrics(fore, test_obs, test_mask):
    fore = np.asarray(fore, dtype=float)
    obs = np.asarray(test_obs, dtype=float)
    mask = np.asarray(test_mask, dtype=bool)
    median = np.median(fore, axis=0)
    lo = np.percentile(fore, 5, axis=0)
    hi = np.percentile(fore, 95, axis=0)
    y_true = obs[mask]
    y_pred = median[mask]
    mae = float(np.abs(y_true - y_pred).mean())
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    cov90 = float(((y_true >= lo[mask]) & (y_true <= hi[mask])).mean())
    bias = float((y_pred - y_true).mean())
    bias_dir = 'over' if bias > 0 else 'under'
    crps_vals = np.abs(fore - obs[None]).mean(axis=0)[mask]
    crps_mean = float(crps_vals.mean())
    print(f'\n  Skill metrics (n={mask.sum():,} observed test weeks)')
    print(f'  MAE         : {mae:.1f}  cases/week')
    print(f'  RMSE        : {rmse:.1f}  cases/week')
    print(f'  90% coverage: {cov90 * 100:.1f}%  (ideal ~90%)')
    print(f'  Bias        : {bias:+.1f}  ({bias_dir}predicting)')
    print(f'  CRPS (mean) : {crps_mean:.1f}')

def _state_shape_diagnostics(median: np.ndarray, obs: np.ndarray, mask: np.ndarray, data_test: dict) -> pd.DataFrame:
    C = obs.shape[0]
    state_idx = np.asarray(data_test.get('state_idx', np.zeros(C, int)))
    geocodes = np.asarray(data_test.get('geocode_order', np.arange(C)), dtype=np.int64)
    uf_codes = geocodes // 100000 if geocodes.size == C else state_idx
    rows = []
    for state in sorted(np.unique(state_idx)):
        city_mask = state_idx == state
        time_mask = mask[city_mask].any(axis=0).reshape(-1)
        if not time_mask.any():
            continue
        obs_series = np.where(mask[city_mask], obs[city_mask], 0.0).sum(axis=0).reshape(-1)[time_mask]
        pred_series = np.where(mask[city_mask], median[city_mask], 0.0).sum(axis=0).reshape(-1)[time_mask]
        total_obs = float(obs_series.sum())
        total_pred = float(pred_series.sum())
        obs_peak = int(np.argmax(obs_series)) + 1 if obs_series.size else np.nan
        pred_peak = int(np.argmax(pred_series)) + 1 if pred_series.size else np.nan
        peak_error = pred_peak - obs_peak if not np.isnan(obs_peak) else np.nan
        corr = float(np.corrcoef(obs_series, pred_series)[0, 1]) if np.std(obs_series) > 0 and np.std(pred_series) > 0 else np.nan
        peak_ratio = float(pred_series[pred_peak - 1] / max(obs_series[obs_peak - 1], 1.0)) if not np.isnan(obs_peak) else np.nan
        uf_vals = uf_codes[city_mask]
        uf_code = int(pd.Series(uf_vals).mode().iloc[0]) if len(uf_vals) else int(state)
        rows.append({'state_idx': int(state), 'uf_code': uf_code, 'n_cities': int(city_mask.sum()), 'n_weeks': int(time_mask.sum()), 'total_obs': total_obs, 'total_pred': total_pred, 'volume_ratio': float(total_pred / max(total_obs, 1.0)), 'obs_peak_week': obs_peak, 'pred_peak_week': pred_peak, 'peak_week_error': peak_error, 'abs_peak_week_error': abs(peak_error) if not np.isnan(peak_error) else np.nan, 'shape_corr': corr, 'peak_value_ratio': peak_ratio})
    return pd.DataFrame(rows)

def evaluate_all_cities(fore: np.ndarray, Y_test: np.ndarray, mask_test: np.ndarray, data_test: dict, save_path: str | Path | None=None) -> pd.DataFrame:
    fore = np.asarray(fore, dtype=float)
    obs = np.asarray(Y_test, dtype=float)
    mask = np.asarray(mask_test, dtype=bool)
    C = fore.shape[1]
    median = np.median(fore, axis=0)
    lo = np.percentile(fore, 5, axis=0)
    hi = np.percentile(fore, 95, axis=0)
    geocodes = data_test.get('geocode_order', list(range(C)))
    state_idx = np.asarray(data_test.get('state_idx', np.zeros(C, int)))
    pop = np.asarray(data_test.get('population', np.ones_like(obs.reshape(C, -1))), dtype=float)
    pop = pop.reshape(obs.shape) if pop.shape != obs.shape else pop
    rows = []
    for ci in range(C):
        m = mask[ci]
        if not m.any():
            rows.append({'city_idx': ci, 'geocode': geocodes[ci], 'state_idx': int(state_idx[ci]), 'n_obs': 0, 'mae': np.nan, 'rmse': np.nan, 'bias': np.nan, 'coverage_90': np.nan, 'crps': np.nan, 'total_obs': 0, 'total_pred': 0, 'inc_mae_100k': np.nan, 'inc_bias_100k': np.nan, 'total_inc_obs_100k': 0, 'total_inc_pred_100k': 0})
            continue
        yt = obs[ci][m]
        yp = median[ci][m]
        yl = lo[ci][m]
        yh = hi[ci][m]
        pop_obs = np.maximum(pop[ci][m], 1.0)
        yt_inc = 100000.0 * yt / pop_obs
        yp_inc = 100000.0 * yp / pop_obs
        mean_pop_obs = float(np.maximum(pop_obs.mean(), 1.0))
        crps_val = float(np.abs(fore[:, ci, :, :][:, m] - obs[ci][m][None]).mean())
        rows.append({'city_idx': ci, 'geocode': geocodes[ci], 'state_idx': int(state_idx[ci]), 'n_obs': int(m.sum()), 'mae': float(np.abs(yt - yp).mean()), 'rmse': float(np.sqrt(((yt - yp) ** 2).mean())), 'bias': float((yp - yt).mean()), 'coverage_90': float(((yt >= yl) & (yt <= yh)).mean()), 'crps': crps_val, 'total_obs': float(yt.sum()), 'total_pred': float(yp.sum()), 'inc_mae_100k': float(np.abs(yt_inc - yp_inc).mean()), 'inc_bias_100k': float((yp_inc - yt_inc).mean()), 'total_inc_obs_100k': float(100000.0 * yt.sum() / mean_pop_obs), 'total_inc_pred_100k': float(100000.0 * yp.sum() / mean_pop_obs)})
    df = pd.DataFrame(rows)
    print(f'\n  All-city evaluation ({C:,} cities):')
    print(f'  Test window: {_observed_span_label(data_test)}')
    valid = df[df['n_obs'] > 0]
    print(f'  Cities with observations: {len(valid):,}')
    print(f"  Median MAE     : {valid['mae'].median():.1f}")
    print(f"  Median RMSE    : {valid['rmse'].median():.1f}")
    print(f"  Mean coverage  : {valid['coverage_90'].mean() * 100:.1f}%")
    print(f"  Mean CRPS      : {valid['crps'].mean():.1f}")
    print(f"  Bias (mean)    : {valid['bias'].mean():+.1f}")
    print(f"  Median inc MAE : {valid['inc_mae_100k'].median():.2f} per 100k")
    print(f"  Inc bias (mean): {valid['inc_bias_100k'].mean():+.2f} per 100k")
    state_diag = _state_shape_diagnostics(median, obs, mask, data_test)
    if not state_diag.empty:
        print('  State diagnostics:')
        print(f"    Median volume ratio : {state_diag['volume_ratio'].median():.3f}")
        print(f"    Median |peak error|: {state_diag['abs_peak_week_error'].median():.1f} weeks")
        print(f"    Median shape corr  : {state_diag['shape_corr'].median():.3f}")
    if save_path is not None:
        save_path = Path(save_path)
        df.to_csv(save_path, index=False)
        print(f'  Saved → {save_path}')
        if not state_diag.empty:
            state_path = save_path.with_name('state_diagnostics.csv')
            state_diag.to_csv(state_path, index=False)
            print(f'  Saved → {state_path}')
    return df

def print_state_summary(metrics_df: pd.DataFrame, data_test: dict) -> None:
    state_idx = np.asarray(data_test.get('state_idx', np.zeros(len(metrics_df), int)))
    df = metrics_df.copy()
    df['state'] = state_idx[df['city_idx'].values]
    valid = df[df['n_obs'] > 0]
    by_state = valid.groupby('state').agg(n_cities=('city_idx', 'count'), mae=('mae', 'mean'), coverage=('coverage_90', 'mean'), crps=('crps', 'mean'), total_obs=('total_obs', 'sum'), total_pred=('total_pred', 'sum')).reset_index()
    by_state['volume_ratio'] = by_state['total_pred'] / np.maximum(by_state['total_obs'], 1)
    by_state['bias_pct'] = (by_state['volume_ratio'] - 1) * 100
    print('\n  State-level summary (top 10 by total burden):')
    print(f"  {'State':>6}  {'Cities':>6}  {'MAE':>7}  {'Cov90%':>7}  {'CRPS':>7}  {'VolRat':>7}  {'Bias%':>7}")
    print('  ' + '-' * 62)
    top = by_state.nlargest(10, 'total_obs')
    for _, row in top.iterrows():
        print(f"  {int(row['state']):>6}  {int(row['n_cities']):>6}  {row['mae']:>7.1f}  {row['coverage'] * 100:>6.1f}%  {row['crps']:>7.1f}  {row['volume_ratio']:>7.2f}  {row['bias_pct']:>+7.1f}%")

def plot_forecast_vs_observed(fore, train_obs, test_obs, test_data, train_data, city_ids, geocode_order, n_cols=2, ci_low=5, ci_high=95, save_path='forecast_vs_observed.png', last_n_train_years=4):
    fore = np.asarray(fore)
    tr_obs = np.asarray(train_obs)
    te_obs = np.asarray(test_obs)
    mask = np.asarray(test_data['obs_mask'])
    time_mask = _time_observed_mask(mask)
    C = fore.shape[1]
    Y_tr = tr_obs.shape[1]
    T = fore.shape[3]
    train_years = train_data.get('year_order', [])
    test_years = test_data.get('year_order', [])
    n_ctx = min(last_n_train_years, Y_tr)
    ctx_st = Y_tr - n_ctx
    n_rows = int(np.ceil(C / n_cols))
    fig = plt.figure(figsize=(11 * n_cols, 3.5 * n_rows))
    outer = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.6, wspace=0.3)
    for i in range(C):
        row, col = divmod(i, n_cols)
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row, col], width_ratios=[n_ctx, fore.shape[2]], wspace=0.05)
        ax_tr = fig.add_subplot(inner[0])
        ax_fo = fig.add_subplot(inner[1], sharey=None)
        gc = geocode_order[city_ids[i]] if geocode_order else city_ids[i]
        ctx = tr_obs[i, ctx_st:, :].reshape(-1)
        x_ctx = np.arange(len(ctx))
        ax_tr.fill_between(x_ctx, 0, ctx, color='#aaa', alpha=0.5)
        ax_tr.plot(x_ctx, ctx, color='#777', lw=0.8)
        ax_tr.set_xticks([j * T for j in range(n_ctx)])
        ax_tr.set_xticklabels([str(y) for y in train_years[ctx_st:]], fontsize=7, rotation=45)
        ax_tr.set_ylabel('Cases/week', fontsize=8)
        ax_tr.set_title(f'Geocode {gc}', fontsize=9)
        ax_tr.set_xlim(0, len(ctx))
        ax_tr.set_ylim(bottom=0)
        ax_tr.grid(axis='y', lw=0.3, alpha=0.5)
        ax_tr.axvspan(len(ctx) - T * 0.3, len(ctx), color='#ddd', alpha=0.4, zorder=0)
        for sp in ['top', 'right']:
            ax_tr.spines[sp].set_visible(False)
        Y_te = fore.shape[2]
        ff = fore[:, i, :, :].reshape(fore.shape[0], -1)
        med = np.median(ff, axis=0)
        lo = np.percentile(ff, ci_low, axis=0)
        hi = np.percentile(ff, ci_high, axis=0)
        x_fo = np.arange(Y_te * T)
        _shade_unobserved_time(ax_fo, time_mask)
        ax_fo.fill_between(x_fo, lo, hi, color='steelblue', alpha=0.25, label=f'{ci_high - ci_low}% CI')
        ax_fo.plot(x_fo, med, color='steelblue', lw=1.8, label='Forecast median')
        te_f = te_obs[i].reshape(-1)
        mk_f = mask[i].reshape(-1).astype(bool)
        ax_fo.scatter(x_fo[mk_f], te_f[mk_f], color='darkorange', s=20, zorder=5, label='Observed', alpha=0.9)
        ax_fo.set_xticks([j * T for j in range(Y_te)])
        ax_fo.set_xticklabels([str(y) for y in test_years], fontsize=7, rotation=45)
        ax_fo.set_xlim(0, Y_te * T)
        ax_fo.set_ylim(bottom=0)
        ax_fo.set_title(_observed_span_label(test_data), fontsize=8)
        ax_fo.grid(axis='y', lw=0.3, alpha=0.5)
        ax_fo.tick_params(labelsize=7)
        ax_fo.yaxis.set_label_position('right')
        ax_fo.yaxis.tick_right()
        if i == 0:
            ax_fo.legend(fontsize=7, loc='upper right')
        for sp in ['top', 'left']:
            ax_fo.spines[sp].set_visible(False)
    for j in range(C, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        for sp in fig.get_axes():
            pass
    fig.suptitle(f'Forecast vs observed — {C} cities ({_observed_span_label(test_data)})', fontsize=11, y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {save_path}')

def plot_coverage_by_week(fore, test_obs, test_mask, test_years, save_path='coverage_by_week.png', ci_low=5, ci_high=95):
    fore = np.asarray(fore)
    te = np.asarray(test_obs)
    mask = np.asarray(test_mask).astype(bool)
    Y_te, T = (fore.shape[2], fore.shape[3])
    lo = np.percentile(fore, ci_low, axis=0)
    hi = np.percentile(fore, ci_high, axis=0)
    inside = (te >= lo) & (te <= hi)
    cov = np.full((Y_te, T), np.nan)
    for y in range(Y_te):
        for w in range(T):
            m = mask[:, y, w]
            if m.any():
                cov[y, w] = inside[:, y, w][m].mean()
    fig, axes = plt.subplots(1, Y_te, figsize=(8 * Y_te, 3), squeeze=False)
    for y in range(Y_te):
        ax = axes[0][y]
        wks = np.where(~np.isnan(cov[y]))[0] + 1
        c = cov[y][~np.isnan(cov[y])]
        ax.bar(wks, c * 100, color='steelblue', alpha=0.6, width=0.8)
        ax.axhline(ci_high - ci_low, color='red', lw=1.2, ls='--', label=f'{ci_high - ci_low}% target')
        ax.set_ylim(0, 105)
        ax.set_xlabel('Week', fontsize=9)
        ax.set_ylabel('Coverage%', fontsize=9)
        ax.set_title(f'CI coverage — {(test_years[y] if test_years else y)}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {save_path}')

def plot_top_cities(fore: np.ndarray, Y_test: np.ndarray, mask_test: np.ndarray, data_test: dict, metrics_df: pd.DataFrame, n_top: int=20, n_cols: int=4, save_path: str='top_cities_forecast.png') -> None:
    del mask_test
    valid = metrics_df[metrics_df['n_obs'] > 0]
    top = valid.nlargest(n_top, 'total_obs')
    city_ids_top = top['city_idx'].values.tolist()
    from .inference import build_subset_data
    sub, sub_obs = build_subset_data(data_test, jnp.array(Y_test), city_ids_top)
    fore_sub = fore[:, city_ids_top, :, :]
    geocodes = data_test.get('geocode_order', list(range(fore.shape[1])))
    plot_forecast_vs_observed(fore=fore_sub, train_obs=np.zeros_like(np.array(sub_obs)), test_obs=np.array(sub_obs), test_data=sub, train_data=sub, city_ids=city_ids_top, geocode_order=geocodes, n_cols=n_cols, last_n_train_years=0, save_path=save_path)
