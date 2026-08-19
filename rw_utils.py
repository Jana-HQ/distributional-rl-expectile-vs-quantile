"""
Shared utilities for the expectile-vs-quantile distributional RL project.

Two families of functions:
  - Decoding / population geometry (CCGP-style cross-condition decoding,
    signal-detection alignment). These operate on any array shaped
    (n_trial_types, n_units, n_trials, n_periods) — real neural data or
    simulated population activity alike.
  - Minimal Rescorla-Wagner-style learners with heterogeneous tau
    (expectile) or fixed-step (quantile) update rules, plus a generic
    multi-armed bandit runner for the task-generalization experiments.

Imported by 00_prepare_data.ipynb (real data) and by 01-03 (simulation
only, no dataset dependency).
"""

import numpy as np
from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

# ── Plot theme ────────────────────────────────────────────────────────────
TEAL, CORAL, PURPLE, GRAY, AMBER = '#1D9E75', '#D85A30', '#7F77DD', '#888780', '#BA7517'
RULE_COLORS = {'expectile': CORAL, 'quantile': AMBER}

# ── The two real task designs (six odor slots, two per distribution) ──────
NOTHING_IDX, FIXED_IDX, VARIABLE_IDX = [0, 1], [2, 3], [4, 5]
UNIFORM_IDX, BIMODAL_IDX = [2, 3], [4, 5]

CONTRASTS_RW = {
    'mean':     dict(task='SameRewDist', neg=NOTHING_IDX, pos=FIXED_IDX),
    'variance': dict(task='SameRewDist', neg=FIXED_IDX,   pos=VARIABLE_IDX),
    'kurtosis': dict(task='SameRewVar',  neg=UNIFORM_IDX, pos=BIMODAL_IDX),
}

# Default RW population size, shared between the noise-correlation
# calibration (00_prepare_data) and the RW population notebooks (01-02),
# so the calibrated Cholesky factor always matches the population it's
# applied to unless a notebook deliberately overrides it.
N_UNITS_RW_DEFAULT = 50


def get_dist_fns(task, rng):
    """odor_idx (0-5) -> fn(n) -> n reward samples, for the two real task designs."""
    if task == 'SameRewDist':
        return {0: lambda n: np.zeros(n), 1: lambda n: np.zeros(n),
                2: lambda n: np.full(n, 4.0), 3: lambda n: np.full(n, 4.0),
                4: lambda n: rng.choice([2.0, 6.0], size=n),
                5: lambda n: rng.choice([2.0, 6.0], size=n)}
    if task == 'SameRewVar':
        half = np.sqrt(12)

        def bimodal(n):
            return np.where(rng.random(n) < 0.5,
                             rng.normal(3, np.sqrt(3), n),
                             rng.normal(5, np.sqrt(3), n))

        return {0: lambda n: np.zeros(n), 1: lambda n: np.zeros(n),
                2: lambda n: rng.uniform(4 - half, 4 + half, size=n),
                3: lambda n: rng.uniform(4 - half, 4 + half, size=n),
                4: bimodal, 5: bimodal}
    raise ValueError(task)


# ── Population decoding / geometry helpers ────────────────────────────────
# CCGP: train a classifier on one odor pair's version of a contrast, test on
# the other odor pair. Above-chance accuracy means the contrast generalizes
# across identity (Bernardi et al. 2020; matches Lowet et al. Methods).

C_SVM = 5e-3
N_SAMPLE_TRIALS = 200


def make_clf():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='linear', C=C_SVM, class_weight='balanced', max_iter=10_000)),
    ])


def get_X(cr, cell_idx, tt, period):
    """Single-trial matrix (n_valid_trials, n_cells), NaN-imputed by column mean."""
    arr = cr[tt, cell_idx, :, period].T.astype(float)
    keep = ~np.isnan(arr).all(axis=1)
    arr = arr[keep]
    col_mean = np.nanmean(arr, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    nan_mask = np.isnan(arr)
    arr[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return arr


def build_Xy(cr, cell_idx, neg_tts, pos_tts, period):
    Xs, ys = [], []
    for tt in neg_tts:
        X = get_X(cr, cell_idx, tt, period)
        Xs.append(X); ys.extend([0] * len(X))
    for tt in pos_tts:
        X = get_X(cr, cell_idx, tt, period)
        Xs.append(X); ys.extend([1] * len(X))
    return np.vstack(Xs), np.array(ys)


def bootstrap_X(X, n_samples=N_SAMPLE_TRIALS, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(len(X), size=n_samples, replace=True)
    return X[idx]


def build_Xy_bootstrap(cr, cell_idx, neg_tts, pos_tts, period,
                        n_samples=N_SAMPLE_TRIALS, rng=None):
    Xs, ys = [], []
    for tt in neg_tts:
        X = bootstrap_X(get_X(cr, cell_idx, tt, period), n_samples, rng)
        Xs.append(X); ys.extend([0] * len(X))
    for tt in pos_tts:
        X = bootstrap_X(get_X(cr, cell_idx, tt, period), n_samples, rng)
        Xs.append(X); ys.extend([1] * len(X))
    return np.vstack(Xs), np.array(ys)


def cross_identity_decode(cr, cell_idx, neg_tts, pos_tts, period, n_cv=5, seed=0,
                           n_sample_trials=N_SAMPLE_TRIALS):
    """CCGP. Returns (cross-identity acc, within-identity acc, shuffle null) or None."""
    rng = np.random.default_rng(seed)
    pairings = [((neg_tts[0], pos_tts[0]), (neg_tts[1], pos_tts[1])),
                ((neg_tts[0], pos_tts[1]), (neg_tts[1], pos_tts[0]))]
    cross_accs = []
    for (negA, posA), (negB, posB) in pairings:
        ya_raw = build_Xy(cr, cell_idx, [negA], [posA], period)[1]
        yb_raw = build_Xy(cr, cell_idx, [negB], [posB], period)[1]
        if min(len(ya_raw), len(yb_raw)) < 8:
            continue
        if len(np.unique(ya_raw)) < 2 or len(np.unique(yb_raw)) < 2:
            continue
        Xa, ya = build_Xy_bootstrap(cr, cell_idx, [negA], [posA], period, n_sample_trials, rng)
        Xb, yb = build_Xy_bootstrap(cr, cell_idx, [negB], [posB], period, n_sample_trials, rng)
        for Xtr, ytr, Xte, yte in [(Xa, ya, Xb, yb), (Xb, yb, Xa, ya)]:
            clf = make_clf()
            clf.fit(Xtr, ytr)
            cross_accs.append(balanced_accuracy_score(yte, clf.predict(Xte)))
    if not cross_accs:
        return None
    cross = float(np.mean(cross_accs))

    Xa, ya = build_Xy_bootstrap(cr, cell_idx, [neg_tts[0]], [pos_tts[0]], period, n_sample_trials, rng)
    skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=seed)
    within = float(np.mean([
        balanced_accuracy_score(ya[te], make_clf().fit(Xa[tr], ya[tr]).predict(Xa[te]))
        for tr, te in skf.split(Xa, ya)]))

    rng_shuf = np.random.default_rng(seed + 777_000)
    Xa_s, ya_s = build_Xy_bootstrap(cr, cell_idx, [neg_tts[0]], [pos_tts[0]], period, n_sample_trials, rng_shuf)
    Xb_s, yb_s = build_Xy_bootstrap(cr, cell_idx, [neg_tts[1]], [pos_tts[1]], period, n_sample_trials, rng_shuf)
    clf_sh = make_clf()
    clf_sh.fit(Xa_s, rng_shuf.permutation(ya_s))
    shuf = float(balanced_accuracy_score(yb_s, clf_sh.predict(Xb_s)))
    return cross, within, shuf


def abstraction_index(cross, within):
    """AI = (CCGP - 0.5) / (within - 0.5). ~1 = fully abstract, ~0 = decodable but not abstract."""
    return (cross - 0.5) / (within - 0.5) if within > 0.5 else np.nan


def sdt_cross_session(cr, cell_idx, neg_tts, pos_tts, period):
    """Matched-filter signal-detection prediction of CCGP. Returns (predicted_cross_acc, cos_align)."""
    all_tts = neg_tts + pos_tts
    cond_mean = {tt: np.nanmean(cr[tt, cell_idx, :, period], axis=1) for tt in all_tts}
    valid = np.all([np.isfinite(cond_mean[tt]) for tt in all_tts], axis=0)
    if valid.sum() < 5:
        return np.nan, np.nan
    pairings = [((neg_tts[0], pos_tts[0]), (neg_tts[1], pos_tts[1])),
                ((neg_tts[0], pos_tts[1]), (neg_tts[1], pos_tts[0]))]
    cross_list, align_list = [], []
    for idA, idB in pairings:
        for (Ltr, Htr), (Lte, Hte) in [(idA, idB), (idB, idA)]:
            u_tr = (cond_mean[Htr] - cond_mean[Ltr])[valid]
            u_te = (cond_mean[Hte] - cond_mean[Lte])[valid]
            n_tr, n_te = np.linalg.norm(u_tr), np.linalg.norm(u_te)
            if n_tr < 1e-9 or n_te < 1e-9:
                continue
            align_list.append(float(u_tr @ u_te / (n_tr * n_te)))
            u_hat = u_tr / n_tr
            pL = cr[Lte, cell_idx, :, period][valid].T
            pH = cr[Hte, cell_idx, :, period][valid].T
            pL = pL[~np.isnan(pL).any(axis=1)] @ u_hat
            pH = pH[~np.isnan(pH).any(axis=1)] @ u_hat
            if len(pL) < 3 or len(pH) < 3:
                continue
            sd_pool = np.sqrt(0.5 * (pL.var(ddof=1) + pH.var(ddof=1)))
            if sd_pool < 1e-9:
                continue
            cross_list.append(float(norm.cdf(((pH.mean() - pL.mean()) / sd_pool) / 2.0)))
    if not cross_list or not align_list:
        return np.nan, np.nan
    return float(np.mean(cross_list)), float(np.mean(align_list))


# ── Minimal Rescorla-Wagner population (passive, cue-driven) ─────────────
# Each unit has an asymmetric learning-rate pair (alpha_plus, alpha_minus).
# tau = alpha_plus / (alpha_plus + alpha_minus) indexes the unit's effective
# expectile level: tau > 0.5 is optimistic (D1-like), tau < 0.5 pessimistic
# (D2-like).

def sample_alphas(n_units, base_rate=0.05, seed=0):
    rng = np.random.default_rng(seed)
    taus = rng.uniform(0.05, 0.95, n_units)
    return base_rate * 2 * taus, base_rate * 2 * (1 - taus), taus


def train_rw_population(task, alpha_plus, alpha_minus, n_trials=30_000, v_init=2.0, seed=0):
    """Run to convergence (no observation noise, no probing). Returns V: (6, n_units)."""
    rng = np.random.default_rng(seed)
    dist_fns = get_dist_fns(task, rng)
    V = np.full((6, len(alpha_plus)), v_init, dtype=float)
    for _ in range(n_trials):
        odor = rng.integers(0, 6)
        delta = dist_fns[odor](1)[0] - V[odor]
        pos = delta > 0
        V[odor, pos]  += alpha_plus[pos]  * delta[pos]
        V[odor, ~pos] += alpha_minus[~pos] * delta[~pos]
    return V


def train_rw_probed(task, alpha_plus, alpha_minus, rule='expectile',
                     n_trials_per_odor=40, n_probe=30, v_init=2.0,
                     learn_noise_sd=0.0, obs_noise_sd=5.0, L_obs=None, seed=0):
    """
    Realistic-trial-count regime: records value estimates (+ observation
    noise) for the last n_probe presentations of each odor.

    rule='expectile': update scales with error magnitude.
    rule='quantile':  fixed-step update in the sign of the error only.
    learn_noise_sd:   corrupts the reward signal used for the update,
                       modeling a noisy teaching/RPE signal.
    L_obs:             Cholesky factor for correlated observation noise at
                       probe time; if None, independent noise (obs_noise_sd).
    """
    rng = np.random.default_rng(seed)
    obs_rng = np.random.default_rng(seed + 999_999)
    dist_fns = get_dist_fns(task, rng)
    n_units = len(alpha_plus)
    V = np.full((6, n_units), v_init, dtype=float)
    counts = np.zeros(6, dtype=int)
    probe_V = np.full((6, n_units, n_probe), np.nan)
    order = np.concatenate([np.full(n_trials_per_odor, o) for o in range(6)])
    rng.shuffle(order)
    probe_start = n_trials_per_odor - n_probe
    for odor in order:
        r_true = dist_fns[odor](1)[0]
        r_learn = r_true + rng.normal(0, learn_noise_sd) if learn_noise_sd > 0 else r_true
        delta = r_learn - V[odor]
        pos = delta > 0
        if rule == 'expectile':
            V[odor, pos]  += alpha_plus[pos]  * delta[pos]
            V[odor, ~pos] += alpha_minus[~pos] * delta[~pos]
        elif rule == 'quantile':
            V[odor, pos]  += alpha_plus[pos]
            V[odor, ~pos] -= alpha_minus[~pos]
        else:
            raise ValueError(rule)
        counts[odor] += 1
        if counts[odor] > probe_start:
            probe_idx = counts[odor] - probe_start - 1
            if probe_idx >= n_probe:
                continue
            noise = (L_obs @ obs_rng.standard_normal(n_units) if L_obs is not None
                     else obs_rng.normal(0, obs_noise_sd, n_units))
            probe_V[odor, :, probe_idx] = V[odor] + noise
    return probe_V


def build_cr_from_probes(probe_V):
    """(6, n_units, n_probe) -> (6, n_units, n_probe, 1), the shape the
    decoding helpers above expect (trial_type, unit, trial, period)."""
    return probe_V[:, :, :, None]


def nothing_relative_distance_V(V, unit_idx):
    """d(Nothing, Variable) - d(Nothing, Fixed) in z-scored value space:
    positive = optimistic geometry, negative = pessimistic."""
    M = V[:, unit_idx]
    Mz = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-9)
    return np.linalg.norm(Mz[0] - Mz[4]) - np.linalg.norm(Mz[0] - Mz[2])


def train_linear_readout(V, task, lr=0.01, n_trials=10_000, seed=0):
    """Delta-rule readout of a scalar behavioral signal from the population."""
    rng = np.random.default_rng(seed + 500_000)
    dist_fns = get_dist_fns(task, rng)
    Vz = (V - V.mean(axis=0)) / (V.std(axis=0) + 1e-9)
    w = np.zeros(V.shape[1])
    for _ in range(n_trials):
        odor = rng.integers(0, 6)
        r = dist_fns[odor](1)[0]
        delta = r - (w @ Vz[odor])
        w += lr * delta * Vz[odor]
    return w, Vz


# ── Generic multi-armed bandit runner (active choice) ─────────────────────
# Shares the expectile/quantile tau-headed update rule above, but the agent
# chooses which arm to sample (epsilon-greedy on the mean estimate).

def tau_update(V, taus, arm, r_learn, rule, alpha):
    """In-place expectile/quantile update of V[arm] given a learning signal."""
    delta = r_learn - V[arm]
    pos = delta > 0
    if rule == 'expectile':
        V[arm, pos]  += alpha * taus[pos]        * delta[pos]
        V[arm, ~pos] += alpha * (1 - taus[~pos]) * delta[~pos]
    elif rule == 'quantile':
        V[arm, pos]  += alpha * taus[pos]
        V[arm, ~pos] -= alpha * (1 - taus[~pos])
    else:
        raise ValueError(rule)


def run_bandit(reward_fn, n_arms, n_trials, rule, alpha=0.1, epsilon=0.1,
               n_taus=10, learn_noise_sd=0.0, init_value=4.0, seed=0):
    """
    Epsilon-greedy multi-armed bandit with a tau-headed expectile/quantile
    learner. reward_fn(arm, t, rng) -> true reward for that arm at trial t
    (allows non-stationary or drifting reward distributions).

    Returns per-trial arrays: rewards, choices, and mean_est (the value
    estimate used to make each choice, i.e. before that trial's update —
    usable directly as a tracking-error signal against known true means).
    """
    rng = np.random.default_rng(seed)
    taus = np.linspace(0.05, 0.95, n_taus)
    V = np.full((n_arms, n_taus), init_value, dtype=float)
    rewards = np.zeros(n_trials)
    choices = np.zeros(n_trials, dtype=int)
    mean_est_hist = np.zeros((n_trials, n_arms))
    for t in range(n_trials):
        mean_est = V.mean(axis=1)
        mean_est_hist[t] = mean_est
        arm = rng.integers(0, n_arms) if rng.random() < epsilon else int(np.argmax(mean_est))
        r_true = reward_fn(arm, t, rng)
        r_learn = r_true + rng.normal(0, learn_noise_sd) if learn_noise_sd > 0 else r_true
        tau_update(V, taus, arm, r_learn, rule, alpha)
        rewards[t] = r_true
        choices[t] = arm
    return dict(rewards=rewards, choices=choices, mean_est=mean_est_hist)
