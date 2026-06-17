import numpy as np
from abc import ABC, abstractmethod
from numbers import Number
from .addm_helpers import (
    _build_alternating_mu_array,
    _build_addm_mu_array_data,
)
from .validation import check_multistage_params
from .validation import check_addm_params, check_single_stage_params
from ._defaults import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LAST_QUAD_ORDER,
    DEFAULT_MID_QUAD_ORDER,
    DEFAULT_TRUNC_NUM,
    DEFAULT_THRESHOLD,
)
from .utils import resolve_quadrature_orders


def _initialize_x0(x0, lower_bdy_t0, upper_bdy_t0, num, rng=None):
    """Shared helper for initializing starting positions.

    Used by both :class:`DDModel` and :class:`aDDModel`.
    """
    rng = (
        np.random.default_rng()
        if rng is None
        else (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        )
    )
    if isinstance(x0, Number):
        return x0 * np.ones(num)
    elif isinstance(x0, dict):
        dist_name = x0.get("dist_name")
        if dist_name == "uniform":
            return rng.uniform(lower_bdy_t0, upper_bdy_t0, num)
        elif dist_name == "beta":
            alpha = x0.get("alpha")
            beta = x0.get("beta")
            if alpha is None or beta is None:
                raise ValueError(
                    "Missing alpha and beta parameters for beta distribution"
                )
            return (
                rng.beta(alpha, beta, num) * (upper_bdy_t0 - lower_bdy_t0)
                + lower_bdy_t0
            )
        else:
            raise ValueError("Unsupported distribution type")
    else:
        raise ValueError("Invalid initial condition format")


class DDModel(ABC):
    """
    Abstract base class for drift-diffusion models of homogeneous trials.
    Defines a template for subclasses to implement specific drift and boundary behaviors.
    """

    def __init__(self, x0):
        self.x0 = x0

    def initialize_X0(self, t0, num, rng=None):
        """Initialize the initial condition X0."""
        return _initialize_x0(
            self.x0, self.lower_bdy(t0), self.upper_bdy(t0), num, rng=rng
        )

    @abstractmethod
    def drift_coeff(self, X: float, t: float) -> float:
        """
        Abstract method to determine the drift coefficient.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def diffusion_coeff(self, X: float, t: float) -> float:
        """
        Abstract method to determine the diffusion coefficient.
        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def is_update_vectorizable(self) -> bool:
        """
        Abstract property to determine whether the updates of the scheme can be vectorized.
        If self.drift_coeff and self.diffusion_coeff are all only functions of time, then the computation can be vectorized.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def upper_bdy(self, t: float) -> float:
        """
        Abstract method to define the upper boundary as a function of time.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def lower_bdy(self, t: float) -> float:
        """
        Abstract method to define the lower boundary as a function of time.
        Must be implemented by subclasses.
        """
        pass

    def simulate_trajs(self, T, Nt=1000, num=1000, rng=None):
        """Simulate multiple trajectories of the drift-diffusion model.

        Intended for visualisation — returns full sample paths.

        Parameters
        ----------
        T : float
            Duration of the simulation.
        Nt : int
            Number of time steps.
        num : int
            Number of independent trajectories.
        rng : int, np.random.Generator, or None
            Seed or RNG for reproducibility.

        Returns
        -------
        t_grid : ndarray (Nt+1,)
        X_grids : ndarray (num, Nt+1)
        """
        rng = np.random.default_rng(rng)
        t0 = 0
        Nt = int(Nt)
        dt = float(T - t0) / Nt
        t_grid = np.linspace(t0, T, Nt + 1)

        X0 = self.initialize_X0(t0, num, rng=rng)
        X_grids = np.zeros((num, Nt + 1))
        X_grids[:, 0] = X0
        dW = np.sqrt(dt) * rng.standard_normal((num, Nt))
        # Euler-Maruyama scheme
        if not self.is_update_vectorizable:
            for i in range(Nt):
                drift = self.drift_coeff(X_grids[:, i], t_grid[i])
                diffusion = self.diffusion_coeff(X_grids[:, i], t_grid[i])
                dX = drift * dt + diffusion * dW[:, i]
                X_grids[:, i + 1] = X_grids[:, i] + dX
        else:
            drift_vals = self.drift_coeff(0, t_grid[:-1]) * dt
            diffusion_vals = self.diffusion_coeff(0, t_grid[:-1]) * dW
            dX = drift_vals + diffusion_vals
            X_grids[:, 1:] = X_grids[:, 0].reshape(num, 1) + np.cumsum(dX, axis=1)

        return t_grid, X_grids

    def simulate_fpt(self, num, T, dt=0.001, rng=None, n_threads=1):
        """Simulate first passage times for *num* independent trials.

        When ``is_update_vectorizable`` is True (drift and diffusion depend
        only on *t*, not on *X*), a fast Cython inner loop is used
        automatically.  Otherwise falls back to a pure-Python implementation.

        Parameters
        ----------
        num : int
            Number of independent trials.
        T : float
            Maximum trial duration.
        dt : float
            Euler-Maruyama time step.
        rng : int, np.random.Generator, or None
            Seed or RNG for reproducibility.
        n_threads : int
            Number of OpenMP threads for the Cython path (1 = serial).

        Returns
        -------
        rt : ndarray (num,) float64
            Reaction times.  -1.0 if the trial did not terminate by *T*.
        choice : ndarray (num,) int32
            +1 (upper boundary), -1 (lower boundary), or 0 (no crossing).
        x_final : ndarray (num,) float64
            Final particle position (at crossing or at the last step).
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        rng = np.random.default_rng(rng)

        if self.is_update_vectorizable:
            return self._simulate_fpt_cython(num, T, dt, rng, n_threads)
        else:
            return self._simulate_fpt_python(num, T, dt, rng)

    def _simulate_fpt_cython(self, num, T, dt, rng, n_threads=1):
        """Cython fast path for models with time-only drift/diffusion."""
        from .cython.simulator import simulate_homog_ddm_fpt

        x0_data = np.ascontiguousarray(
            self.initialize_X0(0.0, num, rng=rng), dtype=np.float64
        )
        if T <= 0:
            rt = np.full(num, -1.0, dtype=np.float64)
            choice = np.zeros(num, dtype=np.int32)
            return rt, choice, x0_data

        max_steps = int(np.ceil(T / dt))
        t_end = np.minimum((np.arange(max_steps, dtype=np.float64) + 1.0) * dt, T)
        t_start = np.empty_like(t_end)
        t_start[0] = 0.0
        t_start[1:] = t_end[:-1]

        # Ensure (max_steps,) even when drift_coeff/diffusion_coeff return scalars
        drift_raw = self.drift_coeff(0, t_start)
        drift_vals = (
            np.full(max_steps, drift_raw, dtype=np.float64)
            if np.ndim(drift_raw) == 0
            else np.ascontiguousarray(drift_raw, dtype=np.float64)
        )
        diff_raw = self.diffusion_coeff(0, t_start)
        diffusion_vals = (
            np.full(max_steps, diff_raw, dtype=np.float64)
            if np.ndim(diff_raw) == 0
            else np.ascontiguousarray(diff_raw, dtype=np.float64)
        )
        upper_vals = np.ascontiguousarray(self.upper_bdy(t_end), dtype=np.float64)
        lower_vals = np.ascontiguousarray(self.lower_bdy(t_end), dtype=np.float64)

        trial_seeds = np.ascontiguousarray(
            rng.integers(0, 2**63, size=num, dtype=np.uint64)
        )

        return simulate_homog_ddm_fpt(
            drift_vals,
            diffusion_vals,
            upper_vals,
            lower_vals,
            x0_data,
            dt,
            max_steps,
            T,
            trial_seeds,
            n_threads,
        )

    def _simulate_fpt_python(self, num, T, dt, rng):
        """Pure-Python fallback for models where drift depends on X."""
        t = 0.0
        X = self.initialize_X0(t, num, rng=rng)
        active_idx = np.arange(num)

        rt = np.full(num, -1.0)
        choice = np.zeros(num, dtype=np.int32)
        x_final = np.zeros(num, dtype=np.float64)

        while t < T and active_idx.size > 0:
            dt_curr = min(dt, T - t)

            X_prev = X[active_idx]
            dW = rng.standard_normal(active_idx.size) * np.sqrt(dt_curr)
            drift = self.drift_coeff(X_prev, t)
            diffusion = self.diffusion_coeff(X_prev, t)
            X_new = X_prev + drift * dt_curr + diffusion * dW

            upper_bound = self.upper_bdy(t + dt_curr)
            lower_bound = self.lower_bdy(t + dt_curr)
            hit_upper = X_new >= upper_bound
            hit_lower = X_new <= lower_bound

            crossing_time = t + 0.5 * dt_curr
            if np.any(hit_upper):
                idx_up = active_idx[hit_upper]
                rt[idx_up] = crossing_time
                choice[idx_up] = 1
                x_final[idx_up] = X_new[hit_upper]
            if np.any(hit_lower):
                idx_lo = active_idx[hit_lower]
                rt[idx_lo] = crossing_time
                choice[idx_lo] = -1
                x_final[idx_lo] = X_new[hit_lower]

            X[active_idx] = X_new
            active_idx = active_idx[~(hit_upper | hit_lower)]

            t += dt_curr

        # Store final positions for non-exited trials
        x_final[active_idx] = X[active_idx]

        return rt, choice, x_final


class SingleStageModel(DDModel):
    """
    Subclass for the "angle" model with a constant drift and symmetric linear collapsing boundaries.
    """

    def __init__(self, mu, sigma, a, b, x0):
        super().__init__(x0)
        check_single_stage_params(mu, sigma, a, b, x0)
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b

    def drift_coeff(self, X: float, t: float) -> float:
        return self.mu

    def diffusion_coeff(self, X: float, t: float) -> float:
        return self.sigma

    @property
    def is_update_vectorizable(self) -> bool:
        return True

    def upper_bdy(self, t):
        return self.a - self.b * t

    def lower_bdy(self, t):
        return -self.a + self.b * t


class MultiStageModel(DDModel):
    """
    Subclass for multi-stage model of homogeneous trials,
    where the drift and diffusion are piecewise constant, and the boundaries are piecewise linear.
    The j-th (0<=j<=d-2) stage corresponds to node_array[j] <= t <= node_array[j+1],
    The (d-1)-th (last) stage corresponds to t >= node_array[d-1].
    In the j-th (0<=j<=d-1) stage,
    the model has drift mu_array[j] and diffusion sigma_array[j],
    the upper boundary has intercept ub[j] and slope b1_array[j].
    the lower boundary has intercept lb[j] and slope b2_array[j].

    mu_array, node_array, sigma_array, b1_array, b2_array MUST have the same length d.
    """

    def __init__(
        self, mu_array, node_array, sigma_array, a1, b1_array, a2, b2_array, x0
    ):
        super().__init__(x0)
        check_multistage_params(
            mu_array,
            node_array,
            sigma_array,
            a1,
            b1_array,
            a2,
            b2_array,
            x0=x0,
            allow_infinite_boundaries=True,
        )
        d = len(mu_array)
        self.d = d
        self.mu_array = mu_array
        self.node_array = node_array
        self.sigma_array = sigma_array
        self.a1 = a1
        self.b1_array = b1_array
        self.a2 = a2
        self.b2_array = b2_array
        ub = a1
        lb = a2
        self.ub_array = np.zeros((d,))
        self.lb_array = np.zeros((d,))
        self.ub_array[0] = ub
        self.lb_array[0] = lb
        for i in range(1, d):
            ub += b1_array[i - 1] * (node_array[i] - node_array[i - 1])
            lb += b2_array[i - 1] * (node_array[i] - node_array[i - 1])
            self.ub_array[i] = ub
            self.lb_array[i] = lb

    def drift_coeff(self, X: float, t: float) -> float:
        return piecewise_const_func(t, self.mu_array, self.node_array)

    def diffusion_coeff(self, X: float, t: float) -> float:
        return piecewise_const_func(t, self.sigma_array, self.node_array)

    @property
    def is_update_vectorizable(self) -> bool:
        return True

    def upper_bdy(self, t):
        return piecewise_linear_func(t, self.ub_array, self.b1_array, self.node_array)

    def lower_bdy(self, t):
        return piecewise_linear_func(t, self.lb_array, self.b2_array, self.node_array)


class aDDModel:
    """Attentional Drift-Diffusion Model of heterogeneous trials.

    Stores all model parameters that are shared across trials.  Per-trial
    covariates (drift rates, saccade times, stage counts) are passed to
    simulation and likelihood methods as arguments.

    Parameters
    ----------
    eta : float
        Attentional discount factor.
    kappa : float
        Drift-rate scaling.
    sigma : float
        Diffusion coefficient (constant across stages).
    a : float
        Boundary intercept (half-width at t=0).
    b : float
        Boundary collapse slope (>= 0).
    x0 : float
        Starting point of the evidence accumulator.
    """

    def __init__(self, eta, kappa, sigma, a, b, x0=0.0):
        check_addm_params(eta, kappa, sigma, a, b, x0)
        self.eta = eta
        self.kappa = kappa
        self.sigma = sigma
        self.a = a
        self.b = b
        self.x0 = x0

    # -- DDM-like interface (shared across all trials) ----------------------

    def diffusion_coeff(self, X, t):
        """Constant diffusion coefficient."""
        return self.sigma

    def upper_bdy(self, t):
        """Symmetric linear collapsing upper boundary."""
        return self.a - self.b * t

    def lower_bdy(self, t):
        """Symmetric linear collapsing lower boundary."""
        return -self.a + self.b * t

    def initialize_X0(self, t0, num, rng=None):
        """Initialize starting positions (delegates to shared helper)."""
        return _initialize_x0(
            self.x0, self.lower_bdy(t0), self.upper_bdy(t0), num, rng=rng
        )

    # -- aDDM-specific methods ----------------------------------------------

    def derive_drift(self, r1, r2):
        """Compute trial-level drift rates from stimulus ratings.

        Parameters
        ----------
        r1, r2 : float or array-like
            Stimulus ratings for item 1 and item 2.

        Returns
        -------
        mu1, mu2 : same shape as *r1*
            Drift rates for fixating item 1 and item 2 respectively.
        Note: whether the participant fixates item 1 first or item 2 first is described by the flag_data.
        """
        r1 = np.asarray(r1, dtype=np.float64)
        r2 = np.asarray(r2, dtype=np.float64)
        mu1 = self.kappa * (r1 - self.eta * r2)
        mu2 = self.kappa * (self.eta * r1 - r2)
        return mu1, mu2

    def simulate_fpt(
        self,
        r1_data,
        r2_data,
        flag_data,
        sacc_array_data,
        d_data,
        T,
        dt=1e-4,
        rng=None,
        chunk_size=DEFAULT_CHUNK_SIZE,
        n_threads=1,
    ):
        """Simulate first-passage times for heterogeneous aDDM trials.

        Derives drift rates from stimulus ratings internally using
        ``self.eta`` and ``self.kappa``, then swaps mu1/mu2 per trial
        according to *flag_data*.

        Parameters
        ----------
        r1_data, r2_data : ndarray (n_trials,)
            Stimulus ratings for item 1 and item 2.
        flag_data : ndarray (n_trials,) int
            0 = fixate item 1 first, 1 = fixate item 2 first.
        sacc_array_data : ndarray (n_trials, max_d)
            Saccade onset times per stage, zero-padded.
        d_data : ndarray (n_trials,) int32
            Number of fixation stages per trial.
        T : float
            Maximum trial duration.
        dt : float
            Euler-Maruyama time step.
        rng : int, np.random.Generator, or None
            Seed or RNG for reproducibility.
        chunk_size : int
            Trials per Cython call (controls peak memory).
        n_threads : int
            Number of OpenMP threads (1 = serial).

        Returns
        -------
        rt : ndarray (n_trials,) float64
            Reaction times (-1.0 if trial did not terminate by *T*).
        choice : ndarray (n_trials,) int32
            +1 (upper) or -1 (lower).  0 if no crossing.
        x_final : ndarray (n_trials,) float64
            Final particle position at the time of crossing or at the last step.
        """
        from .cython.simulator import _simulate_addm_fpt

        if dt <= 0:
            raise ValueError("dt must be positive")
        rng = np.random.default_rng(rng)
        n_trials = len(d_data)
        max_d = sacc_array_data.shape[1]

        # Build mu array from ADDM covariates
        mu_array_data = _build_addm_mu_array_data(
            self.eta,
            self.kappa,
            r1_data,
            r2_data,
            flag_data,
            d_data,
            max_d,
        )

        sacc_array_data = np.ascontiguousarray(sacc_array_data, dtype=np.float64)
        d_data = np.ascontiguousarray(d_data, dtype=np.int32)

        # Sample per-trial x0 (supports scalar and distribution x0)
        x0_data = np.ascontiguousarray(
            self.initialize_X0(0.0, n_trials, rng=rng), dtype=np.float64
        )

        budget_time = min(self.a / self.b, T) if self.b > 0 else T
        if budget_time <= 0:
            rt_all = np.full(n_trials, -1.0, dtype=np.float64)
            choice_all = np.zeros(n_trials, dtype=np.int32)
            return rt_all, choice_all, x0_data

        # Generate all seeds up front for reproducibility across chunk sizes
        trial_seeds = np.ascontiguousarray(
            rng.integers(0, 2**63, size=n_trials, dtype=np.uint64)
        )

        rt_all = np.empty(n_trials, dtype=np.float64)
        choice_all = np.empty(n_trials, dtype=np.int32)
        x_final_all = np.empty(n_trials, dtype=np.float64)

        for start in range(0, n_trials, chunk_size):
            end = min(start + chunk_size, n_trials)

            rt_c, ch_c, xf_c = _simulate_addm_fpt(
                np.ascontiguousarray(mu_array_data[start:end]),
                np.ascontiguousarray(sacc_array_data[start:end]),
                np.ascontiguousarray(d_data[start:end]),
                self.sigma,
                self.a,
                self.b,
                np.ascontiguousarray(x0_data[start:end]),
                dt,
                budget_time,
                np.ascontiguousarray(trial_seeds[start:end]),
                n_threads,
            )

            rt_all[start:end] = rt_c
            choice_all[start:end] = ch_c
            x_final_all[start:end] = xf_c

        return rt_all, choice_all, x_final_all

    def log_fptd(
        self,
        rt,
        choice,
        r1,
        r2,
        flag,
        sacc_array,
        d,
        order_mid=DEFAULT_MID_QUAD_ORDER,
        order_last=DEFAULT_LAST_QUAD_ORDER,
        order=None,
        trunc_num=DEFAULT_TRUNC_NUM,
        threshold=DEFAULT_THRESHOLD,
        log_space=False,
    ):
        """Log first-passage time density for a single trial configuration.

        Parameters
        ----------
        rt : float
            Time at which to evaluate the density.
        choice : int
            +1 for the upper boundary, -1 for the lower boundary.
        r1, r2 : float
            Stimulus ratings for the two options.
        flag : int
            0 = fixate item 1 first, 1 = fixate item 2 first.
        sacc_array : array-like, shape (d,)
            Saccade onset times.
        d : int
            Number of valid stages.
        order_mid, order_last : int
            Intermediate and final-stage Gauss-Legendre quadrature orders.

        Returns
        -------
        float
            Log FPTD value at time *rt*.
        """
        from .cython.multi_stage import compute_addm_logfptd

        sacc_array = np.asarray(sacc_array, dtype=np.float64)
        order_mid, order_last = resolve_quadrature_orders(
            order_mid=order_mid,
            order_last=order_last,
            order=order,
        )
        return compute_addm_logfptd(
            rt,
            choice,
            self.eta,
            self.kappa,
            self.sigma,
            self.a,
            self.b,
            self.x0,
            r1,
            r2,
            flag,
            sacc_array,
            d,
            order_mid=order_mid,
            order_last=order_last,
            trunc_num=trunc_num,
            threshold=threshold,
            log_space=log_space,
        )

    def mean_neg_log_likelihood(
        self,
        rt_data,
        choice_data,
        r1_data,
        r2_data,
        flag_data,
        sacc_array_data,
        d_data,
        order_mid=DEFAULT_MID_QUAD_ORDER,
        order_last=DEFAULT_LAST_QUAD_ORDER,
        order=None,
        trunc_num=DEFAULT_TRUNC_NUM,
        threshold=DEFAULT_THRESHOLD,
        n_threads=1,
        log_space=False,
        invalid_policy="inf",
        warn=True,
    ):
        """Mean negative log-likelihood over observed trials.

        Derives drift rates from stimulus ratings internally using
        ``self.eta`` and ``self.kappa``, then swaps mu1/mu2 per trial
        according to *flag_data* -- consistent with :meth:`simulate_fpt`.

        Parameters
        ----------
        rt_data : ndarray (n_trials,)
        choice_data : ndarray (n_trials,) int
        r1_data, r2_data : ndarray (n_trials,)
            Stimulus ratings for item 1 and item 2.
        flag_data : ndarray (n_trials,) int
            0 = fixate item 1 first, 1 = fixate item 2 first.
        sacc_array_data : ndarray (n_trials, max_d)
        d_data : ndarray (n_trials,) int
        order_mid, order_last : int
            Intermediate and final-stage Gauss-Legendre quadrature orders.
        trunc_num : int
            Series truncation limit.
        threshold : float
            Early-stopping threshold for series terms.
        n_threads : int
            -1 for all available threads.
        log_space : bool
            If True, compute likelihoods in log space internally before
            returning the mean negative log-likelihood.
        invalid_policy : {"inf", "warn"}
            ``"inf"`` propagates bad trials to ``+inf`` or ``NaN``.
            ``"warn"`` warns and skips them.
        warn : bool
            If True, emit warnings for skipped trials with invalid or zero
            likelihoods.

        Returns
        -------
        float
            Mean negative log-likelihood.
        """
        from .cython.batch import compute_addm_nll

        order_mid, order_last = resolve_quadrature_orders(
            order_mid=order_mid,
            order_last=order_last,
            order=order,
        )
        return compute_addm_nll(
            rt_data,
            choice_data,
            self.eta,
            self.kappa,
            self.sigma,
            self.a,
            self.b,
            self.x0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            order_mid=order_mid,
            order_last=order_last,
            trunc_num=trunc_num,
            threshold=threshold,
            n_threads=n_threads,
            log_space=log_space,
            invalid_policy=invalid_policy,
            reduce="mean",
            warn=warn,
        )

    def to_multistage_model(self, mu1, mu2, node_array):
        """Construct a :class:`MultiStageModel` for a single trial.

        Useful for visualisation (``simulate_trajs``), computing densities
        via the Python path, or any use case requiring the full ``DDModel``
        interface (``drift_coeff``, ``upper_bdy``, etc.).

        Parameters
        ----------
        mu1, mu2 : float
            Drift rates for item-1 and item-2 fixation stages.
        node_array : array-like
            Stage onset times.  ``node_array[0]`` should be 0.

        Returns
        -------
        MultiStageModel
        """
        node_array = np.asarray(node_array, dtype=np.float64)
        d = len(node_array)
        mu_array = _build_alternating_mu_array(mu1, mu2, d)
        sigma_array = np.full(d, self.sigma)
        b1_array = np.full(d, -self.b)
        b2_array = np.full(d, self.b)
        return MultiStageModel(
            mu_array,
            node_array,
            sigma_array,
            self.a,
            b1_array,
            -self.a,
            b2_array,
            self.x0,
        )

    def generate_experiment(
        self,
        n_trials,
        gamma_shape=1.0,
        gamma_scale=0.3,
        r_range=(1, 6),
        dt=1e-4,
        T=20.0,
        n_threads=1,
        rng=None,
        chunk_size=None,
    ):
        """Simulate *n_trials* of the attentional drift diffusion model.

        Generates random stimulus ratings, fixation sequences, and first-item
        flags, then runs the Cython Euler-Maruyama simulator.

        Parameters
        ----------
        n_trials : int
            Number of trials to simulate.
        gamma_shape, gamma_scale : float
            Parameters of the Gamma distribution for fixation durations.
        r_range : (int, int)
            Inclusive range for stimulus ratings (drawn uniformly).
        dt : float
            Euler-Maruyama time step.
        T : float
            Maximum trial duration (seconds).
        n_threads : int
            Number of OpenMP threads for the Cython simulator (1 = serial).
        rng : int, np.random.Generator, or None
            Seed or Generator for reproducibility.
        chunk_size : int or None
            Trials processed per Cython call (controls peak memory).

        Returns
        -------
        dict
            Grouped canonical aDDM experiment payload with the same schema used
            by :func:`efpt.io.save_addm_experiment` when persisted to
            the canonical `.npz` experiment archive:

            - ``decision_data``: ``rt_data``, ``choice_data``
            - ``params``: model parameters shared across trials
            - ``covariates``: per-trial stimulus/fixation inputs
            - ``config``: simulation/generation settings
        """
        from .addm_helpers import _generate_sacc_array_data

        if chunk_size is None:
            chunk_size = DEFAULT_CHUNK_SIZE

        rng = np.random.default_rng(rng)

        # --- Stimulus values ---
        r1_data = rng.integers(r_range[0], r_range[1] + 1, size=n_trials)
        r2_data = rng.integers(r_range[0], r_range[1] + 1, size=n_trials)

        # --- Fixation flag (which item first) ---
        flag_data = rng.integers(0, 2, size=n_trials).astype(np.int32)

        # --- Fixation sequences ---
        sacc_array_data, d_data, max_d = _generate_sacc_array_data(
            rng, n_trials, T, gamma_shape, gamma_scale
        )

        # --- Simulate ---
        rt_all, choice_all, _ = self.simulate_fpt(
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            T=T,
            dt=dt,
            rng=rng,
            chunk_size=chunk_size,
            n_threads=n_threads,
        )

        # --- Post-process: truncate to stages that started before RT ---
        terminated = rt_all > 0
        rt_col = rt_all[:, np.newaxis]
        stage_indices = np.arange(max_d)[np.newaxis, :]

        active_before_rt = (stage_indices < d_data[:, np.newaxis]) & (
            sacc_array_data < rt_col
        )
        d_new = active_before_rt.sum(axis=1).astype(np.int32)
        d_data = np.where(terminated, np.maximum(d_new, 1), d_data).astype(np.int32)

        beyond_new_d = stage_indices >= d_data[:, np.newaxis]
        sacc_array_data[beyond_new_d] = 0.0
        max_d = int(d_data.max())
        sacc_array_data = sacc_array_data[:, :max_d]

        return {
            "decision_data": {
                "rt_data": np.ascontiguousarray(rt_all, dtype=np.float64),
                "choice_data": np.ascontiguousarray(choice_all, dtype=np.int32),
            },
            "params": {
                "eta": self.eta,
                "kappa": self.kappa,
                "sigma": self.sigma,
                "a": self.a,
                "b": self.b,
                "x0": self.x0,
            },
            "covariates": {
                "r1_data": np.ascontiguousarray(r1_data),
                "r2_data": np.ascontiguousarray(r2_data),
                "flag_data": np.ascontiguousarray(flag_data, dtype=np.int32),
                "sacc_array_data": np.ascontiguousarray(
                    sacc_array_data, dtype=np.float64
                ),
                "d_data": np.ascontiguousarray(d_data, dtype=np.int32),
            },
            "config": {
                "dt": dt,
                "T": T,
                "gamma_shape": gamma_shape,
                "gamma_scale": gamma_scale,
                "r_range": r_range,
            },
        }

    @property
    def boundary_collapsing_time(self):
        """Time at which collapsing boundaries meet (``inf`` if *b* == 0)."""
        return self.a / self.b if self.b > 0 else float("inf")


# Backward-compatible re-exports from boundaries module
from .boundaries import (
    piecewise_const_func,
    piecewise_linear_func,
    weibull_survival,
)  # noqa: F401
