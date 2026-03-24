import numpy as np
from abc import ABC, abstractmethod
from numbers import Number
from .utils import check_valid_multistage_params, get_alternating_mu_array, build_mu_array_data


_DEFAULT_CHUNK_SIZE = 200


def _swap_and_build_mu(mu1_data, mu2_data, flag_data, d_data, max_d):
    """Swap mu1/mu2 based on fixation flag and build the padded drift array."""
    mu1_eff = np.where(flag_data == 0, mu1_data, mu2_data).astype(np.float64)
    mu2_eff = np.where(flag_data == 0, mu2_data, mu1_data).astype(np.float64)
    return build_mu_array_data(mu1_eff, mu2_eff, d_data, max_d)


def _initialize_x0(x0, lower_bdy_t0, upper_bdy_t0, num, rng=None):
    """Shared helper for initializing starting positions.

    Used by both :class:`DDModel` and :class:`aDDModel`.
    """
    rng = np.random.default_rng() if rng is None else (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
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

        Intended for visualisation — returns full sample paths, not just
        first-passage information.

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
        from .simulator_cy import simulate_homog_ddm_fpt_cy

        x0_data = np.ascontiguousarray(self.initialize_X0(0.0, num, rng=rng), dtype=np.float64)
        if T <= 0:
            rt = np.full(num, -1.0, dtype=np.float64)
            choice = np.zeros(num, dtype=np.int32)
            return rt, choice, x0_data

        max_steps = int(np.ceil(T / dt))
        t_end = np.minimum(
            (np.arange(max_steps, dtype=np.float64) + 1.0) * dt, T
        )
        t_start = np.empty_like(t_end)
        t_start[0] = 0.0
        t_start[1:] = t_end[:-1]

        # Ensure (max_steps,) even when drift_coeff/diffusion_coeff return scalars
        drift_raw = self.drift_coeff(0, t_start)
        drift_vals = np.full(max_steps, drift_raw, dtype=np.float64) if np.ndim(drift_raw) == 0 else np.ascontiguousarray(drift_raw, dtype=np.float64)
        diff_raw = self.diffusion_coeff(0, t_start)
        diffusion_vals = np.full(max_steps, diff_raw, dtype=np.float64) if np.ndim(diff_raw) == 0 else np.ascontiguousarray(diff_raw, dtype=np.float64)
        upper_vals = np.ascontiguousarray(self.upper_bdy(t_end), dtype=np.float64)
        lower_vals = np.ascontiguousarray(self.lower_bdy(t_end), dtype=np.float64)

        trial_seeds = np.ascontiguousarray(
            rng.integers(0, 2**63, size=num, dtype=np.uint64)
        )

        return simulate_homog_ddm_fpt_cy(
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
    Subclass for the angle model with a constant drift and symmetric linear collapsing boundaries.
    """

    def __init__(self, mu, sigma, a, b, x0):
        super().__init__(x0)
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
    The j-th (0<=j<=d-2) stage corresponds to sacc_array[j] <= t <= sacc_array[j+1],
    The (d-1)-th (last) stage corresponds to t >= sacc_array[d-1].
    In the j-th (0<=j<=d-1) stage,
    the model has drift mu_array[j] and diffusion sigma_array[j],
    the upper boundary has intercept ub[j] and slope b1_array[j].
    the lower boundary has intercept lb[j] and slope b2_array[j].

    mu_array, sacc_array, sigma_array, b1_array, b2_array MUST have the same length d.
    """

    def __init__(
        self, mu_array, sacc_array, sigma_array, a1, b1_array, a2, b2_array, x0
    ):
        super().__init__(x0)
        check_valid_multistage_params(
            mu_array, sacc_array, sigma_array, a1, b1_array, a2, b2_array
        )
        d = len(mu_array)
        self.d = d
        self.mu_array = mu_array
        self.sacc_array = sacc_array
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
            ub += b1_array[i - 1] * (sacc_array[i] - sacc_array[i - 1])
            lb += b2_array[i - 1] * (sacc_array[i] - sacc_array[i - 1])
            self.ub_array[i] = ub
            self.lb_array[i] = lb

    def drift_coeff(self, X: float, t: float) -> float:
        return piecewise_const_func(t, self.mu_array, self.sacc_array)

    def diffusion_coeff(self, X: float, t: float) -> float:
        return piecewise_const_func(t, self.sigma_array, self.sacc_array)

    @property
    def is_update_vectorizable(self) -> bool:
        return True

    def upper_bdy(self, t):
        return piecewise_linear_func(t, self.ub_array, self.b1_array, self.sacc_array)

    def lower_bdy(self, t):
        return piecewise_linear_func(t, self.lb_array, self.b2_array, self.sacc_array)


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
        """
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
        chunk_size=_DEFAULT_CHUNK_SIZE,
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
        from .simulator_cy import simulate_addm_fpt_cy

        if dt <= 0:
            raise ValueError("dt must be positive")
        rng = np.random.default_rng(rng)
        n_trials = len(d_data)
        max_d = sacc_array_data.shape[1]

        # Derive drifts from stimulus ratings and build mu array
        mu1_data, mu2_data = self.derive_drift(
            np.asarray(r1_data, dtype=np.float64),
            np.asarray(r2_data, dtype=np.float64),
        )
        mu_array_data = _swap_and_build_mu(mu1_data, mu2_data, flag_data, d_data, max_d)

        sacc_array_data = np.ascontiguousarray(sacc_array_data, dtype=np.float64)
        d_data = np.ascontiguousarray(d_data, dtype=np.int32)

        budget_time = min(self.a / self.b, T) if self.b > 0 else T
        if budget_time <= 0:
            rt_all = np.full(n_trials, -1.0, dtype=np.float64)
            choice_all = np.zeros(n_trials, dtype=np.int32)
            x_final_all = np.full(n_trials, self.x0, dtype=np.float64)
            return rt_all, choice_all, x_final_all

        # Generate all seeds up front for reproducibility across chunk sizes
        trial_seeds = np.ascontiguousarray(
            rng.integers(0, 2**63, size=n_trials, dtype=np.uint64)
        )

        rt_all = np.empty(n_trials, dtype=np.float64)
        choice_all = np.empty(n_trials, dtype=np.int32)
        x_final_all = np.empty(n_trials, dtype=np.float64)

        for start in range(0, n_trials, chunk_size):
            end = min(start + chunk_size, n_trials)

            rt_c, ch_c, xf_c = simulate_addm_fpt_cy(
                np.ascontiguousarray(mu_array_data[start:end]),
                np.ascontiguousarray(sacc_array_data[start:end]),
                np.ascontiguousarray(d_data[start:end]),
                self.sigma,
                self.a,
                self.b,
                self.x0,
                dt,
                budget_time,
                np.ascontiguousarray(trial_seeds[start:end]),
                n_threads,
            )

            rt_all[start:end] = rt_c
            choice_all[start:end] = ch_c
            x_final_all[start:end] = xf_c

        return rt_all, choice_all, x_final_all

    def fptd(
        self,
        t,
        mu_array,
        sacc_array,
        boundary,
        order=30,
        trunc_num=100,
        threshold=1e-20,
    ):
        """First-passage time density for a single trial configuration.

        Parameters
        ----------
        t : float
            Time at which to evaluate the density.
        mu_array : array-like, shape (d,)
            Drift rate per stage.
        sacc_array : array-like, shape (d,)
            Saccade onset times.
        boundary : int
            +1 (upper) or -1 (lower).
        order : int
            Gauss-Legendre quadrature order.

        Returns
        -------
        float
            FPTD value at time *t*.
        """
        from .multi_stage_cy import get_addm_fptd_cy

        mu_array = np.asarray(mu_array, dtype=np.float64)
        sacc_array = np.asarray(sacc_array, dtype=np.float64)
        d = len(mu_array)
        return get_addm_fptd_cy(
            t,
            d,
            mu_array,
            sacc_array,
            self.sigma,
            self.a,
            self.b,
            self.x0,
            boundary,
            trunc_num,
            threshold,
            order,
        )

    def mean_neg_log_likelihood(
        self,
        mu1_data,
        mu2_data,
        rt_data,
        choice_data,
        sacc_array_data,
        d_data,
        max_d,
        threshold=1e-20,
        num_threads=-1,
        order=30,
    ):
        """Mean negative log-likelihood over observed trials.

        Thin wrapper around ``compute_loss_parallel`` with model parameters
        filled in from *self*.

        Parameters
        ----------
        mu1_data, mu2_data : ndarray (n_trials,)
            Per-trial drift rates (pre-swapped for fixation order).
        rt_data : ndarray (n_trials,)
        choice_data : ndarray (n_trials,) int
        sacc_array_data : ndarray (n_trials, max_d)
        d_data : ndarray (n_trials,) int
        max_d : int
        threshold : float
        num_threads : int
            -1 for all available threads.
        order : int
            Gauss-Legendre quadrature order.

        Returns
        -------
        float
            Mean negative log-likelihood.
        """
        from .multi_stage_cy import compute_loss_parallel

        return compute_loss_parallel(
            mu1_data,
            mu2_data,
            rt_data,
            choice_data,
            sacc_array_data,
            d_data,
            max_d,
            self.sigma,
            self.a,
            self.b,
            self.x0,
            threshold,
            num_threads,
            order,
        )

    def to_multistage_model(self, mu1, mu2, sacc_array):
        """Construct a :class:`MultiStageModel` for a single trial.

        Useful for visualisation (``simulate_trajs``), computing densities
        via the Python path, or any use case requiring the full ``DDModel``
        interface (``drift_coeff``, ``upper_bdy``, etc.).

        Parameters
        ----------
        mu1, mu2 : float
            Drift rates for item-1 and item-2 fixation stages.
        sacc_array : array-like
            Saccade onset times.  ``sacc_array[0]`` should be 0.

        Returns
        -------
        MultiStageModel
        """
        sacc_array = np.asarray(sacc_array, dtype=np.float64)
        d = len(sacc_array)
        mu_array = get_alternating_mu_array(mu1, mu2, d)
        sigma_array = np.full(d, self.sigma)
        b1_array = np.full(d, -self.b)
        b2_array = np.full(d, self.b)
        return MultiStageModel(
            mu_array,
            sacc_array,
            sigma_array,
            self.a,
            b1_array,
            -self.a,
            b2_array,
            self.x0,
        )

    @property
    def boundary_collapsing_time(self):
        """Time at which collapsing boundaries meet (``inf`` if *b* == 0)."""
        return self.a / self.b if self.b > 0 else float("inf")


def piecewise_const_func(t, mu_array, sacc_array):
    """
    piecewise constant drift rate function, with drift rates `mu_array` and change points `sacc_array`
    """
    d = len(mu_array)
    if len(sacc_array) != d:
        raise ValueError(f"sacc_array length {len(sacc_array)} != mu_array length {d}")
    if not all(i < j for i, j in zip(sacc_array, sacc_array[1:])):
        raise ValueError("sacc_array must be strictly increasing")
    if d >= 2 and sacc_array[1] <= 0:
        raise ValueError("sacc_array[1] must be positive")
    _sacc_array = np.append(sacc_array, np.inf)
    return np.piecewise(
        t,
        [(t >= _sacc_array[i]) & (t < _sacc_array[i + 1]) for i in range(d)],
        mu_array,
    )


def piecewise_linear_func(t, a_array, b_array, sacc_array):
    """
    piecewise linear function, with intercepts `a_array`, slopes `b_array` and change points `sacc_array`
    """
    d = len(b_array)
    if len(a_array) != d:
        raise ValueError(f"a_array length {len(a_array)} != b_array length {d}")
    if len(sacc_array) != d:
        raise ValueError(f"sacc_array length {len(sacc_array)} != b_array length {d}")
    if not all(i < j for i, j in zip(sacc_array, sacc_array[1:])):
        raise ValueError("sacc_array must be strictly increasing")
    if d >= 2 and sacc_array[1] <= 0:
        raise ValueError("sacc_array[1] must be positive")

    # Extend sacc_array to include boundaries for the piecewise function
    _sacc_array = np.append(sacc_array, np.inf)

    # Define the piecewise function
    conds = [(t >= _sacc_array[i]) & (t < _sacc_array[i + 1]) for i in range(d)]
    funcs = [
        lambda t, i=i: a_array[i] + b_array[i] * (t - _sacc_array[i]) for i in range(d)
    ]
    return np.piecewise(t, conds, funcs)


# Weibull survival function (multiplicative)
def weibull_survival(t=1, lbda=1, k=1):
    """boundary based on weibull survival function.

    Arguments
    ---------
        t (int, optional): Defaults to 1.
        lbda (int, optional): Defaults to 1.
        k (int, optional): Defaults to 1.

    Returns
    -------
        np.array: Array of boundary values, same length as t
    """
    return np.exp(-np.power(np.divide(t, lbda), k))
