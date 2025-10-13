import numpy as np
from abc import ABC, abstractmethod
from numbers import Number
from .utils import check_valid_multistage_params


class DDModel(ABC):
    """
    Abstract base class for drift-diffusion models.
    Defines a template for subclasses to implement specific drift and boundary behaviors.
    """

    def __init__(self, x0):
        self.x0 = x0

    def initialize_X0(self, t0, num):
        """
        Initialize the initial condition X0.
        """
        if isinstance(self.x0, Number):
            X0 = self.x0 * np.ones(num)
        elif isinstance(self.x0, dict):
            dist_name = self.x0.get("dist_name")
            if dist_name == "uniform":
                X0 = np.random.uniform(self.lower_bdy(t0), self.upper_bdy(t0), num)
            elif dist_name == "beta":
                alpha = self.x0.get("alpha")
                beta = self.x0.get("beta")
                if alpha is None or beta is None:
                    raise ValueError("Missing alpha and beta parameters for beta distribution")
                X0 = np.random.beta(alpha, beta, num) * (self.upper_bdy(t0) - self.lower_bdy(t0)) + self.lower_bdy(t0)
            else:
                raise ValueError("Unsupported distribution type")
        else:
            raise ValueError("Invalid initial condition format")
        return X0

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

    def simulate_trajs(self, T, Nt=1000, num=1000):
        """
        Simulates multiple trajectories of the drift-diffusion model, only for visualization.
        """
        t0 = 0
        Nt = int(Nt)
        dt = float(T - t0) / Nt
        t_grid = np.linspace(t0, T, Nt + 1)

        X0 = self.initialize_X0(t0, num)
        X_grids = np.zeros((num, Nt + 1))
        X_grids[:, 0] = X0
        dW = np.sqrt(dt) * np.random.normal(size=(num, Nt))
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

    def simulate_fpt_datum(self, dt=0.001):
        """
        Simulate one instance of the first passage time (FPT) for a given drift-diffusion model.
        used to generate the data for performing inference
        Returns:
        - First passage time `t_fpt`
        - Boundary indicator (`1` for upper boundary, `-1` for lower boundary)
        """
        t = 0.0
        X = self.initialize_X0(t, num=1)[0]
        while True:
            dW = np.random.normal(loc=0.0, scale=np.sqrt(dt))
            drift = self.drift_coeff(X, t)
            diffusion = self.diffusion_coeff(X, t)
            X_new = X + drift * dt + diffusion * dW
            upper_bound = self.upper_bdy(t + dt)
            lower_bound = self.lower_bdy(t + dt)
            t_fpt = t + 0.5 * dt
            if X_new >= upper_bound:
                return t_fpt, 1
            elif X_new <= lower_bound:
                return t_fpt, -1
            X = X_new
            t += dt

    def simulate_fptd_tillT(self, T: float, dt: float = 0.001, num: int = 1000):
        """
        Simulates first passage times (FPT) for multiple sample paths **in parallel**.
        Vectorized to speed up the simulation.
        Automatically halts when the simulation time exceeds T.

        Returns
        -------
        fp_times : np.ndarray
            Array of first passage times (positive = upper boundary, negative = lower boundary)
        nonexit_x : np.ndarray
            Final values for trajectories that did not exit by time T
        """
        t = 0.0
        X = self.initialize_X0(t, num)
        active_idx = np.arange(num)
        fp_times = np.full(num, np.nan)

        steps = 0
        while t < T and active_idx.size > 0:
            if steps % 10000 == 0:
                print(f"Time step {steps}: {active_idx.size} paths active at t={t:.6f}.")
            # Cap final step to not exceed T
            dt_curr = min(dt, T - t)

            # Euler–Maruyama update
            X_prev = X[active_idx]
            dW = np.random.normal(loc=0.0, scale=np.sqrt(dt_curr), size=active_idx.size)
            drift = self.drift_coeff(X_prev, t)
            diffusion = self.diffusion_coeff(X_prev, t)
            X_new = X_prev + drift * dt_curr + diffusion * dW

            # Boundary checks
            upper_bound = self.upper_bdy(t + dt_curr)
            lower_bound = self.lower_bdy(t + dt_curr)
            hit_upper = X_new >= upper_bound
            hit_lower = X_new <= lower_bound

            if np.any(hit_upper):
                crossing_time = t + 0.5 * dt_curr
                fp_times[active_idx[hit_upper]] = crossing_time
            if np.any(hit_lower):
                crossing_time = t + 0.5 * dt_curr
                fp_times[active_idx[hit_lower]] = -crossing_time

            # Update active indices
            X[active_idx] = X_new
            active_idx = active_idx[~(hit_upper | hit_lower)]

            t += dt_curr
            steps += 1

        # Remaining trajectories are censored at T
        nonexit_x = X[np.isnan(fp_times)]
        return fp_times[~np.isnan(fp_times)], nonexit_x

    # # Play the same role as simulate_fptd_tillT, consider removing
    # def simulate_fpt_data(self, dt: float = 1e-3, num: int = 1000, T_max: float = np.inf):
    #     """
    #     Vectorized simulation of *num* first passage times (FPTs).
    #     Halts if simulation time t exceeds T_max.
    #     """
    #     t = 0.0
    #     X = self.initialize_X0(t, num)
    #     active_idx = np.arange(num)

    #     result = np.empty((num, 2))
    #     result[:] = np.nan  # initialize time column with NaN
    #     result[:, 1] = 0  # 0 = no exit yet

    #     steps = 0
    #     while active_idx.size:
    #         if steps % 10000 == 0:
    #             print(f"Time step {steps}: {active_idx.size} paths active at t={t:.6f}.")
    #         if t > T_max:
    #             print(f"Simulation halted: reached time limit T_max = {T_max}. "
    #                 f"{active_idx.size} paths did not exit.")
    #             break

    #         dW = np.random.normal(scale=np.sqrt(dt), size=active_idx.size)

    #         X_prev = X[active_idx]
    #         drift = self.drift_coeff(X_prev, t)
    #         diffusion = self.diffusion_coeff(X_prev, t)

    #         X_new = X_prev + drift * dt + diffusion * dW

    #         upper_bound = self.upper_bdy(t + dt)
    #         lower_bound = self.lower_bdy(t + dt)

    #         hit_u = X_new >= upper_bound
    #         hit_l = X_new <= lower_bound
    #         hits = hit_u | hit_l

    #         if np.any(hits):
    #             exit_time = t + 0.5 * dt
    #             exited = active_idx[hits]
    #             result[exited, 0] = exit_time
    #             result[exited, 1] = np.where(hit_u[hits], 1, -1)

    #         X[active_idx] = X_new
    #         active_idx = active_idx[~hits]
    #         t += dt
    #         steps += 1

    #     return result



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
    Subclass for multi-stage model, where the drift and diffusion are piecewise constant, and the boundaries are piecewise linear.
    The j-th (0<=j<=d-2) stage corresponds to sacc_array[j] <= t <= sacc_array[j+1],
    The (d-1)-th (last) stage corresponds to t >= sacc_array[d-1].
    In the j-th (0<=j<=d-1) stage,
    the model has drift mu_array[j] and diffusion sigma_array[j],
    the upper boundary has intercept ub[j] and slope b1_array[j].
    the lower boundary has intercept lb[j] and slope b2_array[j].

    mu_array, sacc_array, sigma_array, b1_array, b2_array MUST have the same length d.
    """

    def __init__(self, mu_array, sacc_array, sigma_array, a1, b1_array, a2, b2_array, x0):
        super().__init__(x0)
        check_valid_multistage_params(mu_array, sacc_array, sigma_array, a1, b1_array, a2, b2_array)
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
        for i in range(d):
            if i == 0:
                pass
            else:
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


def piecewise_const_func(t, mu_array, sacc_array):
    """
    piecewise constant drift rate function, with drift rates `mu_array` and change points `sacc_array`
    """
    d = len(mu_array)
    assert len(sacc_array) == d
    assert all(i < j for i, j in zip(sacc_array, sacc_array[1:]))
    if d >= 2:
        assert sacc_array[1] > 0
    _sacc_array = np.append(sacc_array, np.inf)
    return np.piecewise(t, [(t >= _sacc_array[i]) & (t < _sacc_array[i + 1]) for i in range(d)], mu_array)


def piecewise_linear_func(t, a_array, b_array, sacc_array):
    """
    piecewise linear function, with intercepts `a_array`, slopes `b_array` and change points `sacc_array`
    """
    d = len(b_array)
    assert len(a_array) == d
    assert len(sacc_array) == d
    assert all(i < j for i, j in zip(sacc_array, sacc_array[1:]))
    if d >= 2:
        assert sacc_array[1] > 0

    # Extend sacc_array to include boundaries for the piecewise function
    _sacc_array = np.append(sacc_array, np.inf)

    # Define the piecewise function
    conds = [(t >= _sacc_array[i]) & (t < _sacc_array[i + 1]) for i in range(d)]
    funcs = [lambda t, i=i: a_array[i] + b_array[i] * (t - _sacc_array[i]) for i in range(d)]
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
