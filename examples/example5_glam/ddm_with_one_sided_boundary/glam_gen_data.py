import numpy as np
import time
import pickle
import argparse
from efficient_fpt.models import DDModel

class ExampleModel(DDModel):
    def __init__(self, mu, b, T):
        super().__init__(x0=0)
        self.mu = mu
        self.b = b
        self.T = T
        self.sigma = 1

    def drift_coeff(self, X: float, t: float) -> float:
        return self.mu * (t < self.T)

    def diffusion_coeff(self, X: float, t: float) -> float:
        return self.sigma

    def is_update_vectorizable(self) -> bool:
        return True

    def upper_bdy(self, t: float) -> float:
        return self.b * np.ones_like(t)

    def lower_bdy(self, t: float) -> float:
        return -np.inf * np.ones_like(t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate FPT data for ExampleModel.")
    parser.add_argument('--mu', type=float, required=True, help='Drift rate (mu)')
    parser.add_argument('--b', type=float, required=True, help='Upper boundary (b)')
    parser.add_argument('--T', type=float, required=True, help='Duration of nonzero drift (T)')
    parser.add_argument('--dt', type=float, default=0.0001, help='Time step size')
    parser.add_argument('--num', type=int, default=10000, help='Number of FPT samples to simulate')
    parser.add_argument('--T_max', type=int, default=10000, help='Maximum simulation time')

    args = parser.parse_args()

    model = ExampleModel(mu=args.mu, b=args.b, T=args.T)

    start_time = time.time()
    fp_times, np_poss = model.simulate_fptd_tillT(T=args.T_max, dt=args.dt, num=args.num)
    end_time = time.time()

    num_fpt = len(fp_times)
    print('Number of FPT data:', num_fpt)
    print(f'Simulation takes {end_time - start_time:.2f} seconds.')

    data_to_save = {
        "fp_times": fp_times,
        "np_poss": np_poss,
        "mu": args.mu,
        "T": args.T,
        "sigma": 1,
        "b": args.b,
        "x0": 0,
        "num_fpt": num_fpt,
        "dt": args.dt,
    }

    fname = f"glam_data_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Saved data to {fname}")
