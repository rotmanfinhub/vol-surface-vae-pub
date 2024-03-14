import numpy as np

class SABRSurface:
    def __init__(self, init_ttm=30, np_seed=1234, init_vol=0.3,
                r=0, q=0, t=252, frq=1,
                beta=1, rho=-0.7, volvol=0.3, 
                S=10, mu=0.0, num_sim=10000, 
                moneyness_grid=[0.7, 0.85, 1, 1.15, 1.3], ttm_grid=[0.08333, 0.25, 0.5, 1, 2]):
        # set the np random seed
        np.random.seed(np_seed)

        # Annual Risk Free Rate
        self.r = r
        # Annual Dividend
        self.q = q
        # Annual Trading Day
        self.T = t
        # Annual Volatility
        self.init_vol = init_vol
        # frequency of trading
        self.frq = frq
        self.dt = self.frq / self.T

        # Initial time to maturity
        self.init_ttm = init_ttm
        # Number of periods
        self.num_period = int(self.init_ttm / self.frq)
        # for grids
        self.moneyness_grid = moneyness_grid
        self.ttm_grid = ttm_grid

        # SABR parameters
        self.beta = beta
        self.rho = rho
        self.volvol = volvol

        # default strike price and mean, for gbm, sabr simulated data
        self.S = S
        self.mu = mu

        self.num_sim = num_sim
    
    def _sabr_sim(self, start_price=None):
        """Simulate SABR model
        1). stock price
        2). instantaneous vol

        Returns:
            np.ndarray: stock price in shape (num_path, num_period)
            np.ndarray: instantaneous vol in shape (num_path, num_period)
        """
        if start_price is None:
            num_sim = self.num_sim
        else:
            num_sim = start_price.shape[0]
        qs = np.random.normal(size=(num_sim, self.num_period + 1))
        qi = np.random.normal(size=(num_sim, self.num_period + 1))
        qv = self.rho * qs + np.sqrt(1 - self.rho * self.rho) * qi

        vol = np.zeros((num_sim, self.num_period + 1))
        vol[:, 0] = self.init_vol

        a_price = np.zeros((num_sim, self.num_period + 1))
        if start_price is None:
            a_price[:, 0] = self.S
        else:
            a_price[:, 0] = start_price
        
        for t in range(self.num_period):
            gvol = vol[:, t] * (a_price[:, t] ** (self.beta - 1))
            a_price[:, t + 1] = a_price[:, t] * np.exp(
                (self.mu - (gvol ** 2) / 2) * self.dt + gvol * np.sqrt(self.dt) * qs[:, t]
            )
            vol[:, t + 1] = vol[:, t] * np.exp(
                -self.volvol * self.volvol * 0.5 * self.dt + self.volvol * qv[:, t] * np.sqrt(self.dt)
            )

        return a_price, vol
    
    def _sabr_implied_vol(self, vol, price):
        """Convert SABR instantaneous vol to option implied vol

        Args:
            vol (np.ndarray): SABR instantaneous vol in shape (num_path, num_period)
            price (np.ndarray): underlying stock price in shape (num_path, num_period)
        Returns:
            np.ndarray: implied vol in shape (num_paths, num_period, ttm_grid, moneyness_grid)
        """
        SABRIVs = np.zeros((vol.shape[0], vol.shape[1], len(self.ttm_grid), len(self.moneyness_grid)))
        for t in range(len(self.ttm_grid)):
            for i in range(len(self.moneyness_grid)):
                K = price * self.moneyness_grid[i] # (num_path, num_period)
                F = price * np.exp((self.r - self.q) * t) # (num_path, num_period)
                x = (F * K) ** ((1 - self.beta) / 2) # (num_path, num_period)
                y = (1 - self.beta) * np.log(F / K) # (num_path, num_period)
                A = vol / (x * (1 + y * y / 24 + y * y * y * y / 1920)) # (num_path, num_period)
                B = 1 + t * (
                        ((1 - self.beta) ** 2) * (vol * vol) / (24 * x * x)
                        + self.rho * self.beta * self.volvol * vol / (4 * x)
                        + self.volvol * self.volvol * (2 - 3 * self.rho * self.rho) / 24
                ) # (num_path, num_period)
                Phi = (self.volvol * x / vol) * np.log(F / K) # (num_path, num_period)
                Chi = np.log((np.sqrt(1 - 2 * self.rho * Phi + Phi * Phi) + Phi - self.rho) / (1 - self.rho)) # (num_path, num_period)
                SABRIV = np.where(F == K, vol * B / (F ** (1 - self.beta)), A * B * Phi / (Chi+1e-8)) # (num_path, num_period)
                SABRIVs[:, :, t, i] = SABRIV

        return SABRIVs

    def get_sim_path_sabr(self, start_price=None):
        """ Simulate SABR underlying dynamic and implied volatility dynamic 
        
        Returns:
            np.ndarray: underlying asset price in shape (num_path, num_period)
            np.ndarray: implied volatility in shape (num_paths, num_period, ttm_grid, moneyness_grid)
        """

        # asset price 2-d array; sabr_vol
        print("Generating asset price paths (SABR)")
        a_price, sabr_vol = self._sabr_sim(start_price)

        # BS price 2-d array and bs delta 2-d array
        print("Generating implied vol")

        # SABR implied vol
        implied_vol = self._sabr_implied_vol(sabr_vol, a_price)

        self.implied_vol = implied_vol
        return a_price, implied_vol, sabr_vol
    
if __name__ == "__main__":
    sabr = SABRSurface()
    prices, ivs = sabr.get_sim_path_sabr()
    print(prices.shape)
    print(ivs.shape)