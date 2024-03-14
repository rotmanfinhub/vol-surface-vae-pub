import torch
import torch.nn as nn
import torch.nn.functional as Func

class SABRBackwardModel(nn.Module):
    def __init__(self, r=0, q=0,
                beta=1, rho=-0.7, volvol=0.3, 
                moneyness_grid=[0.7, 0.85, 1, 1.15, 1.3], ttm_grid=[0.08333, 0.25, 0.5, 1, 2]) -> None:
        super().__init__()
        self.r = r
        self.q = q
        self.beta = beta
        self.rho = rho
        self.volvol = volvol
        self.moneyness_grid = moneyness_grid
        self.ttm_grid = ttm_grid
    
    def forward(self, vol, price):
        """Convert SABR instantaneous vol to option implied vol

        Args:
            vol (np.ndarray): SABR instantaneous vol in shape (num_path, num_period)
            price (np.ndarray): underlying stock price in shape (num_path, num_period)
        Returns:
            np.ndarray: implied vol in shape (num_paths, num_period, ttm_grid, moneyness_grid)
        """
        SABRIVs = torch.zeros((vol.shape[0], vol.shape[1], len(self.ttm_grid), len(self.moneyness_grid)))
        for t in range(len(self.ttm_grid)):
            for i in range(len(self.moneyness_grid)):
                K = price * self.moneyness_grid[i] # (num_path, num_period)
                F = price * torch.exp(torch.tensor((self.r - self.q) * t)) # (num_path, num_period)
                x = (F * K) ** ((1 - self.beta) / 2) # (num_path, num_period)
                y = (1 - self.beta) * torch.log(F / K) # (num_path, num_period)
                A = vol / (x * (1 + y * y / 24 + y * y * y * y / 1920)) # (num_path, num_period)
                B = 1 + t * (
                        ((1 - self.beta) ** 2) * (vol * vol) / (24 * x * x)
                        + self.rho * self.beta * self.volvol * vol / (4 * x)
                        + self.volvol * self.volvol * (2 - 3 * self.rho * self.rho) / 24
                ) # (num_path, num_period)
                Phi = (self.volvol * x / vol) * torch.log(F / K) # (num_path, num_period)
                Chi = torch.log((torch.sqrt(1 - 2 * self.rho * Phi + Phi * Phi) + Phi - self.rho) / (1 - self.rho)) # (num_path, num_period)
                SABRIV = torch.where(F == K, vol * B / (F ** (1 - self.beta)), A * B * Phi / (Chi+1e-8)) # (num_path, num_period)
                SABRIVs[:, :, t, i] = SABRIV

        return SABRIVs
    
    def search(self, prices, vols, target, lr=0.1, iterations=50):
        '''
            This class and function tries to find the realized price and volatility path that fits the SABR surface path.
        '''
        prices = prices.clone().detach().requires_grad_(True)
        vols = vols.clone().detach().requires_grad_(True)
        optim = torch.optim.SGD([prices, vols], lr=lr)

        for i in range(iterations):
            optim.zero_grad()
            predicted = self.forward(vols, prices)
            loss = Func.mse_loss(predicted, target)
            print(f"iteration {i}: loss={loss}")
            loss.backward()
            optim.step()
        return prices, vols