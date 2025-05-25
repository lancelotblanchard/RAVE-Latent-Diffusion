import torch
import torch.nn as nn
from tqdm import tqdm


class EDM(nn.Module):
    def __init__(self, net, datashape=(1024), sigma_data=0.5):
        """
        EDM-Style Diffusion Model.
        Attributes:
            net (nn.Module): underlying neural network
            datashape (Tuple): shape of the data example, excluding batch size
            sigma_data (float): per-dim standard deviation of the dataset
        """
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data
        self.datashape = datashape
        assert len(datashape) > 0

    def forward(self, y, condition=None, cond_mask_prob=0.1, P_mean=-1.2, P_std=1.2):
        """
        Given noisy data, predicts a mixture of data and noise, computes loss.

        Args:
            P_mean (float): mean of log(sigma)
            P_std (float): std of log(sigma) sampled during training.

        Notes:
            c_noise is inside [-2.1, 1.5].
            The chances it doesn't is one in 500 million (z score=6)
        """
        batch_size = y.shape[0]
        sigmas = torch.exp(P_mean + torch.randn(batch_size, device=y.device) * P_std)
        sigmas = self._add_dims(sigmas, batch_size)

        # c_skip, c_out, c_in have shapes (batch_size, 1, ..., 1)
        # c_noise has shape (batch_size, )
        c_skip, c_out, c_in, c_noise = self._get_cs(sigmas)

        n = torch.randn_like(y, device=y.device) * sigmas

        net_out = self.net(c_in * (y + n), ts=c_noise, condition=condition, cond_mask_prob=cond_mask_prob)

        target = (y - c_skip * (y + n)) / c_out
        loss = nn.functional.mse_loss(net_out, target)
        return loss

    @torch.no_grad()
    def generate(
        self,
        batch_size=1,
        condition=None,
        cond_scale=1.0,
        num_steps=20,
        sigma_max=80,
        sigma_min=0.002,
        rho=7,
        return_intermediates=False,
    ):
        device = next(self.parameters()).device

        x_curr = torch.randn((batch_size, *self.datashape), device=device) * sigma_max

        sigmas = (
            torch.linspace(
                sigma_max ** (1 / rho),
                sigma_min ** (1 / rho),
                num_steps,
            )
            ** rho
        )
        sigmas = torch.cat((sigmas, torch.tensor([0.0])))
        sigmas = sigmas.to(device)

        if return_intermediates:
            intermediates = []

        for step in tqdm(range(num_steps), desc="Generating"):
            sigma = sigmas[step]
            sigma_next = sigmas[step + 1]
            delta_sigma = sigma_next - sigma

            if return_intermediates:
                intermediates.append(x_curr.cpu())

            d_curr = self._get_derivative(x_curr, self._add_dims(sigma, batch_size),
                                          condition=condition, cond_scale=cond_scale)

            x_next = x_curr + d_curr * delta_sigma

            if step != num_steps - 1:
                d_next = self._get_derivative(
                    x_next, self._add_dims(sigma_next, batch_size),
                    condition=condition, cond_scale=cond_scale
                )
                d = (d_curr + d_next) / 2

                x_next = x_curr + d * delta_sigma

            x_curr = x_next

        if return_intermediates:
            intermediates.append(x_curr.cpu())
            return x_curr.cpu(), torch.stack(intermediates, dim=0), sigmas.cpu()

        return x_curr

    def _add_dims(self, x, batch_size=1):
        if x.ndim == 0 or x.shape[0] != batch_size:
            x = x.expand(batch_size)
        return x.view((batch_size,) + (1,) * len(self.datashape))

    def _get_cs(self, sigma: torch.Tensor) -> torch.Tensor:
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = (self.sigma_data * sigma) / torch.sqrt(self.sigma_data**2 + sigma**2)
        c_in = torch.sqrt(sigma**2 + self.sigma_data**2).reciprocal()
        c_noise = 0.25 * torch.log(sigma)
        c_noise = c_noise.reshape(-1)
        return c_skip, c_out, c_in, c_noise

    def _denoise(self, x, sigma, condition=None, cond_scale=1.0):
        c_skip, c_out, c_in, c_noise = self._get_cs(sigma)
        # CFG
        if condition is not None and cond_scale != 1.0:
            out_unmasked = self.net(c_in * x, c_noise, condition=condition, cond_mask_prob=0.0)
            out_masked = self.net(c_in * x, c_noise, condition=condition, cond_mask_prob=1.0)
            out = out_unmasked * cond_scale + out_masked * (1 - cond_scale)
        else:
            out = self.net(c_in * x, c_noise)
        return c_skip * x + c_out * out

    def _get_derivative(self, x, sigma, condition=None, cond_scale=1.0):
        return (x - self._denoise(x, sigma=sigma, condition=condition, cond_scale=cond_scale)) / sigma
