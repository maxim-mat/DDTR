import torch


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(self, n_timesteps):
        return torch.randint(low=1, high=self.noise_steps, size=(n_timesteps,))

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_data(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def denoise(self, model, x_t, t, denoiser_output='noise'):
        model.eval()
        with ((torch.no_grad())):
            x = x_t.to(self.device)
            for i in reversed(range(1, t)):
                t_tensor = torch.tensor([i]).long().to(self.device)
                t_prev = torch.tensor([i - 1]).long().to(self.device)
                predicted = model(x, t_tensor)
                alpha = self.alpha[t_tensor][:, None, None]
                alpha_hat = self.alpha_hat[t_tensor][:, None, None]
                alpha_hat_prev = self.alpha_hat[t_prev][:, None, None]
                beta = self.beta[t_tensor][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                if denoiser_output == 'noise':
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted)
                    + ((1 - alpha_hat_prev) / (1 - alpha_hat)) * torch.sqrt(beta) * noise
                elif denoiser_output == 'original':
                    x = (torch.sqrt(alpha_hat_prev) / (1 - alpha_hat)) * predicted
                    + ((torch.sqrt(alpha) * (1 - alpha_hat_prev)) / (1 - alpha_hat)) * x
                    + ((1 - alpha_hat_prev) / (1 - alpha_hat)) * torch.sqrt(beta) * noise
        model.train()
        return x

    def sample(self, model, n, num_categories, sequence_length, y=None, denoiser_output='noise'):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, num_categories, sequence_length)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t_tensor = (torch.ones(n) * i).long().to(self.device)
                t_prev = (torch.ones(n) * (i - 1)).long().to(self.device)
                predicted = model(x, t_tensor, y)
                alpha = self.alpha[t_tensor][:, None, None]
                alpha_hat = self.alpha_hat[t_tensor][:, None, None]
                alpha_hat_prev = self.alpha_hat[t_prev][:, None, None]
                beta = self.beta[t_tensor][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                if denoiser_output == 'noise':
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted)
                    + ((1 - alpha_hat_prev) / (1 - alpha_hat)) * torch.sqrt(beta) * noise
                elif denoiser_output == 'original':
                    x = (torch.sqrt(alpha_hat_prev) / (1 - alpha_hat)) * predicted
                    + ((torch.sqrt(alpha) * (1 - alpha_hat_prev)) / (1 - alpha_hat)) * x
                    + ((1 - alpha_hat_prev) / (1 - alpha_hat)) * torch.sqrt(beta) * noise
        model.train()
        return x

    def sample_with_matrix(self, model, n, num_categories, sequence_length, transition_dim, transition_matrix, 
                           x_ref=None, y=None, denoiser_output='noise'):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, num_categories, sequence_length)).to(self.device)
            m = torch.randn((num_categories, transition_dim, transition_dim)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t_tensor = (torch.ones(n) * i).long().to(self.device)
                t_prev = (torch.ones(n) * (i - 1)).long().to(self.device)
                predicted, matrix_hat, loss, seq_loss, mat_loss = model(x, t_tensor, x_ref, transition_matrix, y)
                alpha = self.alpha[t_tensor][:, None, None]
                alpha_hat = self.alpha_hat[t_tensor][:, None, None]
                alpha_hat_prev = self.alpha_hat[t_prev][:, None, None]
                beta = self.beta[t_tensor][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                if denoiser_output == 'noise':
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted)
                    + ((1 - alpha_hat_prev) / (1 - alpha_hat)) * torch.sqrt(beta) * noise
                elif denoiser_output == 'original':
                    x = (torch.sqrt(alpha_hat_prev) / (1 - alpha_hat)) * predicted
                    + ((torch.sqrt(alpha) * (1 - alpha_hat_prev)) / (1 - alpha_hat)) * x
                    + ((1 - alpha_hat_prev) / (1 - alpha_hat)) * torch.sqrt(beta) * noise
                    m = matrix_hat
        model.train()
        loss = loss.item() if loss is not None else None
        return x, m, loss, seq_loss, mat_loss
