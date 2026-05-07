
import torch


class SimulationDataset:

    def __init__(
        self, 
        num_samples, 
        num_eqclasses=2, 
        samples_per_eqclass=5, 
        sigma_K=0.1,
        sigma_N=0.1, 
        suboptimal_T=1.0, 
        seed=42
    ):
        self.num_samples = num_samples
        self.num_eqclasses = num_eqclasses
        self.samples_per_eqclass = samples_per_eqclass
        self.sigma_K = sigma_K
        self.sigma_N = sigma_N
        self.suboptimal_T = suboptimal_T
        self.seed = seed
        torch.manual_seed(self.seed)

    def generate_confidences(self):
        # Create mean and std of each gaussian
        # Shape: (num_eqclasses, samples_per_eqclass, num_eqclasses)
        one_hots = torch.eye(self.num_eqclasses).unsqueeze(1)  # (K, 1, K)
        mu = torch.randn(self.num_eqclasses, self.samples_per_eqclass, self.num_eqclasses) * self.sigma_K + one_hots

        # Sample true classes and prototypes for all samples at once
        k_sampled = torch.randint(0, self.num_eqclasses, (self.num_samples,))         # (N_samples,)
        n_sampled = torch.randint(0, self.samples_per_eqclass, (self.num_samples,))   # (N_samples,)

        # Get the corresponding prototype means and sample observations
        mu_selected = mu[k_sampled, n_sampled]  # (N_samples, K)
        x_samples = torch.randn(self.num_samples, self.num_eqclasses) * self.sigma_N + mu_selected  # (N_samples, K)

        # Compute likelihoods p(x | n, k) for all samples, prototypes, and classes at once
        # mu shape: (K, N, K_dim) -> (1, K, N, K_dim)
        # x  shape: (N_samples, K_dim) -> (N_samples, 1, 1, K_dim)
        mu_expanded = mu.unsqueeze(0)                  # (1, K, N, K_dim)
        x_expanded = x_samples.unsqueeze(1).unsqueeze(1)  # (N_samples, 1, 1, K_dim)

        # Squared distances: (N_samples, K, N)
        sq_dist = ((x_expanded - mu_expanded) ** 2).sum(dim=-1)

        # Gaussian likelihood (dropping shared normalisation constant)
        p_x_given_nk = torch.exp(-0.5 * sq_dist / self.sigma_N ** 2)  # (N_samples, K, N)
        p_x_given_nk = p_x_given_nk.permute(0, 2, 1)                  # (N_samples, N, K)

        # Joint: p(x, n, k) = p(x|n,k) * p(n|k) * p(k)  — both priors are uniform so they cancel in normalisation
        p_joint = p_x_given_nk  # (N_samples, N, K)

        # Posterior p(n, k | x)
        p_nk_given_x = p_joint / p_joint.sum(dim=(1, 2), keepdim=True)  # (N_samples, N, K)

        # Marginalise over n -> p(k | x)
        p_k_given_x = p_nk_given_x.sum(dim=1)  # (N_samples, K)

        # Prediction
        k_pred = torch.argmax(p_k_given_x, dim=1)  # (N_samples,)
        correctness = (k_pred == k_sampled).float()  # (N_samples,)

        # Gather confidences for the true (n, k) pair of each sample
        confidences_eqclass = p_k_given_x[torch.arange(self.num_samples), k_pred]
        confidences_answer  = p_nk_given_x[torch.arange(self.num_samples), n_sampled, k_pred]

        return (
            confidences_eqclass,
            confidences_answer,
            correctness,
            x_samples,
            mu,
        )
    


class SimulationDataset1D:

    def __init__(
        self, 
        num_samples, 
        num_eqclasses=2, 
        samples_per_eqclass=5, 
        sigma_K=0.1,
        sigma_N=0.1, 
        suboptimal_T=1.0, 
        seed=42
    ):
        self.num_samples = num_samples
        self.num_eqclasses = num_eqclasses
        self.samples_per_eqclass = samples_per_eqclass
        self.sigma_K = sigma_K
        self.sigma_N = sigma_N
        self.suboptimal_T = suboptimal_T
        self.seed = seed
        torch.manual_seed(self.seed)

    def generate_confidences(self):
        # Create mean and std of each gaussian (mean = 0, 1, ..., K-1, std = sigma_K)
        mu = torch.arange(self.num_eqclasses).view(-1, 1) + torch.randn(self.num_eqclasses, self.samples_per_eqclass) * self.sigma_K  # (K, N)

        # Sample true classes and prototypes for all samples at once
        k_sampled = torch.randint(0, self.num_eqclasses, (self.num_samples,))         # (N_samples,)
        n_sampled = torch.randint(0, self.samples_per_eqclass, (self.num_samples,))   # (N_samples,)

        # Get the corresponding prototype means and sample observations
        mu_selected = mu[k_sampled, n_sampled]  # (N_samples,)
        x_samples = torch.randn(self.num_samples) * self.sigma_N + mu_selected  # (N_samples,)

        # Compute likelihoods p(x | n, k) for all samples, prototypes, and classes at once
        # mu shape: (K, N) -> (1, K, N)
        # x  shape: (N_samples,) -> (N_samples, 1, 1)
        mu_expanded = mu.unsqueeze(0)
        x_expanded = x_samples.unsqueeze(1).unsqueeze(1)  # (N_samples, 1, 1)

        # Squared distances: (N_samples, K, N)
        sq_dist = ((x_expanded - mu_expanded) ** 2)

        # Gaussian likelihood (dropping shared normalisation constant)
        p_x_given_nk = torch.exp(-0.5 * sq_dist / self.sigma_N ** 2)  # (N_samples, K, N)
        p_x_given_nk = p_x_given_nk.permute(0, 2, 1)                  # (N_samples, N, K)

        # Joint: p(x, n, k) = p(x|n,k) * p(n|k) * p(k)  — both priors are uniform so they cancel in normalisation
        p_joint = p_x_given_nk  # (N_samples, N, K)

        # Posterior p(n, k | x)
        p_nk_given_x = p_joint / p_joint.sum(dim=(1, 2), keepdim=True)  # (N_samples, N, K)

        # Marginalise over n -> p(k | x)
        p_k_given_x = p_nk_given_x.sum(dim=1)  # (N_samples, K)

        # Prediction
        k_pred = torch.argmax(p_k_given_x, dim=1)  # (N_samples,)
        correctness = (k_pred == k_sampled).float()  # (N_samples,)

        # Gather confidences for the true (n, k) pair of each sample
        confidences_eqclass = p_k_given_x[torch.arange(self.num_samples), k_pred]
        confidences_answer  = p_nk_given_x[torch.arange(self.num_samples), n_sampled, k_pred]

        return (
            confidences_eqclass,
            confidences_answer,
            correctness,
            x_samples,
            mu,
        )