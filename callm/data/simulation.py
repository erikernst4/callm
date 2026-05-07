
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

    # def generate_confidences(self):

    #     priors = torch.ones(self.num_eqclasses) / self.num_eqclasses

    #     # Create mean and std of each gaussian
    #     mu = []
    #     for k in range(self.num_eqclasses):
    #         one_hot = torch.zeros(self.num_eqclasses)
    #         one_hot[k] = 1.0
    #         # mu_n = torch.rand(self.samples_per_eqclass, self.num_eqclasses) - 0.5 + one_hot
    #         mu_n = torch.randn(self.samples_per_eqclass, self.num_eqclasses) * self.sigma_K + one_hot
    #         mu.append(mu_n)
    #         # compute minimum distance between mu_n points
    #         # min_diff = float("inf")
    #         # for i in range(self.samples_per_eqclass):
    #         #     for j in range(i + 1, self.samples_per_eqclass):
    #         #         diff = torch.norm(mu_n[i] - mu_n[j])
    #         #         min_diff = min(min_diff, diff.item())
    #         # min_diff = torch.cdist(mu_n, mu_n).triu(1).view(-1)
    #         # min_diff = min_diff[min_diff > 0].min().item()
    #         # sigma_n = min_diff / 2 / 3 * self.sigma_expand
    #         # sigma.append(sigma_n)

    #     confidences_eqclass = []
    #     confidences_answer = []
    #     correctness = []
    #     x_samples = []
    #     for k_sampled in torch.randint(0, self.num_eqclasses, (self.num_samples,)):
    #         # Sample an eqclass
    #         k = k_sampled.item()
    #         n = torch.randint(0, self.samples_per_eqclass, (1,)).item()
    #         # x = torch.normal(mu[k][n], sigma[k])
    #         x = torch.randn(self.num_eqclasses) * self.sigma_N + mu[k][n]
    #         x_samples.append(x)
    #         p_k = priors
    #         p_n_given_k = 1.0 / self.samples_per_eqclass / self.num_eqclasses
    #         p_x_given_n_k = torch.zeros(self.samples_per_eqclass, self.num_eqclasses)
    #         for kk in range(self.num_eqclasses):
    #             for nn in range(self.samples_per_eqclass):
    #                 # p_x_given_n_k[nn, kk] = torch.exp(
    #                 #     -0.5 * (torch.sum((x - mu[kk][nn]) ** 2) / sigma[kk] ** 2)
    #                 # ) / (sigma[kk] * torch.sqrt(torch.tensor(2.0) * torch.pi))
    #                 p_x_given_n_k[nn, kk] = torch.exp(
    #                     -0.5 * (torch.sum((x - mu[kk][nn]) ** 2) / self.sigma_N ** 2)
    #                 ) / (self.sigma_N * torch.sqrt(torch.tensor(2.0) * torch.pi))
    #         p_n_k_given_x = (
    #             p_x_given_n_k
    #             * p_n_given_k
    #             * p_k
    #             / (p_x_given_n_k * p_n_given_k * p_k).sum()
    #         )
    #         p_k_given_x = p_n_k_given_x.sum(dim=0)
    #         if self.suboptimal_T is None:
    #             k_pred = torch.argmax(p_k_given_x).item()
    #         else:
    #             p_k_given_x_T = p_k_given_x ** (1.0 / self.suboptimal_T)
    #             p_k_given_x_T /= p_k_given_x_T.sum()
    #             k_pred = torch.multinomial(p_k_given_x_T, 1).item()

    #         corr = 1.0 if k_pred == k else 0.0
    #         confidences_eqclass.append(p_k_given_x[k].item())
    #         confidences_answer.append(p_n_k_given_x[n, k].item())
    #         correctness.append(corr)

    #     return (
    #         torch.tensor(confidences_eqclass),
    #         torch.tensor(confidences_answer),
    #         torch.tensor(correctness),
    #         torch.stack(x_samples),
    #         mu
    #     )


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
        confidences_eqclass = p_k_given_x[torch.arange(self.num_samples), k_sampled]
        confidences_answer  = p_nk_given_x[torch.arange(self.num_samples), n_sampled, k_sampled]

        return (
            confidences_eqclass,
            confidences_answer,
            correctness,
            x_samples,
            mu,
        )
    


class SimulationDataset2D:

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
        confidences_eqclass = p_k_given_x[torch.arange(self.num_samples), k_sampled]
        confidences_answer  = p_nk_given_x[torch.arange(self.num_samples), n_sampled, k_sampled]

        return (
            confidences_eqclass,
            confidences_answer,
            correctness,
            x_samples,
            mu,
        )