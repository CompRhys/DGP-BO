import gpytorch
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)


class DGPLastLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_inducing=16, n_latents=10, n_tasks=3):
        self.num_latents = n_latents
        self.n_tasks = n_tasks
        inducing_points = torch.randn(self.num_latents, num_inducing, self.n_tasks)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=torch.Size([self.num_latents])
        )

        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=self.n_tasks,
            num_latents=self.num_latents,
            latent_dim=-1,
        )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([self.num_latents])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([self.num_latents])),
            batch_shape=torch.Size([self.num_latents]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=64, linear_mean=True):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
            jitter_val=1e-3,
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                batch_shape=batch_shape,
                ard_num_dims=input_dims,
                lengthscale_constraint=gpytorch.constraints.Interval(0.2, 0.8),
            ),
            batch_shape=batch_shape,
            ard_num_dims=None,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class MultitaskHetDeepGP(DeepGP):
    def __init__(self, n_input_dims, n_output_dims, n_tasks):
        super().__init__()

        self.hidden_layer = DGPHiddenLayer(
            input_dims=n_input_dims, output_dims=n_output_dims, linear_mean=True
        )
        self.last_layer = DGPLastLayer(n_latents=n_input_dims * 2, n_tasks=n_tasks)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            num_tasks=n_tasks, noise_constraint=gpytorch.constraints.Interval(0.001, 10)
        )

    def forward(self, inputs, task_indices):
        task_indices1 = (
            torch.from_numpy(np.ones(inputs.shape[0], dtype=int) * 1).long()
            * task_indices
        )
        hidden_rep1 = self.hidden_layer(inputs)
        return self.last_layer(
            torch.distributions.Normal(
                loc=hidden_rep1.mean, scale=hidden_rep1.variance.sqrt()
            ).rsample(),
            task_indices=task_indices1,
        )

    def predict(self, test_x, task_indices):
        with torch.no_grad():
            preds = self.likelihood(self.forward(test_x, task_indices=task_indices))

        return preds.mean, preds.variance


class MultitaskIsoDeepGP(DeepGP):
    def __init__(self, n_input_dims, n_output_dims, n_tasks):
        super().__init__()

        self.hidden_layer = DGPHiddenLayer(
            input_dims=n_input_dims,
            output_dims=n_output_dims,
            linear_mean=True,
        )
        self.last_layer = DGPHiddenLayer(
            input_dims=self.hidden_layer.output_dims,
            output_dims=n_tasks,
            linear_mean=False,
        )

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=n_tasks, noise_constraint=gpytorch.constraints.Interval(0.001, 10)
        )

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        return self.last_layer(hidden_rep1)

    def predict(self, test_x):
        with torch.no_grad():
            preds = self.likelihood(self.forward(test_x)).to_data_independent_dist()

        return preds.mean.mean(0), preds.variance.mean(0)
