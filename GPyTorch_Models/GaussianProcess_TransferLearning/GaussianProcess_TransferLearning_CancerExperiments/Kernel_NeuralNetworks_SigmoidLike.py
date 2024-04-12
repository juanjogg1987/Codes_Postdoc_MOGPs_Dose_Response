import torch
import gpytorch
# import positivity constraint
from gpytorch.constraints import Positive, Interval


class NNetwork_kern(gpytorch.kernels.Kernel):
    # the kernel is stationary
    is_stationary = False

    # We will register the parameter when initializing the kernel
    def __init__(self, sig0_prior=None,sig0_constraint=None,sig_prior=None, sig_constraint=None, **kwargs):
        super().__init__(**kwargs)

        # set the parameter constraint to be positive, when nothing is specified
        if sig0_constraint is None:
            sig0_constraint = Positive()

        if sig_constraint is None:
            sig_constraint = Positive()

        # register the raw parameter
        self.register_parameter(
            name='raw_sig0', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_parameter(
            name='raw_sig', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        # register the constraint
        self.register_constraint("raw_sig0", sig0_constraint)
        self.register_constraint("raw_sig", sig_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        "The ifs below are just a template, but not already tested by Juan, so I do not know if it would work"
        if sig0_prior is not None:
            self.register_prior(
                "sig0_prior",
                sig0_prior,
                lambda m: m.sig0,
                lambda m, v : m._set_sig0(v),
            )
        if sig_prior is not None:
            self.register_prior(
                "sig_prior",
                sig_prior,
                lambda m: m.sig,
                lambda m, v : m._set_sig(v),
            )


    # now set up the 'actual' paramter
    @property
    def sig0(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_sig0_constraint.transform(self.raw_sig0)

    @property
    def sig(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_sig_constraint.transform(self.raw_sig)

    @sig0.setter
    def sig0(self, value):
        return self._set_sig0(value)

    @sig.setter
    def sig(self, value):
        return self._set_sig(value)

    def _set_sig0(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sig0)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_sig0=self.raw_sig0_constraint.inverse_transform(value))

    def _set_sig(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sig)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_sig=self.raw_sig_constraint.inverse_transform(value))

    # this is the kernel function
    def forward(self, x1, x2,square_dist=True, diag=False,**params):

        x1_ = torch.cat([self.sig0 * torch.ones(x1.shape[0], 1), self.sig*x1],1)
        x2_ = torch.cat([self.sig0 * torch.ones(x2.shape[0], 1), self.sig*x2], 1)

        # Compute inner product
        inner_product = torch.matmul(x1_, x2_.t())

        # Compute norms
        norm_x1_tilde_sq = torch.matmul(x1_ * x1_,torch.ones_like(x2_).t())
        norm_x2_tilde_sq = torch.matmul(x2_ * x2_, torch.ones_like(x1_).t()).t()

        # Compute kernel value
        numerator = 2 * inner_product
        denominator = torch.sqrt((1 + 2 * norm_x1_tilde_sq) * (1 + 2 * norm_x2_tilde_sq))
        kernel_value = torch.arcsin(numerator / denominator) * (2 / torch.pi)

        return kernel_value

# Define input space in 1-D
#X = torch.linspace(-5, 5, 6)[:, None]

# Define input space in 1-D
X1 = torch.linspace(-5, 5, 1000)[:, None]
X2 = torch.linspace(-5, 5, 1000)[:, None]

# Compute covariance matrix using Neumann kernel
#Sigma = torch.array([[0.5]])  # Covariance matrix (1-D case)
#K = torch.zeros((len(x1), len(x1)))
mykern = NNetwork_kern()
mykern.sig0 = 0.01
mykern.sig = 1.5
K = mykern(X1, X2).evaluate()

# K = torch.zeros((len(X1), len(X2)))
# for i, x1 in enumerate(X1):
#     for j, x2 in enumerate(X2):
#         K[i, j] = mykern(x1, x2).evaluate()

import numpy as np
import matplotlib.pyplot as plt
# Perform Cholesky decomposition
L = np.linalg.cholesky(K.detach().numpy() + 1e-5 * np.eye(len(X1)))  # Adding jitter for numerical stability

# Generate samples from standard normal distribution
num_samples = 10
standard_normal_samples = np.random.randn(len(X1), num_samples)

# Transform samples to obtain samples from Gaussian process
gp_samples = np.dot(L, standard_normal_samples)

# Plot samples
plt.figure(figsize=(10, 6))
for i in range(num_samples):
    plt.plot(X1, gp_samples[:, i], label=f'Sample {i + 1}')
plt.title('Samples from Gaussian Process with Neumann Kernel')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()


print(K)