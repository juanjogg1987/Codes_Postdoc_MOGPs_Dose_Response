import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

train_x1 = torch.rand(2)
train_x2 = torch.rand(1)

train_y1 = torch.sin(train_x1 * (2 * math.pi)) + torch.randn(train_x1.size()) * 0.2
#train_y2 = -torch.cos(train_x1*train_x2 * (2 * math.pi)) + torch.randn(train_x2.size()) * 0.2
train_y2 = torch.cos(0.3+2*train_x2 * (2 * math.pi)) + torch.randn(train_x2.size()) * 0.2 + train_x2

# train_x1 = torch.rand(100,2)
# train_x2 = torch.rand(30,2)
#
# train_y1 = torch.sin(torch.sum(train_x1,1) * (2 * math.pi)) + torch.randn(train_x1.shape[0]) * 0.2
# train_y2 = torch.cos(torch.sum(train_x2,1) * (2 * math.pi)) + torch.randn(train_x2.shape[0]) * 0.2

#MyInterval = gpytorch.constraints.Interval(0.0001,0.1)

from typing import Optional
from gpytorch.priors import Prior

from gpytorch.constraints import Interval, Positive
from gpytorch.lazy import DiagLazyTensor, InterpolatedLazyTensor, PsdSumLazyTensor, RootLazyTensor

from gpytorch.utils.broadcasting import _mul_broadcast_shape

class TL_Kernel(gpytorch.kernels.Kernel):
    def __init__(
            self,
            num_tasks: int,
            rank: Optional[int] = 1,
            prior: Optional[Prior] = None,
            lambda_constraint: Optional[Interval] = None,
            **kwargs,
    ):
        if rank > num_tasks:
            raise RuntimeError("Cannot create a task covariance matrix larger than the number of tasks")
        super().__init__(**kwargs)

        # if var_constraint is None:
        #     var_constraint = Positive()

        if lambda_constraint is None:
            lambda_constraint = Interval(-100.0,100.0)

        self.register_parameter(
            name="covar_factor", parameter=torch.nn.Parameter(2*torch.rand(*self.batch_shape, num_tasks, rank)-1)
        )
        #self.register_parameter(name="raw_var", parameter=torch.nn.Parameter(torch.randn(*self.batch_shape, num_tasks)))
        if prior is not None:
            if not isinstance(prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(prior).__name__)
            self.register_prior("IndexKernelPrior", prior, lambda m: m._eval_covar_matrix())

        #self.register_constraint("raw_var", var_constraint)
        self.register_constraint("covar_factor", lambda_constraint)

    # @property
    # def var(self):
    #     return self.raw_var_constraint.transform(self.raw_var)
    #
    # @var.setter
    # def var(self, value):
    #     self._set_var(value)
    #
    # def _set_var(self, value):
    #     self.initialize(raw_var=self.raw_var_constraint.inverse_transform(value))

    def _eval_covar_matrix(self):
        cf = 2.0/(1+torch.exp(-self.covar_factor))-1.0
        #print(cf)
        num_tasks = cf.shape[0]
        res_aux = (cf @ cf.transpose(-1, -2)) * (torch.ones((num_tasks,num_tasks)) - torch.eye(num_tasks)) + torch.eye(num_tasks) #+ torch.diag_embed(self.var)
        #print(res_aux)
        return res_aux

    # @property
    # def covar_matrix(self):
    #     #var = self.var
    #     #res = PsdSumLazyTensor(RootLazyTensor(self.covar_factor), DiagLazyTensor(var))
    #     res = RootLazyTensor(self.covar_factor)
    #     return res

    def forward(self, i1, i2, **params):

        i1, i2 = i1.long(), i2.long()
        covar_matrix = self._eval_covar_matrix()
        batch_shape = _mul_broadcast_shape(i1.shape[:-2], i2.shape[:-2], self.batch_shape)

        res = InterpolatedLazyTensor(
            base_lazy_tensor=covar_matrix,
            left_interp_indices=i1.expand(batch_shape + i1.shape[-2:]),
            right_interp_indices=i2.expand(batch_shape + i2.shape[-2:]),
        )
        return res

class TL_GPModel(gpytorch.models.ExactGP):
    def __init__(self, source_target_x, source_target_y,target_x,target_y, likelihood, num_tasks):
        super(TL_GPModel, self).__init__(target_x, target_y, likelihood)
        #super(TL_GPModel, self).__init__(source_target_x, source_target_y, likelihood)
        self.source_target_x = source_target_x
        self.source_target_y = source_target_y
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()
        self.num_tasks = num_tasks
        # self.covar_module = gpytorch.kernels.MultitaskKernel(
        #     gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        # )
        # We learn an IndexKernel for 2 tasksT = {Tensor: (1000,)} tensor([0.4005, 0.1535, 0.0994, 0.4777, 0.9611, 0.4716, 0.8745, 0.5570, 0.5175,\n        0.5066, 0.7989, 0.2964, 0.6281, 0.3874, 0.6797, 0.3161, 0.6082, 0.9790,\n        0.8192, 0.8591, 0.2998, 0.0856, 0.3485, 0.5403, 0.7131, 0.7362, 0.9797,\n        0.2136, 0.4425, 0.0171, 0.8181, 0.9462, 0.3013, 0.2165, 0.6909, 0.9003,\n        0.3088, 0.3639, 0.3246, 0.8997, 0.1554, 0.8359, 0.8999, 0.1437, 0.4792,\n        0.8246, 0.3231, 0.9210, 0.9165, 0.1418, 0.0692, 0.7277, 0.3422, 0.4344,\n        0.5298, 0.4167, 0.0920, 0.7126, 0.6037, 0.9610, 0.3977, 0.8877, 0.9717,\n        0.1467, 0.8345, 0.8467, 0.9623, 0.2985, 0.0996, 0.8530, 0.9099, 0.3528,\n        0.0913, 0.8560, 0.2538, 0.5059, 0.9356, 0.4866, 0.6034, 0.7542, 0.3961,\n        0.6874, 0.5145, 0.4162, 0.0784, 0.0827, 0.9629, 0.1104, 0.4972, 0.3561,\n        0.6250, 0.8727, 0.6935, 0.8392, 0.5786, 0.9030, 0.6588, 0.2742, 0.9786,\n        0.9282, 0.9814, 0.1917, 0.8979, 0.0526, 0.1060, 0.6497, 0.8511, 0.4860,\n        0.3762, 0.7926, 0.9123, 0.6993, ...... View
        # (so we'll actually learn 2x2=4 tasks with correlations)
        #self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1, var_constraint=MyInterval)
        self.task_covar_module = TL_Kernel(num_tasks=2, rank=1)#, lambda_constraint=MyInterval)

    def forward(self,x):
        #mean_x = self.mean_module(x)

        Target_N, Target_Dim = x.shape
        label_i_Ttask = torch.full((Target_N, 1), dtype=torch.long, fill_value=self.num_tasks-1)
        i = torch.cat([self.source_target_x[1][:-Target_N], label_i_Ttask])
        "The cat below can have problems in D>1, have to change the [:,None]"
        SS_and_T = torch.cat([self.source_target_x[0][:-Target_N].reshape(-1,Target_Dim), x])
        # Get input-input covariance
        covar_x = self.covar_module(SS_and_T)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        KTT = covar[-Target_N:,-Target_N:]
        KSS = covar[:-Target_N, :-Target_N]
        KTS = covar[-Target_N:, :-Target_N]
        KST = covar[:-Target_N,-Target_N:]
        YS = self.source_target_y[:-Target_N]
        YT = self.source_target_y[-Target_N:]
        CSS = KSS#.clone() #+ Isig_S
        CTT = KTT#.clone()  # + Isig_T
        #CSS_i = torch.inverse(CSS)#torch.linalg.inv(CSS)
        #CSS_i = CSS.root_inv_decomposition()
        #mean_x = KTS.matmul( torch.linalg.solve(CSS,YS))    #KTS @ CSS_i @ YS
        #Covar_C = CTT - KTS.matmul(torch.linalg.solve(CSS,KST)) #+ 1e-4*torch.eye(CTT.shape[0])   #KTS @ CSS_i @ KST
        #print(torch.linalg.eigvals(KSS))
        mean_x = KTS.matmul(CSS.inv_matmul(YS))  # KTS @ CSS_i @ YS
        Covar_C = CTT - KTS.matmul(CSS.inv_matmul(KST.evaluate()))  # + 1e-4*torch.eye(CTT.shape[0])   #KTS @ CSS_i @ KST

        #mean_x = self.mean_module(x)
        #print("check Cov_C:",Covar_C.evaluate())

        return gpytorch.distributions.MultivariateNormal(mean_x, Covar_C)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
#likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)

likelihood.noise = 1.01  # Some small value, but don't make it too small or numerical performance will suffer. I recommend 1e-4.
#likelihood.noise_covar.raw_noise.requires_grad_(False)  # Mark that we don't want to train the noise.

train_i_task1 = torch.full((train_x1.shape[0],1), dtype=torch.long, fill_value=0)
train_i_task2 = torch.full((train_x2.shape[0],1), dtype=torch.long, fill_value=1)

full_train_x = torch.cat([train_x1, train_x2])
full_train_i = torch.cat([train_i_task1, train_i_task2])
full_train_y = torch.cat([train_y1, train_y2])

# Here we have two iterms that we're passing in as train_inputs
model = TL_GPModel((full_train_x, full_train_i), full_train_y, train_x2,train_y2,likelihood, num_tasks = 2)
model.covar_module.lengthscale=0.1
#model.covar_module.lengthscale = 0.3
#model.covar_module.raw_lengthscale.requires_grad_(False)

#model.task_covar_module.covar_factor.requires_grad_(False)


# this is for running the notebook in our testing framework
import os
smoke_test = ('CI' in os.environ)
training_iterations = 2 if smoke_test else 500


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    optimizer.zero_grad()
    #output = model(full_train_x, full_train_i)
    output = model(train_x2)#, full_train_i)
    #loss = -mll(output, full_train_y)
    loss = -mll(output, train_y2)
    loss.backward()
    print('Iter %d/50 - Loss: %.3f' % (i + 1, loss.item()))
    optimizer.step()

# Set into eval mode
model.eval()
likelihood.eval()

# Initialize plots
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

# Test points every 0.02 in [0,1]
#test_x = torch.rand(30,2)
test_x = torch.linspace(0, 1, 80)
#test_x = train_x2
test_i_task1 = torch.full((test_x.shape[0],1), dtype=torch.long, fill_value=0)
test_i_task2 = torch.full((test_x.shape[0],1), dtype=torch.long, fill_value=1)

# Make predictions - one task at a time
# We control the task we cae about using the indices

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpytorch.settings.fast_pred_var():
#with torch.no_grad():
    #observed_pred_y1 = likelihood(model(test_x, test_i_task1))
    #observed_pred_y2 = likelihood(model(test_x, test_i_task2))
    #observed_pred_y2 = likelihood(model(train_x2))
    observed_pred_y2 = likelihood(model(test_x))
    #my_pred_y2 = model(test_x)

# Define plotting function
def ax_plot(ax, train_y, train_x, rand_var, title):
    # Get lower and upper confidence bounds
    lower, upper = rand_var.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.detach().numpy(), train_y.detach().numpy(), 'k*')
    # Predictive mean as blue line
    ax.plot(test_x.detach().numpy(), rand_var.mean.detach().numpy(), 'b')
    # Shade in confidence
    ax.fill_between(test_x.detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(title)

# Plot both tasks
#ax_plot(y1_ax, train_y1, train_x1, observed_pred_y1, 'Observed Values (Likelihood)')
ax_plot(y2_ax, train_y2, train_x2, observed_pred_y2, 'Observed Values (Likelihood)')

mystd = observed_pred_y2.variance**0.5

# plt.plot(train_x2,train_y2,'x')
# plt.plot(test_x,observed_pred_y2.mean,'b.')
# plt.plot(test_x,observed_pred_y2.mean+2*mystd,'.',color='cyan')
# plt.plot(test_x,observed_pred_y2.mean-2*mystd,'.',color='cyan')

#plt.plot(train_y2,'x')
# plt.plot(observed_pred_y2.mean,'b.')
# plt.plot(observed_pred_y2.mean+2*mystd,'.',color='cyan')
# plt.plot(observed_pred_y2.mean-2*mystd,'.',color='cyan')