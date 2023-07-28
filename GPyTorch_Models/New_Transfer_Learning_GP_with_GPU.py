#TO DO: Everything
#Probably instead of a GPU we just could use woodbury identity
#and just invert the KTT instead of KSS which potentially would
#become big, but actually KTT generally is very small in Transfer Learning

import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# import positivity constraint
from gpytorch.constraints import Positive

class TL_Kernel(gpytorch.kernels.Kernel):
    # the kernel is stationary
    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self, NDomains,length_prior=None, length_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.NDomains = NDomains
        # set the parameter constraint to be positive, when nothing is specified
        if length_constraint is None:
            length_constraint = Positive()

        # register the raw parameter
        self.register_parameter(
            name='raw_length', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, NDomains, 1))
        )
        # register the constraint
        self.register_constraint("raw_length", length_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if length_prior is not None:
            self.register_prior(
                "length_prior",
                length_prior,
                lambda m: m.length,
                lambda m, v : m._set_length(v),
            )

    # now set up the 'actual' paramter
    @property
    def length(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_length_constraint.transform(self.raw_length)

    @length.setter
    def length(self, value):
        return self._set_length(value)

    def _set_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_length=self.raw_length_constraint.inverse_transform(value))

    # this is the kernel function
    def forward(self, x1, x2, idx1=None,idx2=None,square_dist=True, diag=False,**params):
        # apply lengthscale
        if idx2 is None:
            idx2 = idx1

        A, B = torch.meshgrid(self.length[idx1].flatten() ** 2.0, self.length[idx2].flatten() ** 2.0)
        # calculate the distance between inputs
        diff = -1.0/(A+B) * self.covar_dist(x1, x2, square_dist=square_dist, diag=diag, **params)
        return diff.exp_()

#gpytorch.kernels.RBFKernel()

class TL_Kernel_var(gpytorch.kernels.Kernel):
    # the kernel is stationary
    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self, NDomains,variance_prior=None,variance_constraint=None,length_prior=None, length_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.NDomains = NDomains

        # set the parameter constraint to be positive, when nothing is specified
        if length_constraint is None:
            length_constraint = Positive()

        if variance_constraint is None:
            variance_constraint = Positive()

        muDi_constraint = Positive()
        bDi_constraint = Positive()
        # register the raw parameter
        self.register_parameter(
            name='raw_length', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, NDomains, 1))
        )
        self.register_parameter(
            name='raw_variance', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, NDomains, 1))
        )
        self.register_parameter(
            name='raw_muDi', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, NDomains, 1))
        )
        self.register_parameter(
            name='raw_bDi', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, NDomains, 1))
        )
        # register the constraint
        self.register_constraint("raw_length", length_constraint)
        self.register_constraint("raw_variance", variance_constraint)
        self.register_constraint("raw_muDi", muDi_constraint)
        self.register_constraint("raw_bDi", bDi_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        "The ifs below are just a template, but not already tested by Juan, so I do not know if it would work"
        if length_prior is not None:
            self.register_prior(
                "length_prior",
                length_prior,
                lambda m: m.length,
                lambda m, v : m._set_length(v),
            )
        if variance_prior is not None:
            self.register_prior(
                "variance_prior",
                variance_prior,
                lambda m: m.variance,
                lambda m, v : m._set_variance(v),
            )

        self.alphaDi = 2.0*(1.0/(1.0+self.muDi)).pow(self.bDi) - 1.0

    # now set up the 'actual' paramter
    @property
    def length(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_length_constraint.transform(self.raw_length)

    @property
    def variance(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_variance_constraint.transform(self.raw_variance)

    @property
    def muDi(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_muDi_constraint.transform(self.raw_muDi)

    @property
    def bDi(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_bDi_constraint.transform(self.raw_bDi)

    @length.setter
    def length(self, value):
        return self._set_length(value)

    @variance.setter
    def variance(self, value):
        return self._set_variance(value)

    @muDi.setter
    def muDi(self, value):
        return self._set_muDi(value)

    @bDi.setter
    def bDi(self, value):
        return self._set_bDi(value)

    def _set_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_length=self.raw_length_constraint.inverse_transform(value))

    def _set_variance(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_variance)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_variance=self.raw_variance_constraint.inverse_transform(value))

    def _set_muDi(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_muDi)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_muDi=self.raw_muDi_constraint.inverse_transform(value))
        self.alphaDi = 2.0 * (1.0 / (1.0 + self.muDi)).pow(self.bDi) - 1.0

    def _set_bDi(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_bDi)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_bDi=self.raw_bDi_constraint.inverse_transform(value))
        self.alphaDi = 2.0 * (1.0 / (1.0 + self.muDi)).pow(self.bDi) - 1.0

    # this is the kernel function
    def forward(self, x1, x2, idx1=None,idx2=None,square_dist=True, diag=False,**params):
        # apply lengthscale
        if idx2 is None:
            idx2 = idx1

        A, B = torch.meshgrid(self.length[idx1].flatten() ** 2.0, self.length[idx2].flatten() ** 2.0)
        # Below the s2i and s2j are already treated as variances
        s2i, s2j = torch.meshgrid(self.variance[idx1].flatten(), self.variance[idx2].flatten())
        root_lengths = torch.sqrt(s2i*s2j)*torch.sqrt((2.0*torch.sqrt(A)*torch.sqrt(B))/(A+B))
        # Compute the relatedness variables
        self.alphaDi = 2.0 * (1.0 / (1.0 + self.muDi)).pow(self.bDi) - 1.0
        alphaDi, alphaDj = torch.meshgrid(self.alphaDi[idx1].flatten(), self.alphaDi[idx2].flatten())
        lambdasDij = alphaDi*alphaDj
        # Here we identify same Domains i==j, so the lambdaDii should be 1.0, if not then lambdaDij = alphai*alphaj
        IndxDi, IndxDj = torch.meshgrid(torch.tensor(idx1), torch.tensor(idx2))
        Dii = (IndxDi == IndxDj)
        # Here we apply a mask with Dii that contains the locations where i==j and makes lambdasDij = 1.0
        lambdasDij = lambdasDij * torch.logical_not(Dii) + Dii
        # calculate the distance between inputs
        diff = -1.0/(A+B) * self.covar_dist(x1, x2, square_dist=square_dist, diag=diag, **params)

        return lambdasDij*root_lengths*diff.exp_()

#gpytorch.kernels.RBFKernel()


Nseed = 1
torch.manual_seed(Nseed)
import random
random.seed(Nseed)
x1 = torch.randn(3,2)
x2 = torch.randn(4,2)

"TODO run to find a matrix that is not squared, say x1 is got more data than x2!!!!!!!!!!!"

covar = TL_Kernel_var(NDomains= 3) #gpytorch.kernels.RBFKernel()
print(covar)
covar._set_length([0.8,0.1,0.4])
covar._set_variance([1,0.5,1.0])
covar._set_muDi([10.,20.5,1.0])
indx1 = [0,1,1]
indx2 = [0,0,1,1]
MyK = covar(x1,x2,idx1=indx1,idx2=indx2).evaluate()
print(MyK)
#We could try to build kernel per region: KSS, KTS and KTT, maybe use idea from the kernel index of MOGP
#something like
#def forward(self, x1, x2, ind_1,ind_2,**params):

# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = ExactGPModel(train_x, train_y, likelihood)
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
# mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.numel())