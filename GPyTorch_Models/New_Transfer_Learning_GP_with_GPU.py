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
    # the sinc kernel is stationary
    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self, NDomains,length_prior=None, length_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.NDomains = NDomains
        # set the parameter constraint to be positive, when nothing is specified
        if length_constraint is None:
            length_constraint = Positive()

        # register the raw parameter
        for i in range(NDomains):
            self.register_parameter(
                name='raw_length'+str(i+1), parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
            )
            # register the constraint
            self.register_constraint("raw_length"+str(i+1), length_constraint)

        #self.raw_length_list = [self.raw_length1,self.raw_length2] [exec("'self.raw_length'+str(i+1)) for i in range(Ntasks)"]
        #self.raw_length_list = exec("['self.raw_length'+str(i+1) for i in range(Ntasks)]")

        #self.length1 = self.raw_length1_constraint.transform(self.raw_length1)

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
        # self.raw_length1_constraint.transform(self.raw_length1)
        return exec("['self.raw_length'+str(i+1)+'_constraint.transform(self.raw_length'+str(i+1)+')' for i in range(self.NDomains)]")

    #['self.raw_length'+str(i+1) for i in range(Ntasks)]

    # @length.setter
    # def length(self, value):
    #     return self._set_length(value)

    @length.setter
    def length(self, value, i_domain = None):
        if i_domain is None:
            print("Remeber values has to be a unique value or a vector of NDomains size")
            assert value.shape == self.NDomains
        return [self._set_length(value,i=i) for i in range(self.NDomains)]


    def _set_length(self, value,i=0):
        if not torch.is_tensor(value):
            value = exec("'torch.as_tensor(value).to(self.raw_length'+str(i+1)+')'")
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        #self.initialize(raw_length1=self.raw_length1_constraint.inverse_transform(value))
        exec("'self.initialize(raw_length'+str(i+1)+'=self.raw_length'+str(i+1)+'_constraint.inverse_transform(value))'")

    # def _set_length(self, value,i_domain=None):
    #     if i_domain is None:
    #         for i in range(self.NDomains):
    #             if not torch.is_tensor(value[i]):
    #                 value_aux = exec("'torch.as_tensor(value[i]).to(self.raw_length'+str(i+1)+')'")
    #         # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
    #     #self.initialize(raw_length1=self.raw_length1_constraint.inverse_transform(value))
    #             #self.initialize(raw_length1=self.raw_length1_constraint.inverse_transform(value))
    #             exec("'self.initialize(raw_length'+str(i+1)+'=self.raw_length'+str(i+1)+'_constraint.inverse_transform(value_aux))'")

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # apply lengthscale
        x1_ = x1.div(self.length[0])
        x2_ = x2.div(self.length[0])
        # calculate the distance between inputs
        diff = self.covar_dist(x1_, x2_, **params)
        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20))
        # return sinc(diff) = sin(diff) / diff
        return torch.sin(diff).div(diff)

#gpytorch.kernels.RBFKernel()

x = torch.randn(5,2)

covar = TL_Kernel(NDomains= 1) #gpytorch.kernels.RBFKernel()
print(covar)
covar._set_length(0.5,i=0)

#MyK = covar(x).evaluate()

#We could try to build kernel per region: KSS, KTS and KTT, maybe use idea from the kernel index of MOGP
#something like
#def forward(self, x1, x2, ind_1,ind_2,**params):