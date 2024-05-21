import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# import positivity constraint
from gpytorch.constraints import Positive, Interval

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

class Kernel_CrossDomains(gpytorch.kernels.Kernel):
    # the kernel is stationary
    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self, NDomains,variance_prior=None,variance_constraint=None,length_prior=None, length_constraint=None, **kwargs):
        super().__init__(**kwargs)

        # set the parameter constraint to be positive, when nothing is specified
        if length_constraint is None:
            length_constraint = Positive()

        if variance_constraint is None:
            variance_constraint = Positive()

        # register the raw parameter
        self.register_parameter(
            name='raw_length', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, NDomains, 1))
        )
        self.register_parameter(
            name='raw_variance', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, NDomains, 1))
        )

        # register the constraint
        self.register_constraint("raw_length", length_constraint)
        self.register_constraint("raw_variance", variance_constraint)

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


    # now set up the 'actual' paramter
    @property
    def length(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_length_constraint.transform(self.raw_length)

    @property
    def variance(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_variance_constraint.transform(self.raw_variance)

    @length.setter
    def length(self, value):
        return self._set_length(value)

    @variance.setter
    def variance(self, value):
        return self._set_variance(value)

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

    # this is the kernel function
    def forward(self, x1, x2, idx1=None,idx2=None,square_dist=True, diag=False,**params):
        # apply lengthscale
        if idx2 is None:
            idx2 = idx1

        A, B = torch.meshgrid(self.length[idx1].flatten() ** 2.0, self.length[idx2].flatten() ** 2.0)
        # Below the s2i and s2j are already treated as variances
        s2i, s2j = torch.meshgrid(self.variance[idx1].flatten(), self.variance[idx2].flatten())
        root_lengths = torch.sqrt(s2i*s2j)*torch.sqrt((2.0*torch.sqrt(A)*torch.sqrt(B))/(A+B))
        # calculate the distance between inputs
        diff = -1.0/(A+B) * self.covar_dist(x1, x2, square_dist=square_dist, diag=diag, **params)

        return root_lengths*diff.exp_()

class TLRelatedness(gpytorch.kernels.Kernel):
    # the kernel is stationary
    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self, NDomains,muDi_constraint=None,bDi_constraint=None,**kwargs):
        super().__init__(**kwargs)

        # set the parameter constraint to be positive, when nothing is specified
        if muDi_constraint is None:
            muDi_constraint = Positive()
        if bDi_constraint is None:
            bDi_constraint = Positive()

        # register the raw parameter
        self.register_parameter(
            name='raw_muDi', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, NDomains, 1))
        )
        self.register_parameter(
            name='raw_bDi', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, NDomains, 1))
        )
        # register the constraint
        self.register_constraint("raw_muDi", muDi_constraint)
        self.register_constraint("raw_bDi", bDi_constraint)

        self.alphaDi = 2.0*(1.0/(1.0+self.muDi)).pow(self.bDi) - 1.0

    # now set up the 'actual' paramter
    @property
    def muDi(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_muDi_constraint.transform(self.raw_muDi)

    @property
    def bDi(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_bDi_constraint.transform(self.raw_bDi)

    @muDi.setter
    def muDi(self, value):
        return self._set_muDi(value)

    @bDi.setter
    def bDi(self, value):
        return self._set_bDi(value)

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
    def forward(self,x1, x2, idx1=None,idx2=None,square_dist=True, diag=False,**params):
        # apply lengthscale
        if idx2 is None:
            idx2 = idx1

        # Compute the relatedness variables
        self.alphaDi = 2.0 * (1.0 / (1.0 + self.muDi)).pow(self.bDi) - 1.0
        alphaDi, alphaDj = torch.meshgrid(self.alphaDi[idx1].flatten(), self.alphaDi[idx2].flatten())
        lambdasDij = alphaDi*alphaDj
        # Here we identify same Domains i==j, so the lambdaDii should be 1.0, if not then lambdaDij = alphai*alphaj
        IndxDi, IndxDj = torch.meshgrid(torch.tensor(idx1), torch.tensor(idx2))
        Dii = (IndxDi == IndxDj)
        # Here we apply a mask with Dii that contains the locations where i==j and makes lambdasDij = 1.0
        lambdasDij = lambdasDij * torch.logical_not(Dii) + Dii

        return lambdasDij

class Kernel_ICM(gpytorch.kernels.Kernel):
    # the kernel is stationary
    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self, NDomains,adq_prior=None,adq_constraint=None,length_prior=None, length_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.NDomains = NDomains

        # set the parameter constraint to be positive, when nothing is specified
        if length_constraint is None:
            length_constraint = Positive()

        if adq_constraint is None:
            adq_constraint = Interval(-1, 1) #Positive()

        # register the raw parameter
        self.register_parameter(
            name='raw_length', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_parameter(
            name='raw_adq', parameter=torch.nn.Parameter(torch.randn(*self.batch_shape, NDomains, 1))
        )
        # register the constraint
        self.register_constraint("raw_length", length_constraint)
        self.register_constraint("raw_adq", adq_constraint)

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
        if adq_prior is not None:
            self.register_prior(
                "adq_prior",
                adq_prior,
                lambda m: m.adq,
                lambda m, v : m._set_adq(v),
            )


    # now set up the 'actual' paramter
    @property
    def length(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_length_constraint.transform(self.raw_length)

    @property
    def adq(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_adq_constraint.transform(self.raw_adq)

    @length.setter
    def length(self, value):
        return self._set_length(value)

    @adq.setter
    def adq(self, value):
        return self._set_adq(value)

    def _set_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_length=self.raw_length_constraint.inverse_transform(value))

    def _set_adq(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_adq)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_adq=self.raw_adq_constraint.inverse_transform(value))

    # this is the kernel function
    def forward(self, x1, x2, idx1=None,idx2=None,square_dist=True, diag=False,**params):
        # apply lengthscale
        if idx2 is None:
            idx2 = idx1
        x1_ = x1.div(self.length)
        x2_ = x2.div(self.length)
        # calculate the distance between inputs
        # Below the s2i and s2j are already treated as variances
        a2i, a2j = torch.meshgrid(self.adq[idx1].flatten(), self.adq[idx2].flatten())
        a2i_times_a2j = a2i*a2j  #Multiplication of linear combination
        # calculate the distance between inputs
        diff = -1.0 * self.covar_dist(x1_, x2_, square_dist=square_dist, diag=diag, **params)
        # prevent divide by 0 errors
        #diff.where(diff == 0, torch.as_tensor(1e-20))
        return a2i_times_a2j*diff.exp_()

class Kernel_ICM_LessParams(gpytorch.kernels.Kernel):
    "This version of the Kernel_ICM_LessParams aims to avoid having so many hyper-parameters adq that are used for"
    "the linear combination coefficients fd = ad1*u1(x) + ad2*u2(x)+...+adQ*uQ(x), where"
    "Cov(fd(x),fd'(x')) = \sum_{q=1}^{Q} (adq*ad'q)*kq(x,x'). So instead of having all the combinations (adq*ad'q)"
    "That would represent the coregionalisation matrix coefficients, we use a covariance kernel to operate over the"
    "output indexes as follows Covq(d,d') = kernq_d(d,d'), and the initial covariance Cov(fd(x),fd'(x'))"
    "would become Cov(fd(x),fd'(x')) = \sum_{q=1}^{Q} kernq_d(adq*ad'q)*kq(x,x')"
    "We would assume a unique GP u(x)~GP(0,k(x,x')). Therefore, the actual Cov(fd(x),fd'(x')) = kern_d(ad*ad')*k(x,x')"
    "We would implement the LMC externally by combining sums of the kernel_ICM_LessParams as:"
    "Kernel_LMC = Kernel_ICM_LessParams(NDomains=NDomains)+...+Kernel_ICM_LessParams(NDomains=NDomains) as many Q wanted"
    # the kernel is stationary
    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self, NDomains,adq_length_prior=None,adq_length_constraint=None,length_prior=None, length_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.NDomains = NDomains

        # set the parameter constraint to be positive, when nothing is specified
        if length_constraint is None:
            length_constraint = Positive()

        if adq_length_constraint is None:
            adq_length_constraint = Positive()

        # register the raw parameter
        self.register_parameter(
            name='raw_length', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_parameter(
            name='raw_adq_length', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        # register the constraint
        self.register_constraint("raw_length", length_constraint)
        self.register_constraint("raw_adq_length", adq_length_constraint)

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
        if adq_length_prior is not None:
            self.register_prior(
                "adq_length_prior",
                adq_length_prior,
                lambda m: m.adq_length,
                lambda m, v : m._set_adq_length(v),
            )


    # now set up the 'actual' paramter
    @property
    def length(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_length_constraint.transform(self.raw_length)

    @property
    def adq_length(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_adq_length_constraint.transform(self.raw_adq_length)

    @length.setter
    def length(self, value):
        return self._set_length(value)

    @adq_length.setter
    def adq(self, value):
        return self._set_adq_length(value)

    def _set_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_length=self.raw_length_constraint.inverse_transform(value))

    def _set_adq_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_adq_length)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_adq_length=self.raw_adq_length_constraint.inverse_transform(value))

    # this is the kernel function
    def forward(self, x1, x2, idx1=None,idx2=None,square_dist=True, diag=False,**params):
        # apply lengthscale
        if idx2 is None:
            idx2 = idx1
        x1_ = x1.div(self.length)
        x2_ = x2.div(self.length)
        # calculate the distance between inputs
        # Below the s2i and s2j are already treated as variances
        #a2i, a2j = torch.meshgrid(self.adq[idx1].flatten(), self.adq[idx2].flatten())
        #a2i_times_a2j = a2i*a2j  #Multiplication of linear combination
        idx1_ = torch.Tensor(idx1)[:,None].div(self.adq_length)
        idx2_ = torch.Tensor(idx2)[:,None].div(self.adq_length)
        a2i_times_a2j = -1.0 * self.covar_dist(idx1_, idx2_, square_dist=square_dist, diag=diag, **params)
        #a2i_times_a2j.where(a2i_times_a2j == 0, torch.as_tensor(1e-20))
        # calculate the distance between inputs
        diff = -1.0 * self.covar_dist(x1_, x2_, square_dist=square_dist, diag=diag, **params)
        # prevent divide by 0 errors
        #diff.where(diff == 0, torch.as_tensor(1e-20))
        return (a2i_times_a2j.exp_())*(diff.exp_())

class NNetwork_kern(gpytorch.kernels.Kernel):
    # the kernel is stationary
    is_stationary = False

    # We will register the parameter when initializing the kernel
    def __init__(self, sig0_prior=None,sig0_constraint=None,sig_prior=None, sig_constraint=None, **kwargs):
        super().__init__(**kwargs)

        # set the parameter constraint to be positive, when nothing is specified
        if sig0_constraint is None:
            sig0_constraint = Interval(10, 100) #Positive()10-100

        if sig_constraint is None:
            sig_constraint = Interval(0, 1) #Positive() 0-1.5

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

class Kernel_Sig2Constrained(gpytorch.kernels.Kernel):
    # the kernel is stationary
    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self, sig2_prior=None,sig2_constraint=None,length_prior=None, length_constraint=None, **kwargs):
        super().__init__(**kwargs)

        # set the parameter constraint to be positive, when nothing is specified
        if length_constraint is None:
            length_constraint = Interval(1.5, 100) #Positive()

        if sig2_constraint is None:
            sig2_constraint = Interval(-1000, 1000)

        # register the raw parameter
        self.register_parameter(
            name='raw_length', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_parameter(
            name='raw_sig2', parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1, 1))
        )
        # register the constraint
        self.register_constraint("raw_length", length_constraint)
        self.register_constraint("raw_sig2", sig2_constraint)

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
        if sig2_prior is not None:
            self.register_prior(
                "sig2_prior",
                sig2_prior,
                lambda m: m.sig2,
                lambda m, v : m._set_sig2(v),
            )

    # now set up the 'actual' paramter
    @property
    def length(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_length_constraint.transform(self.raw_length)

    @property
    def sig2(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_sig2_constraint.transform(self.raw_sig2)

    @length.setter
    def length(self, value):
        return self._set_length(value)

    @sig2.setter
    def sig2(self, value):
        return self._set_sig2(value)

    def _set_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_length=self.raw_length_constraint.inverse_transform(value))

    def _set_sig2(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sig2)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_sig2=self.raw_sig2_constraint.inverse_transform(value))

    # this is the kernel function
    def forward(self, x1, x2, idx1=None,idx2=None,square_dist=True, diag=False,**params):
        # apply lengthscale
        if idx2 is None:
            idx2 = idx1

        # apply lengthscale
        x1_ = x1.div(self.length)
        x2_ = x2.div(self.length)
        # calculate the distance between inputs
        diff = self.covar_dist(x1_, x2_, square_dist=square_dist, diag=diag, **params)
        diff = -0.5*diff
        # prevent divide by 0 errors
        #diff.where(diff == 0, torch.as_tensor(1e-20))
        var = 0.005 / (1.0+torch.exp(-self.sig2)) #0.005
        return var*(diff.exp_())
