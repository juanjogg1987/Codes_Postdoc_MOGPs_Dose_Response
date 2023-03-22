import GPy
import numpy as np
from matplotlib import pyplot as plt

import sys
#plt.close('all')
sys.path.append('code')
#import GP_varsel as varsel

from scipy.linalg import cholesky, cho_solve
from GPy.util import linalg

which_case = 'case4'
Niter = 0

# Kullback-Leibler Divergence between Gaussian distributions
def KLD_Gaussian(m1, V1, m2, V2, use_diag=False):
    Dim = m1.shape[0]
    # Cholesky decomposition of the covariance matrix
    if use_diag:
        L1 = np.diag(np.sqrt(np.diag(V1)))
        L2 = np.diag(np.sqrt(np.diag(V2)))
        V2_inv = np.diag(1.0 / np.diag(V2))
    else:
        L1 = cholesky(V1, lower=True)
        L2 = cholesky(V2, lower=True)
        V2_inv, _ = linalg.dpotri(np.asfortranarray(L2))
        # V2_inv = np.linalg.inv(V2)
    # print(V2_inv)

    KL = 0.5 * np.sum(V2_inv * V1) + 0.5 * np.dot((m1 - m2).T, np.dot(V2_inv, (m1 - m2))) \
         - 0.5 * Dim + 0.5 * 2. * np.sum(np.log(np.abs(np.diag(L2)))) - 0.5 * 2. * np.sum(np.log(np.abs(np.diag(L1))))
    return np.abs(KL)  # This is to avoid any negative due to numerical instability


def KLD_Gaussian_NoChol(m1, V1, m2, V2, use_diag=False):
    Dim = m1.shape[0]
    # print("shape m1", m1.shape)
    # Cholesky decomposition of the covariance matrix
    if use_diag:
        V1 = np.diag(np.sqrt(np.diag(V1)))
        V2 = np.diag(np.sqrt(np.diag(V2)))
        V2_inv = np.diag(1.0 / np.diag(V2))
    else:
        # V2_inv, _  = linalg.dpotri(np.asfortranarray(L2))
        V2_inv = np.linalg.inv(V2)
    KL = 0.5 * np.sum(V2_inv * V1) + 0.5 * np.dot((m1 - m2).T, np.dot(V2_inv, (m1 - m2))) \
         - 0.5 * Dim + 0.5 * np.log(np.linalg.det(V2)) - 0.5 * np.log(np.linalg.det(V1))
    return KL  # This is to avoid any negative due to numerical instability


def KLRel_Juan(train_x, model, delta,use_diag = False):
    N, P = train_x.shape  # Do we need to merge the train with val in here?
    relevance = np.zeros((N, P))
    # delta = 1.0e-4
    jitter = 1.0e-15
    # which_p = int(config.feature)
    print(f"Analysing {P} Features...")
    for p in range(P):
        for n in range(N):
            # x_plus = X[n,:].copy()
            x_plus = train_x[n:n + 1, :].clone()
            x_minus = train_x[n:n + 1, :].clone()
            x_plus[0, p] = x_plus[0, p] + delta
            x_minus[0, p] = x_minus[0, p] - delta

            # Make predictions
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # test_x = torch.linspace(0, 1, 51)
                predict_xn = likelihood(model(train_x[n:n + 1, :]))
                predict_xn_delta = likelihood(model(x_plus))
                predict_xn_delta_min = likelihood(model(x_minus))

                m1 = predict_xn.mean  # np.ones((D, 1))  # The dimension of this is related to the number of Outputs D
                m2 = predict_xn_delta.mean  # np.ones((D, 1))  # The dimension of this is related to the number of Outputs D
                m2_minus = predict_xn_delta_min.mean  # np.ones((D, 1))  # The dimension of this is related to the number of Outputs D
                # np.random.seed(1)
                #use_diag = False

                if use_diag:
                    V1 = np.diag(predict_xn.variance.numpy()[0,:])  # np.dot(A.T, A)  # The dimension of this is related to the number of Outputs D
                    V2 = np.diag(predict_xn_delta.variance.numpy()[0,:])  # The dimension of this is related to the number of Outputs D
                    V2_minus = np.diag(predict_xn_delta_min.variance.numpy()[0,:])  # The dimension of this is related to the number of Outputs D
                else:
                    V1 = predict_xn.covariance_matrix.numpy()  # np.dot(A.T, A)  # The dimension of this is related to the number of Outputs D
                    V2 = predict_xn_delta.covariance_matrix.numpy()  # The dimension of this is related to the number of Outputs D
                    V2_minus = predict_xn_delta_min.covariance_matrix.numpy()  # The dimension of this is related to the number of Outputs D
            KL_plus = np.sqrt(KLD_Gaussian(m1.numpy().T, V1, m2.numpy().T, V2,use_diag=use_diag) + jitter)  # In code the authors don't use the Mult. by 2
            KL_minus = np.sqrt(KLD_Gaussian(m1.numpy().T, V1, m2_minus.numpy().T, V2_minus,use_diag=use_diag) + jitter)  # In code the authors don't use the Mult. by 2
            relevance[n, p] = 0.5 * (KL_plus + KL_minus) / delta
    return np.mean(relevance, 0)

np.random.seed(1)

# number of repetitions to average over
repeats = 3# 1*3
# number of covariates
m = 8
# number of data points
n = 200
# Delta for KL method
delta = 0.0001
# number of quadrature points for VAR method
nquadr = 11

# if x are uniformly distributed
# compute the analytical scaling coefficients for the m components
phi = np.pi * np.linspace(0.1, 1, m);
Aunif = np.zeros(m)
for i in range(0, m):
    Aunif[i] = np.sqrt(4 / (2 - 1 / phi[i] * np.sin(2 * phi[i])))

# if x are normally distributed
# compute the scaling coefficients for the m components
nquadr2 = 11
[points, weights] = np.polynomial.hermite.hermgauss(nquadr2)
std = 0.4
Anor = np.zeros(m)
for i in range(0, m):
    coeff = np.sin(phi[i] * np.sqrt(2) * std * points) * np.sin(phi[i] * np.sqrt(2) * std * points)
    var = 1.0 / np.sqrt(np.pi) * np.dot(coeff.T, weights)
    Anor[i] = 1 / np.sqrt(var)

rel_ARD = np.zeros(m)
rel_KL = np.zeros(m)
rel_KL_juan = np.zeros(m)
rel_VAR = np.zeros(m)

Repeats_KL_juan_IndMOGP = []
Repeats_KL_juan_MOGP = []
Repeats_KL_juan_MOGP_diag = []
Repeats_KL_ARD = []
Repeats_Lasso = []

Dtasks = 2

for i in range(0, repeats):
    if i < repeats/3:
        which_model = 'IndMOGP'
    elif i < 2*repeats/3:
        which_model = 'MOGP'
    else:
        which_model = 'Lasso'
    print("Seed used:",i%(repeats//3))
    np.random.seed(i%(repeats//3))
    x = np.random.uniform(-1., 1., (n, m))
    # x = np.random.normal(0.0,std,(n,m))
    print('starting repetition ', i + 1, '/', repeats)
    phi = np.tile(np.linspace(np.pi / 10, np.pi, m), (n, 1))
    xphi = np.multiply(x[:, 0:m], phi)
    f1 = np.sin(xphi)
    f2 = np.sin(xphi)
    Aunif = 1.0/f1.std(0)
    #Aunif[1] = Aunif[1]*2.0
    # Anor[0] = Anor[0] + 0.5
    for j in range(0, m):
        if which_case =="case1":
            f1[:, j] = f1[:, j] * Aunif[j]
            # f[:,j] = f[:,j]*Anor[j]
            f2[:, j] = f2[:, j] * Aunif[j]*((j**2)*1/49+0.1)
        elif which_case =="case2":
            f1[:, j] = f1[:, j] * Aunif[j] * ((j ** 2)*1/49+0.1)  # *(1.7-j*0.1)
            # f[:,j] = f[:,j]*Anor[j]
            f2[:, j] = f2[:, j] * Aunif[j] * ((j ** 1.5) * 1/(7**1.5) + 0.1)
        elif which_case =="case3":
            f1[:, j] = f1[:, j] * Aunif[j]*0.1
            f2[:, j] = f2[:, j] * Aunif[j]*0.1# * ((j ** 2) * 0.1 + 0.1)
            if j==0:
                f1[:, j] = f1[:, j] * 10#(7**2)*1
                #f2[:, j] = f2[:, j] * (7 ** 2) * 0.1
            if j==3:
                #f1[:, j] = f1[:, j] * (7**2)*0.1
                f2[:, j] = f2[:, j] * 10#(7**2)*1
        elif which_case =="case4":
            f1[:, j] = f1[:, j] * Aunif[j]
            f2[:, j] = f2[:, j] * Aunif[j]# * ((j ** 2) * 0.1 + 0.1)
            if j<4:
                f1[:, j] = f1[:, j] * 0.1#(7**2)*1
                #f2[:, j] = f2[:, j] * (7 ** 2) * 0.1
            else:
                #f1[:, j] = f1[:, j] * (7**2)*0.1
                f2[:, j] = f2[:, j] * 0.1#(7**2)*1
        # if j==6:
        #     f1[:, j] = f1[:, j] * 0
        #     f2[:, j] = f2[:, j] * 0
        # f[:,j] = f[:,j]*Anor[j]
    #break
    # f = f[:, ::-1].copy()
    # f = f/f.std(0)
    # y is a sum of the m components plus Gaussian noise
    yval1 = f1.sum(axis=1) + np.random.normal(0, 0.3, (n,))
    yval2 = f2.sum(axis=1) + np.random.normal(0, 0.3, (n,))
    Yall = np.concatenate((np.asmatrix(yval1),np.asmatrix(yval2))).T

    #Yall = F + 0.01 * np.random.randn(x1.shape[0], Dtasks)

    # standard_Y = False
    # if standard_Y:
    #     Yall = (Yall - np.mean(Yall, 0)) / np.std(Yall, 0)
    #     F = (F - np.mean(F, 0)) / np.std(F, 0)

    Ntrain = n  # //3
    Xall = x.copy()
    Xtrain = Xall[0:Ntrain, :].copy()
    Ytrain = Yall[0:Ntrain, :].copy()
    # Yclean = F[index_N[0:Ntrain],:].copy()

    # Xtrain = Xall[index_N[Ntrain:2*Ntrain],:].copy()
    # Ytrain = Yall[index_N[Ntrain:2*Ntrain],:].copy()
    # Yclean = F[index_N[Ntrain:2*Ntrain],:].copy()

    Xval = Xall[0:Ntrain, :].copy()
    Yval = Yall[0:Ntrain, :].copy()
    #Yclean = F[0:Ntrain, :].copy()

    # # RBF kernel plus constant term
    # kernel = GPy.kern.RBF(input_dim=m, ARD=True) + GPy.kern.Bias(input_dim=m)
    # model = GPy.models.GPRegression(x, y, kernel)
    # model.optimize()

    # # ARD relevance value is the inverse of the length scale
    # rel_ARD = rel_ARD + 1 / model.sum.rbf.lengthscale
    #
    # # KL relevance value
    # rel_KL = rel_KL + varsel.KLrel(x, model, delta)
    #
    # rel_KL_juan = rel_KL_juan + KLRel_Juan(x, model, delta)
    #
    # # VAR relevance value
    # rel_VAR = rel_VAR + varsel.VARrel(x, model, nquadr)

    import math
    import torch
    import gpytorch
    from matplotlib import pyplot as plt

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "SEED"
    torch.manual_seed(23)  # 23 for Ind  #use 49 for MOGP ICM3
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    import os

    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    rank = Ytrain.shape[1]  # Rank for the MultitaskKernel, we make rank equal to number of tasks
    num_tasks = Ytrain.shape[1]
    Dim = Xtrain.shape[1]
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    train_x = torch.from_numpy(Xtrain.astype(np.float32))
    train_y = torch.from_numpy(Ytrain.astype(np.float32))
    val_x = torch.from_numpy(Xval.astype(np.float32))
    val_y = torch.from_numpy(Yval.astype(np.float32))
    # val_x = torch.from_numpy(Xtrain.astype(np.float32))
    # val_y = torch.from_numpy(Ytrain.astype(np.float32))

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    myseed = 15  # int(config.which_seed)
    np.random.seed(myseed)


    class MultitaskGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=num_tasks
            )

            size_dims = Dim
            # mykern = gpytorch.kernels.MaternKernel(nu=0.5,ard_num_dims=size_dims)
            mykern = gpytorch.kernels.RBFKernel(ard_num_dims=size_dims)
            # mykern = gpytorch.kernels.RBFKernel()

            self.covar_module = gpytorch.kernels.MultitaskKernel(
                mykern, num_tasks=num_tasks, rank=rank
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    Q = 4
    num_latents = Q


    class MultitaskGPModelLMC(gpytorch.models.ApproximateGP):
        def __init__(self):
            # Let's use a different set of inducing points for each latent function
            # inducing_points = torch.rand(num_latents, 30, Dim)
            inducing_points = torch.from_numpy(
                np.repeat(Xtrain[None, np.random.permutation(Ntrain)[0:50], :], num_latents, 0).astype(np.float32))

            # We have to mark the CholeskyVariationalDistribution as batch
            # so that we learn a variational distribution for each task
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_latents])
            )

            # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
            # so that the output will be a MultitaskMultivariateNormal rather than a batch output
            variational_strategy = gpytorch.variational.LMCVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                ),
                num_tasks=Dtasks,
                num_latents=num_latents,
                latent_dim=-1
            )

            super().__init__(variational_strategy)

            # The mean and covariance modules should be marked as batch
            # so we learn a different set of hyperparameters
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
                batch_shape=torch.Size([num_latents])
            )

        def forward(self, x):
            # The forward function should be written as if we were dealing with each output
            # dimension in batch
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
            size_dims = Dim
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=size_dims, batch_shape=torch.Size([num_tasks])),
                batch_shape=torch.Size([num_tasks])
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            )


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks,
                                                                  noise_constraint=gpytorch.constraints.Interval(1.0e-6,
                                                                                                                 1.0e-1),
                                                                  rank=num_tasks)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    if which_model == "IndMOGP":
        print("Running IndMOGP...")
        model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood)
    elif which_model == "SpMOGP":
        print("Running SparseMOGP...")
        # model = MultitaskGPModel(train_x, train_y, likelihood)
        model = MultitaskGPModelLMC()
    elif which_model=="MOGP":
        print("Running MOGP...")
        # model = MultitaskGPModel(train_x, train_y, likelihood)
        model = MultitaskGPModel(train_x, train_y, likelihood)
    else:
        print("Running Lasso...")
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    if not(which_model=="Lasso"):
        import os

        num_epochs = Niter  # int(config.N_iter_epoch)

        model.train()
        likelihood.train()

        if which_model == "SpMOGP":
            # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': likelihood.parameters()},
            ], lr=0.005)
        else:
            # Our loss object We're using the Exact Marginal Log Likelohood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        Ntrain, _ = Ytrain.shape
        show_each = 100  # Ntrain//train_loader.batch_size
        # refine_lr = [0.01,0.005,0.001,0.0005]
        refine_lr = [0.007, 0.005, 0.001, 0.0005]
        # refine_lr = [0.005,0.001,0.0005,0.0001]

        if which_model == "SpMOGP":
            refine_lr = [0.03, 0.007, 0.003, 0.001]

        refine_num_epochs = [num_epochs, int(num_epochs * 0.5), int(num_epochs * 0.2), int(num_epochs * 0.2)]
        for Nrefine in range(len(refine_lr)):
            print(f"\nRefine Learning Rate {Nrefine}; lr={refine_lr[Nrefine]}")
            for g in optimizer.param_groups:
                g['lr'] = refine_lr[Nrefine]

            for j in range(refine_num_epochs[Nrefine]):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                if j % (show_each) == 0:
                    print(f"Iter {j}, Loss {loss}")

                loss.backward()
                optimizer.step()
            # break
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # Set into eval mode
        model.eval()
        likelihood.eval()

        # Initialize plots
        # fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # test_x = torch.linspace(0, 1, 51)
            predictions = likelihood(model(val_x))
            mean = predictions.mean
            # lower, upper = predictions.confidence_region()
            lower = mean - 2 * torch.sqrt(predictions.variance)
            upper = mean + 2 * torch.sqrt(predictions.variance)

        for task in range(num_tasks):
            # Plot training data as black stars
            plt.figure(task + 1)
            plt.plot(val_y[:, task].detach().numpy(), 'k.')
            #plt.plot(Yclean[:, task], '-r')
            # Predictive mean as blue line
            plt.plot(mean[:, task].numpy(), '.b')
            # Shade in confidence
            plt.plot(lower[:, task].numpy(), 'c--')
            plt.plot(upper[:, task].numpy(), 'c--')
            # ax.fill_between(lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
            # plt.ylim([-0.1, 1.3])
            plt.legend(['Observed Data', 'y', 'mean_pred', '2std'])
            plt.title(f'Task {task + 1} {which_model}')

        print("MSE:", np.mean((mean.numpy() - val_y.numpy()) ** 2))
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    else:
        from sklearn import datasets
        from sklearn.linear_model import Lasso
        from sklearn.model_selection import GridSearchCV

        lasso = Lasso(random_state=0, max_iter=10000)
        alphas = np.logspace(-4, -0.5, 50)

        tuned_parameters = [{"alpha": alphas}]
        n_folds = 5

        clf_CV = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
        clf_CV.fit(Xtrain, Ytrain)

        clf = Lasso(alpha=clf_CV.best_params_['alpha'])  # alpha=0.000002 alpha=0.01
        clf.fit(Xtrain, Ytrain)
        print(clf.coef_)

    if which_model == 'MOGP':
        rel_KL_juan_MOGP = KLRel_Juan(train_x, model, delta)
        rel_KL_juan_norm_MOGP = rel_KL_juan_MOGP / np.max(rel_KL_juan_MOGP)
        Repeats_KL_juan_MOGP.append(rel_KL_juan_norm_MOGP.copy())

        rel_KL_juan_MOGP_diag = KLRel_Juan(train_x, model, delta,use_diag=True)
        rel_KL_juan_norm_MOGP_diag = rel_KL_juan_MOGP_diag / np.max(rel_KL_juan_MOGP_diag)
        Repeats_KL_juan_MOGP_diag.append(rel_KL_juan_norm_MOGP_diag.copy())
    elif which_model== 'IndMOGP':
        rel_KL_juan_IndMOGP = KLRel_Juan(train_x, model, delta)
        rel_KL_juan_norm_IndMOGP = rel_KL_juan_IndMOGP / np.max(rel_KL_juan_IndMOGP)
        Repeats_KL_juan_IndMOGP.append(rel_KL_juan_norm_IndMOGP.copy())

        ARD = model.covar_module.base_kernel.lengthscale.detach().numpy()
        rel_KL_ARD = (1.0 / ARD[1].flatten() + 1 / ARD[0].flatten()) / 2.0
        rel_KL_ARD_norm = rel_KL_ARD / np.max(rel_KL_ARD)
        Repeats_KL_ARD.append(rel_KL_ARD_norm.copy())
    else:
        rel_Lasso = clf.coef_.mean(0)
        rel_Lasso_norm = rel_Lasso/np.max(rel_Lasso)
        Repeats_Lasso.append(rel_Lasso_norm.copy())

"Load with lines below setting the name case1, case2, case3 and case4"
Repeats_KL_juan_MOGP,Repeats_KL_juan_MOGP_diag,Repeats_KL_juan_IndMOGP,Repeats_KL_ARD,Repeats_Lasso=np.load(which_case+'.npy')

# # True relevance, the covariates are equally relevant in the L2 sense
rel_true = np.ones(m)

# plot
covariates = np.arange(1, m + 1)
fig2 = plt.figure(0)
ax2 = plt.subplot(111)

style_plot_m = '-oc'
style_plot_std = '--c'
ax2.plot(covariates, np.mean(Repeats_KL_juan_MOGP, 0), style_plot_m, label='KL_MOGP')
ax2.fill_between(covariates,np.mean(Repeats_KL_juan_MOGP, 0) - np.std(Repeats_KL_juan_MOGP, 0),np.mean(Repeats_KL_juan_MOGP, 0) + np.std(Repeats_KL_juan_MOGP, 0),alpha=0.2,color='c')
#ax2.plot(covariates, np.mean(Repeats_KL_juan_MOGP, 0) + np.std(Repeats_KL_juan_MOGP, 0), style_plot_std)
#ax2.plot(covariates, np.mean(Repeats_KL_juan_MOGP, 0) - np.std(Repeats_KL_juan_MOGP, 0), style_plot_std)
style_plot_m = '-om'
style_plot_std = '--m'
ax2.plot(covariates, np.mean(Repeats_KL_juan_MOGP_diag, 0), style_plot_m, label='KL_MOGP_diag')
ax2.fill_between(covariates,np.mean(Repeats_KL_juan_MOGP_diag, 0) - np.std(Repeats_KL_juan_MOGP_diag, 0),np.mean(Repeats_KL_juan_MOGP_diag, 0) + np.std(Repeats_KL_juan_MOGP_diag, 0),alpha=0.2,color='m')
#ax2.plot(covariates, np.mean(Repeats_KL_juan_MOGP_diag, 0) + np.std(Repeats_KL_juan_MOGP_diag, 0), style_plot_std)
#ax2.plot(covariates, np.mean(Repeats_KL_juan_MOGP_diag, 0) - np.std(Repeats_KL_juan_MOGP_diag, 0), style_plot_std)
style_plot_m = '-or'
style_plot_std = '--r'
ax2.plot(covariates, np.mean(Repeats_KL_juan_IndMOGP, 0), style_plot_m, label='KL_IndGPs')
ax2.fill_between(covariates,np.mean(Repeats_KL_juan_IndMOGP, 0) - np.std(Repeats_KL_juan_IndMOGP, 0),np.mean(Repeats_KL_juan_IndMOGP, 0) + np.std(Repeats_KL_juan_IndMOGP, 0),alpha=0.2,color='r')
#ax2.plot(covariates, np.mean(Repeats_KL_juan_IndMOGP, 0) + np.std(Repeats_KL_juan_IndMOGP, 0), style_plot_std)
#ax2.plot(covariates, np.mean(Repeats_KL_juan_IndMOGP, 0) - np.std(Repeats_KL_juan_IndMOGP, 0), style_plot_std)
style_plot_m = '-ob'
style_plot_std = '--b'
ax2.plot(covariates, np.mean(Repeats_KL_ARD, 0).flatten(), style_plot_m, label='ARD_IndGPs')
ax2.fill_between(covariates,np.mean(Repeats_KL_ARD, 0) - np.std(Repeats_KL_ARD, 0),np.mean(Repeats_KL_ARD, 0) + np.std(Repeats_KL_ARD, 0),alpha=0.2,color='b')
#ax2.plot(covariates, np.mean(Repeats_KL_ARD, 0).flatten() + np.std(Repeats_KL_ARD, 0).flatten(), style_plot_std)
#ax2.plot(covariates, np.mean(Repeats_KL_ARD, 0).flatten() - np.std(Repeats_KL_ARD, 0).flatten(), style_plot_std)
style_plot_m = '-og'
style_plot_std = '--g'
ax2.plot(covariates, np.mean(Repeats_Lasso, 0).flatten(), style_plot_m, label='Lasso')
ax2.fill_between(covariates,np.mean(Repeats_Lasso, 0) - np.std(Repeats_Lasso, 0),np.mean(Repeats_Lasso, 0) + np.std(Repeats_Lasso, 0),alpha=0.2,color='g')
#ax2.plot(covariates, np.mean(Repeats_Lasso, 0).flatten() + np.std(Repeats_Lasso, 0).flatten(), style_plot_std)
#ax2.plot(covariates, np.mean(Repeats_Lasso, 0).flatten() - np.std(Repeats_Lasso, 0).flatten(), style_plot_std)

#if which_case=="case4":
#    ax2.plot(covariates, rel_true, '--ok', label='True')

rel_true = (f1.std(0)+f2.std(0))/2
rel_true = rel_true/rel_true.max()
ax2.plot(covariates, rel_true, '-ok', label='(std(f1)+std(f2))/max')

ax2.legend()
ax2.set_ylabel('relevance')
ax2.set_xlabel('input')
plt.ylim([0,1.1])
plt.show()
plt.figure(0)
plt.title(which_case)

# with open(which_case+'.npy', 'wb') as myfile:
#     np.save(myfile, [Repeats_KL_juan_MOGP,Repeats_KL_juan_MOGP_diag,Repeats_KL_juan_IndMOGP,Repeats_KL_ARD,Repeats_Lasso])

#"Load with lines below setting the name case1, case2, case3 and case4"
#Repeats_KL_juan_MOGP,Repeats_KL_juan_MOGP_diag,Repeats_KL_juan_IndMOGP,Repeats_KL_ARD=np.load('case1.npy')






