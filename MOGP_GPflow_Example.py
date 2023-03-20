import numpy as np
import matplotlib.pyplot as plt
import gpflow as gpf
import tensorflow as tf

from gpflow.utilities import print_summary
from gpflow.ci_utils import ci_niter

gpf.config.set_default_float(np.float64)
gpf.config.set_default_summary_fmt("notebook")
np.random.seed(0)
#%matplotlib inline

"Generate synthetic data"

MAXITER = ci_niter(1000)

N = 1000  # number of points
D = 2  # number of input dimensions
M = 15 #15  # number of inducing points
L = 2  # number of latent GPs
P = 3  # number of observations = output dimensions

def generate_data(N=100):
    X = np.random.rand(N)[:, None] * 10 - 5  # Inputs = N x D
    G = np.hstack((0.5 * np.sin(3 * X) + X, 3.0 * np.cos(X) - X))  # G = N x L
    W = np.array([[0.5, -0.3, 1.5], [-0.4, 0.43, 0.0]])  # L x P
    F = np.matmul(G, W)  # N x P
    Y = F + np.random.randn(*F.shape) * [0.2, 0.2, 0.2]

    return X, Y

X, Y = data = generate_data(N)
Zinit = np.linspace(-5, 5, M)[:, None]

def plot_model(m, lower=-8.0, upper=8.0):
    pX = np.linspace(lower, upper, 100)[:, None]
    pY, pYv = m.predict_y(pX)
    if pY.ndim == 3:
        pY = pY[:, 0, :]
    plt.plot(X, Y, "x")
    plt.gca().set_prop_cycle(None)
    plt.plot(pX, pY)
    for i in range(pY.shape[1]):
        top = pY[:, i] + 2.0 * pYv[:, i] ** 0.5
        bot = pY[:, i] - 2.0 * pYv[:, i] ** 0.5
        plt.fill_between(pX[:, 0], top, bot, alpha=0.3)
    plt.xlabel("X")
    plt.ylabel("f")
    plt.title(f"ELBO: {m.elbo(data):.3}")
    plt.plot(Z, Z * 0.0, "o")

# # create multi-output kernel
# kernel = gpf.kernels.SharedIndependent(gpf.kernels.SquaredExponential() + gpf.kernels.Linear(), output_dim=P)
# # initialization of inducing input locations (M random points from the training inputs)
# Z = Zinit.copy()
# # create multi-output inducing variables from Z
# iv = gpf.inducing_variables.SharedIndependentInducingVariables(gpf.inducing_variables.InducingPoints(Z))
#
# # create SVGP model as usual and optimize
# m = gpf.models.SVGP(kernel, gpf.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=P)
# print_summary(m)

def optimize_model_with_scipy(model):
    optimizer = gpf.optimizers.Scipy()
    optimizer.minimize(
        model.training_loss_closure(data),
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": MAXITER},
    )


#optimize_model_with_scipy(m)
#print_summary(m)

# Plot predictions and observations
#plot_model(m)

#print_summary(m.kernel)
#print(m.kernel.kernel.kernels[0].lengthscales)

# Create list of kernels for each output
kern_list = [gpf.kernels.SquaredExponential() + gpf.kernels.Linear() for _ in range(L)]
# Create multi-output kernel from kernel list
kernel = gpf.kernels.LinearCoregionalization(kern_list, W=np.random.randn(P, L))  # Notice that we initialise the mixing matrix W
# initialisation of inducing input locations (M random points from the training inputs)
Z = Zinit.copy()
# create multi-output inducing variables from Z
iv = gpf.inducing_variables.SharedIndependentInducingVariables(gpf.inducing_variables.InducingPoints(Z))

# initialize mean of variational posterior to be of shape MxL
q_mu = np.zeros((M, L))
# initialize \sqrt(Σ) of variational posterior to be of shape LxMxM
q_sqrt = np.repeat(np.eye(M)[None, ...], L, axis=0) * 1.0

# create SVGP model as usual and optimize
m = gpf.models.SVGP(kernel, gpf.likelihoods.Gaussian(1.0), inducing_variable=iv, q_mu=q_mu, q_sqrt=q_sqrt)
gpf.set_trainable(m.inducing_variable, False)
gpf.set_trainable(m.likelihood, False)
optimize_model_with_scipy(m)

plot_model(m)





