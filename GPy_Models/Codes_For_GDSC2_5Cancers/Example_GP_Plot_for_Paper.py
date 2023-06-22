import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Create some synthetic data
#np.random.seed(2)
#X = np.sort(1 * np.random.rand(10, 1), axis=0)
#y = (1.5*X.ravel())*np.sin(5*X).ravel() + 0.01*np.random.randn(X.shape[0]).ravel()
X = np.array([0.1,0.21,0.4,0.55,0.8])[:,None]
y = np.array([-1.5,0.6,1.2,2.3,-0.2])

# Define the kernel function
kernel = C(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-3, 1e3))

#kernel = RBF(0.1, (1e-3, 1e3))
# Create a Gaussian process regression model and fit it to the data
gp = GaussianProcessRegressor(kernel=kernel,alpha=1e-2,optimizer='fmin_l_bfgs_b', n_restarts_optimizer=20)
gp.fit(X, y)

# Predict on some new points
X_new = np.linspace(0, 1, 100).reshape(-1,1)
y_pred, cov_sigma = gp.predict(X_new, return_cov=True)

import matplotlib.pyplot as plt

for i in range(30):
    y_sampled = np.random.multivariate_normal(y_pred,cov_sigma)
    plt.plot(X_new, y_sampled,alpha=0.18,color='grey')
# Plot the results

plt.plot(X_new, y_pred, 'r-',linewidth = 2)
plt.plot(X, y, 'k.', markersize=7)

"Plot distributions over predicted value"
X_n = np.array([X[3]]).reshape(-1,1)
delta = 0.03
X_delt = X_n + delta
y_n, sigma_n = gp.predict(X_n, return_std=True)
y_delt, sigma_delt = gp.predict(X_delt, return_std=True)

from scipy.stats import norm
Npoints = 100
yn_lin = np.linspace(-3.8*sigma_n,3.8*sigma_n,Npoints)
yn_pdf = norm.pdf(yn_lin,loc=0,scale=sigma_n)
plt.plot(0.05*(yn_pdf)+X_n,yn_lin+y_n,'--',color='orangered',linewidth = 2)
plt.plot(X_n.ravel()*np.ones(20),y_n*np.linspace(-4.5,1,20),'|',color='orangered',linewidth = 2)
plt.plot(X_n.ravel(),y_n,'.',color='black',markersize=10)
plt.plot(X_n.ravel(),y_n,'.',color='orangered',markersize=8)

ydelt_lin = np.linspace(-2.8*sigma_delt,2.8*sigma_delt,Npoints)
ydelt_pdf = norm.pdf(ydelt_lin,loc=0,scale=sigma_delt)
plt.plot(0.05*(ydelt_pdf)+X_delt,ydelt_lin+y_delt,'--',color='green',linewidth = 2)
plt.plot(X_delt.ravel()*np.ones(21),y_delt*np.linspace(-4.5,1,21),'|',color='green',linewidth = 2)
plt.plot(X_delt.ravel(),y_delt,'.',color='green',markersize=10)
#plt.plot(yn_pdf.ravel(),X_n.ravel()*np.ones(Npoints), 'r-')
#plt.fill(np.concatenate([X_new, X_new[::-1]]),np.concatenate([y_pred - 1.96 * sigma,(y_pred + 1.96 * sigma)[::-1]]),
#         alpha=.5, fc='b', ec='None', label='95% confidence interval')

plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-4.5, 4.5)
plt.grid()
#plt.legend(loc='upper left')
plt.show()