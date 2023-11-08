import numpy as np
from sklearn.gaussian_process.kernels import Kernel
import matplotlib.pyplot as plt

class ConstrainedSigmoidKernel(Kernel):
    def __init__(self, length_scale=1.0, constant_value=1.0):
        self.length_scale = length_scale
        self.constant_value = constant_value

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        if eval_gradient:
            raise ValueError("Gradient evaluation is not supported.")
        Z = np.tanh(np.dot(X / self.length_scale, (Y / self.length_scale).T))
        #K = 0.5 * self.constant_value * (Z + 1)
        return Z#K

    def diag(self, X):
        #return 0.5 * self.constant_value * np.ones(X.shape[0])
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True

# Example usage:
# Generate some example input data
X = np.sort(10*np.random.rand(100, 1), axis=0)
length_scale = 0.5
constant_value = 1.0

# Create the custom kernel
custom_kernel = ConstrainedSigmoidKernel(length_scale, constant_value)

# Compute the covariance matrix (kernel matrix)
K = custom_kernel(X)

# K now enforces the [0, 1] constraint

# Generate samples from the GP
y_samples = np.random.multivariate_normal(np.zeros(X.shape[0]), K, size=15).T

# Plot the samples
plt.figure(figsize=(8, 6))
for i in range(y_samples.shape[1]):
    plt.plot(X, y_samples[:, i], lw=2, label=f'Sample {i + 1}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Samples from GP with Modified Sigmoid Kernel (Constrained to [0, 1])')
plt.ylim(-1.5, 2.5)
plt.legend()
plt.show()