import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Define the kernel (EQ kernel or RBF kernel)
kernel = 1.0 * RBF(length_scale=1.0)

# Define the range of x values
x = np.linspace(-5, 5, 100).reshape(-1, 1)  # 100 points from -5 to 5

# Initialize Gaussian Process Regressor with the defined kernel
gp = GaussianProcessRegressor(kernel=kernel)

# Generate samples from the Gaussian Process
samples = gp.sample_y(X=x, n_samples=5)

# Plot the samples
import matplotlib.pyplot as plt

for i in range(samples.shape[0]):
    plt.plot(x, samples[i], label=f'Sample {i+1}')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Samples from Gaussian Process with EQ Kernel')
plt.legend()
plt.show()