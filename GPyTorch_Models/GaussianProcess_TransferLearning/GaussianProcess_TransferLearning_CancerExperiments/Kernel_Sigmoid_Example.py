import numpy as np
import matplotlib.pyplot as plt


def preprocess_data_1d(x, Sigma):
    # Preprocess 1-D input data
    return np.array([1, x]) * np.linalg.cholesky(Sigma).flatten()


def neumann_kernel_1d(x1, x2, Sigma):
    # Preprocess 1-D data
    x1_tilde = preprocess_data_1d(x1, Sigma)
    x2_tilde = preprocess_data_1d(x2, Sigma)

    # Compute inner product
    inner_product = np.dot(x1_tilde, x2_tilde)

    # Compute norms
    norm_x1_tilde_sq = np.dot(x1_tilde, x1_tilde)
    #print(x1_tilde)
    norm_x2_tilde_sq = np.dot(x2_tilde, x2_tilde)

    # Compute kernel value
    numerator = 2 * inner_product
    denominator = np.sqrt((1 + 2 * norm_x1_tilde_sq) * (1 + 2 * norm_x2_tilde_sq))
    kernel_value = np.arcsin(numerator / denominator) * (2 / np.pi)

    return kernel_value

# Define input space in 1-D
X1 = np.linspace(-5, 5, 100)[:, None]
X2 = np.linspace(-5, 5, 100)[:, None]

# Compute covariance matrix using Neumann kernel
Sigma = np.array([[0.6931]])**2  # Covariance matrix (1-D case)
K = np.zeros((len(X1), len(X2)))
for i, x1 in enumerate(X1):
    for j, x2 in enumerate(X2):
        K[i, j] = neumann_kernel_1d(x1, x2, Sigma)

# Perform Cholesky decomposition
L = np.linalg.cholesky(K + 1e-8 * np.eye(len(X1)))  # Adding jitter for numerical stability

# Generate samples from standard normal distribution
num_samples = 5
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