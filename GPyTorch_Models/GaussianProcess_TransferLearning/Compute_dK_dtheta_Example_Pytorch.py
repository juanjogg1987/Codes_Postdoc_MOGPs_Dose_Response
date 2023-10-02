import torch

# Define your custom covariance function (e.g., Gaussian kernel)
def k(x, y, length_scale):
    return torch.exp(-0.5 * ((x - y) / length_scale)**2)

# Create input data (N is the number of data points)
N = 100
x = torch.rand(N)  # Replace with your actual data
length_scale = torch.tensor(1.0, requires_grad=True)  # Initialize the length-scale with 1.0

# Initialize the covariance matrix K
K = torch.zeros(N, N)

# Compute the elements of the covariance matrix
for i in range(N):
    for j in range(N):
        K[i, j] = k(x[i], x[j], length_scale)

# Compute the derivative of K with respect to the length-scale
K.backward(torch.ones_like(K), retain_graph=True)

# Access the derivative matrix
dK_dlength_scale = length_scale.grad