import numpy as np
from scipy.linalg import cholesky,cho_solve
from GPy.util import linalg

N = 3000
P = 825
D = 9
X = np.random.randn(N,P)
pred_covariance = np.ones((N,D,D))

#Kullback-Leibler Divergence between Gaussian distributions
def KLD_Gaussian(m1,V1,m2,V2):
    Dim = V1.shape[0]
    # Cholesky decomposition of the covariance matrix
    L1 = cholesky(V1, lower=True)
    L2 = cholesky(V2, lower=True)
    V2_inv, _  = linalg.dpotri(np.asfortranarray(L2))
    #V2_inv_check = np.linalg.inv(V2)
    KL = 0.5 * np.sum(V2_inv * V1) + 0.5 * np.dot((m1-m2).T, np.dot(V2_inv, (m1-m2))) \
         - 0.5 * Dim + 0.5 * 2. * np.sum(np.log(np.abs(np.diag(L2)))) - 0.5 * 2. * np.sum(np.log(np.abs(np.diag(L1))))
    return KL


m1 = np.ones((D,1))   #The dimension of this is related to the number of Outputs D
m2 = np.ones((D,1)) #The dimension of this is related to the number of Outputs D
np.random.seed(1)
A = np.random.rand(D,D)
B = np.random.rand(D,D)
#V1 = np.eye(3)
#V2 = np.eye(3)
V1 = np.dot(A.T,A)   #The dimension of this is related to the number of Outputs D
#V2 = np.dot(A.T,A)
V2 = np.dot(B.T,B)   #The dimension of this is related to the number of Outputs D

myKL = KLD_Gaussian(m1,V1,m2,V2)
print("KL div:",myKL)

relevance = np.zeros((N,P))
delta = 1.0e-4
x_plus = np.zeros((1,P))
for p in range(P):
    for n in range(N):
        x_plus = X[n,:].copy()
        x_plus[p] = x_plus[p]+delta

        m1 = np.ones((D, 1))  # The dimension of this is related to the number of Outputs D
        m2 = np.ones((D, 1))  # The dimension of this is related to the number of Outputs D
        np.random.seed(1)
        A = np.random.rand(D, D)
        B = np.random.rand(D, D)
        # V1 = np.eye(3)
        # V2 = np.eye(3)
        V1 = np.dot(A.T, A)  # The dimension of this is related to the number of Outputs D
        # V2 = np.dot(A.T,A)
        V2 = np.dot(B.T, B)  # The dimension of this is related to the number of Outputs D

        # m1 =  # mean prediction from MOGP using x_original_data
        # V1 =  # Covar prediction from MOGP using x_original_data
        # m2 = #mean prediction from MOGP using x_plus
        # V2 = #Covar prediction from MOGP using x_plus
        relevance[n,p] = np.sqrt(2.0*KLD_Gaussian(m1,V1,m2,V2))/delta #In code the authors don't use the Mult. by 2

print(relevance)
