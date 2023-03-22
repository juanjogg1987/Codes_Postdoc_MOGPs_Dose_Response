
import GPy
import pods
import numpy as np

data = pods.datasets.olympic_sprints()
X = data['X']
y = data['Y']
print(data['info'], data['details'])
#GPy.kern.Coregionalize?

Ntasks = 6
kern = GPy.kern.RBF(1, lengthscale=80)**GPy.kern.Coregionalize(1,output_dim=Ntasks, rank=5)

model = GPy.models.GPRegression(X, y, kern)
model.optimize()

#m_pred,v_pred = model.predict(X[0:1,:],full_cov=True)
B = model.mul.coregion.B + np.eye(Ntasks) * model.mul.coregion.kappa

# K_X2X2 = model.kern.rbf.K(X2)
# b22K_inv = np.linalg.inv(B[1,1]*K_X2X2+np.eye(K_X2X2.shape[0])*model.likelihood.variance)
# K_x1X2 = model.kern.rbf.K(x1_new,X2)
# b12Kx1X2_K22_inv = np.dot(B[0,1]*K_x1X2, b22K_inv)
# m1_2 = np.dot(b12Kx1X2_K22_inv,y2)
# K_x1x1 = model.kern.rbf.K(x1_new)
# var1_2 = B[0,0]*K_x1x1+model.likelihood.variance-np.dot(b12Kx1X2_K22_inv,B[1,0]*K_x1X2.T)