from algorithms import QR_GS1, QR_GS2, QR_GS2_reorth, QR_Householder
import matplotlib.pyplot as plt
import numpy as np
'''
Generates mat_per_dim square random matrices for each dimension in dim
and compares any two QR algorithms by plotting mean errors in
orthogonality and reconstruction, with standard deviation shown.
'''
dim = [2, 4, 8, 16, 32, 64, 128, 256]
mat_per_dim = 10
machine_eps = np.finfo(np.float64).eps

err1_means = []
err2_means = []
err1_stds = []
err2_stds = []

for d in dim:
    err1_all = []
    err2_all = []
    for _ in range(mat_per_dim):
        A = 1000 * np.random.rand(d, d)
        Q1, R1 = QR_GS2(A)
        Q2, R2 = np.linalg.qr(A)
        Id = np.identity(d)
## Change two lines below with commented ones to switch error type
##        err1 = np.max(np.abs(np.dot(Q1.T, Q1) - Id)) / machine_eps
##        err2 = np.max(np.abs(np.dot(Q2.T, Q2) - Id)) / machine_eps
        err1 = np.max(np.abs( A - np.dot(Q1,R1) ))/machine_eps
        err2 = np.max(np.abs( A - np.dot(Q2,R2) ))/machine_eps
        err1_all.append(err1)
        err2_all.append(err2)
    
    err1_all = np.array(err1_all)
    err2_all = np.array(err2_all)
    
    err1_means.append( np.mean(err1_all) )
    err2_means.append( np.mean(err2_all) )
    err1_stds.append( np.std(err1_all) )
    err2_stds.append( np.std(err2_all) )

dim = np.array(dim)
err1_means = np.array(err1_means)
err2_means = np.array(err2_means)
err1_stds = np.array(err1_stds)
err2_stds = np.array(err2_stds)

plt.plot(dim, err1_means, 'o-', color='blue', label='GS2')
plt.plot(dim, err2_means, 'o-', color='magenta', label='np.linalg.qr')

plt.fill_between(dim, err1_means - err1_stds, err1_means + err1_stds, color='blue', alpha=0.2)
plt.fill_between(dim, err2_means - err2_stds, err2_means + err2_stds, color='magenta', alpha=0.2)

plt.grid()
plt.xlabel('Dimension')
##plt.ylabel('Error in orthogonality, eps')
plt.ylabel('Error in reconstruction, eps')
plt.legend()
plt.savefig('GS2 vs np.linalg.qr Reconst.png')
plt.show()
