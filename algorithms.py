import numpy as np
'''
Different QR-factorization algorithms
'''
# Unstable Gram-Schmidt
def QR_GS1(A):
    m = np.shape(A)[1]
    Q = np.copy(A.astype('float64'))
    R = np.zeros( (m,m) )
    for i in range(m):
        for j in range(-1,i-1): 
            R[j+1,i] = np.dot( A[:,i],Q[:,j+1] )
            Q[:,i] -= R[j+1,i]*Q[:,j+1]
        R[i,i] = np.linalg.norm(Q[:,i])
        Q[:,i] /= R[i,i]
    return Q,R
# Stable Gram-Schmidt
def QR_GS2(A):
    m = np.shape(A)[1]
    Q = np.copy(A.astype('float64'))
    R = np.zeros( (m,m) )
    for i in range(m):
        for j in range(-1,i-1): 
            R[j+1,i] = np.dot( Q[:,i],Q[:,j+1] )#differs from GS1 only in this line
            Q[:,i] -= R[j+1,i]*Q[:,j+1]
        R[i,i] = np.linalg.norm(Q[:,i])
        Q[:,i] /= R[i,i]
    return Q,R
# Attempt to add reorthogonalization to Gram-Schmidt
def QR_GS2_reorth(A):
    m = np.shape(A)[1]
    Q = np.copy(A.astype('float64'))
    R = np.zeros( (m,m) )
    for i in range(m):
        for j in range(-1,i-1): 
            R[j+1,i] = np.dot( Q[:,i],Q[:,j+1] )
            Q[:,i] -= R[j+1,i]*Q[:,j+1]
        R[i,i] = np.linalg.norm(Q[:,i])
        if np.linalg.norm(A[:,i])/np.linalg.norm(Q[:,i]) <= 10:
            Q[:,i] /= R[i,i]
        else:
            for j in range(-1,i-1):
                proj = np.dot(Q[:,i],Q[:,j+1])
                Q[:,i] -= proj*Q[:,j+1]
                R[i,i] = np.linalg.norm(Q[:,i])
                Q[:,i] /= R[i,i]
    return Q,R
# Householder algorithm. For details, see attached PDF.
def QR_Householder(A):
    n,m = A.shape
    R = np.copy(A.astype('float64'))
    Q = np.identity(n)
    for i in range(min(m,n)):
        x = R[i:,i]
        s = np.linalg.norm(x[1:])**2
        u = np.copy(x)
        u[0] = 1
        if s == 0 and x[0] >= 0:
            b = 0
        elif s == 0 and x[0] < 0:
            b = 2
        else:
            mu = np.sqrt( x[0]**2 + s )
            if x[0] <= 0:
                u[0] = x[0] - mu
            else:
                u[0] = -s/(x[0] + mu)
            b = 2/(s + u[0]**2)
        H = np.identity(n)
        H[i:,i:] -= b*np.dot( u.reshape(-1,1),u.reshape(1,len(u)) )
        R = np.dot(H,R)
        Q = np.dot(H,Q)
    return Q.T,R
