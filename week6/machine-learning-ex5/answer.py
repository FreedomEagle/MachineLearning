
import numpy as np
import matplotlib.pyplot as plt
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.optimize #fmin_cg to train the linear regression

dataFile = ('./ex5/ex5data1.mat')
data = scipy.io.loadmat(dataFile)
X = data['X']
Y = data['y']
Xcrossval = data['Xval']
Ycrossval = data['yval']
Xtest = data['Xtest']
Ytest = data['ytest']
X = np.insert(X,0,1.,axis=1)
Xcrossval = np.insert(Xcrossval,0,1,axis=1)
Xtest = np.insert(Xtest,0,1,axis=1)
myTheta = np.array([[1.],[1.]])



def h(theta,X): #Linear hypothesis function
    return np.dot(X,theta)

def computeCost(mytheta,myX,myy,mylambda=0.): #Cost function
    """
    theta_start is an n- dimensional vector of initial theta guess
    X is matrix with n- columns and m- rows
    y is a matrix with m- rows and 1 column
    """
    m = myX.shape[0]
    myh = h(mytheta,myX).reshape((m,1))
    mycost = float((1./(2*m)) * np.dot((myh-myy).T,(myh-myy)))
    regterm = (float(mylambda)/(2*m)) * float(mytheta[1:].T.dot(mytheta[1:]))
    return mycost + regterm

mytheta = np.array([[1.],[1.]])
print(computeCost(mytheta,X,Y,mylambda=1.))

def computeGradient(mytheta,myX,myy,mylambda=0.):
    mytheta = mytheta.reshape((mytheta.shape[0],1))
    m = myX.shape[0]
    #grad has same shape as myTheta (2x1)
    myh = h(mytheta,myX).reshape((m,1))
    grad = (1./float(m))*myX.T.dot(h(mytheta,myX)-myy)
    regterm = (float(mylambda)/m)*mytheta
    regterm[0] = 0 #don't regulate bias term
    regterm.reshape((grad.shape[0],1))
    return grad + regterm

#Here's a wrapper for computeGradient that flattens the output
#This is for the minimization routine that wants everything flattened
def computeGradientFlattened(mytheta,myX,myy,mylambda=0.):
    return computeGradient(mytheta,myX,myy,mylambda=0.).flatten()

    # Using theta initialized at [1; 1] you should expect to see a
# gradient of [-15.303016; 598.250744] (with lambda=1)
mytheta = np.array([[1.],[1.]])
print(computeGradientFlattened(mytheta,X,Y,1.))



def optimizeTheta(myTheta_initial, myX, myy, mylambda=0.,print_output=True):
    fit_theta = scipy.optimize.fmin_cg(computeCost,x0=myTheta_initial,\
                                       fprime=computeGradientFlattened,\
                                       args=(myX,myy,mylambda),\
                                       disp=True,\
                                       maxiter=1000,full_output=True)
    return fit_theta

mytheta = np.array([[1.],[1.]])
fit_theta = optimizeTheta(mytheta,X,Y,0.)

print(mytheta.shape)
print(fit_theta)