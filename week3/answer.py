import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy.special import expit as sigmoid
from scipy import optimize

'''Graphing the Data'''

# datafile = 'data/ex2data2.txt'

# cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True)
# X = np.transpose(np.array(cols[0:2]))
# Y = np.transpose(np.array(cols[2:]))
# m = Y.size

# X = np.insert(X,0,1,axis=1)
# pos = np.array([X[i] for i in range(X.shape[0]) if Y[i]==1])
# neg = np.array([X[i] for i in range(X.shape[0]) if Y[i]==0])

datafile2= 'data/ex2data2.txt'
data2 = np.loadtxt(datafile2, delimiter =',', unpack=True)
X2 = np.transpose(np.array(data2[0:2]))
Y2 = np.transpose(np.array(data2[2:3]))
X2 = np.insert(X2, 0, 1, axis =1)
sampleSize = Y2.size

pos = np.array([X2[i] for i in range(X2.shape[0]) if Y2[i] == 1])
neg = np.array([X2[i] for i in range(X2.shape[0]) if Y2[i] == 0])


'''sigmoid function'''
def h(theta,Xvariable):  #logistic hypothetical function
    return sigmoid(np.dot(Xvariable,theta))

def Jtheta(theta,Xvariable,Yvariable,mylambda=0):
    term1 = np.dot(-np.array(Yvariable).T,np.log(h(theta,Xvariable)))
    term2 = np.dot((1-np.array(Yvariable)).T,np.log(1-h(theta,Xvariable)))
    regularizationTerm = (mylambda/2)*np.sum(np.dot(theta[1:].T,theta[1:]))
    
    return float (1./sampleSize)*np.sum((term1-term2)+regularizationTerm)

# initialTheta = np.zeros((X.shape[1],1))


# def optimizeTheta(theta,Xvariable,Yvariable,mylambda=0):
#     result = optimize.fmin(Jtheta,x0=theta,args=(Xvariable,Yvariable,mylambda),maxiter = 400,full_output=True)
#     return result

# optimizedTheta = optimizeTheta(initialTheta,X,Y)[0]
# print(Jtheta(optimizedTheta,X,Y))
# def plotData():
#     plt.figure(figsize=(10,6))
#     plt.plot(pos[:,1],pos[:,2],'cD',label='Admitted')
#     plt.plot(neg[:,1],neg[:,2],'yo',label='NotAdmitted')
#     plt.xlabel('Exam 1 Score')
#     plt.ylabel('Exam 2 Score')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
# boundaryxs = np.array([np.min(X[:,1]),np.max(X[:,1])])
# boundaryys=np.array(-1./optimizedTheta[2]*(optimizedTheta[0]+optimizedTheta[1]*boundaryxs))
# print(optimizedTheta)

# def makePrediction(theta,Xvariable):
#     return h(theta,Xvariable) >= 0.5

# posCorrect = np.sum(makePrediction(optimizedTheta,pos))
# negCorrect = np.sum(np.invert(makePrediction(optimizedTheta,neg)))
# percentCorrect = (posCorrect+negCorrect)/(len(pos)+len(neg))


def plotData2():
    plt.plot(pos[:,1],pos[:,2],'k+',label='y=1')
    plt.plot(neg[:,1],neg[:,2],'yo',label='y=0')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()
    plt.grid(True)
    
def mapFeatures(X1col,X2col):
    degree = 6
    out = np.ones((X1col.shape[0],1))
    for i in range(1,degree+1):
        for j in range(0, i+1):
            term1 = X1col**(i-j)
            term2 = X2col**(j)
            term = (term1*term2).reshape(term1.shape[0],1)
            out = np.hstack((out,term))
    return out
    
mapX = mapFeatures(X2[:,1],X2[:,2])
initialTheta2 = np.zeros((mapX.shape[1],1))



# initialTheta = np.zeros((X.shape[1],1))


# def optimizeTheta(theta,Xvariable,Yvariable,mylambda=0):
#     result = optimize.fmin(Jtheta,x0=theta,args=(Xvariable,Yvariable,mylambda),maxiter = 400,full_output=True)
#     return result

# optimizedTheta = optimizeTheta(initialTheta,X,Y)[0]
# print(Jtheta(optimizedTheta,X,Y))
# def plotData():
#     plt.figure(figsize=(10,6))
#     plt.plot(pos[:,1],pos[:,2],'cD',label='Admitted')
#     plt.plot(neg[:,1],neg[:,2],'yo',label='NotAdmitted')
#     plt.xlabel('Exam 1 Score')
#     plt.ylabel('Exam 2 Score')
#     plt.legend()


def optimizeRegularizedTheta(mytheta,myX,myy,mylambda=0.):
    result = optimize.minimize(Jtheta, mytheta, args=(myX, myy, mylambda),  method='BFGS', options={"maxiter":500, "disp":False} )
    return np.array([result.x]), result.fun

optimizedTheta, mincost = optimizeRegularizedTheta(initialTheta2,mapX,Y2)

#     plt.grid(True)
#     plt.show()
# boundaryxs = np.array([np.min(X[:,1]),np.max(X[:,1])])
# boundaryys=np.array(-1./optimizedTheta[2]*(optimizedTheta[0]+optimizedTheta[1]*boundaryxs))
# print(optimizedTheta)

# def makePrediction(theta,Xvariable):
#     return h(theta,Xvariable) >= 0.5

# posCorrect = np.sum(makePrediction(optimizedTheta,pos))
# negCorrect = np.sum(np.invert(makePrediction(optimizedTheta,neg)))
# percentCorrect = (posCorrect+negCorrect)/(len(pos)+len(neg))


def plotData2():
    plt.plot(pos[:,1],pos[:,2],'k+',label='y=1')
    plt.plot(neg[:,1],neg[:,2],'yo',label='y=0')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()
    plt.grid(True)
    
def mapFeatures(X1col,X2col):
    degree = 6
    out = np.ones((X1col.shape[0],1))
    for i in range(1,degree+1):
        for j in range(0, i+1):
            term1 = X1col**(i-j)
            term2 = X2col**(j)
            term = (term1*term2).reshape(term1.shape[0],1)
            out = np.hstack((out,term))
    return out
    
mapX = mapFeatures(X2[:,1],X2[:,2])
initialTheta2 = np.zeros((mapX.shape[1],1))


def optimizeRegularizedTheta(mytheta,myX,myy,mylambda=0.):
    result = optimize.minimize(Jtheta, mytheta, args=(myX, myy, mylambda),  method='BFGS', options={"maxiter":500, "disp":False} )
    return np.array([result.x]), result.fun

optimizedTheta, mincost = optimizeRegularizedTheta(initialTheta2,mapX,Y2)
print(Jtheta(initialTheta2,mapX,Y2))

def linespace(start,end,iterations):
    xvalArray = [start]
    xval = start
    for i in range(iterations-1):
        xval = xval + (end-start)/(iterations-1)
        xvalArray.append(xval)
    return np.array(xvalArray)


def plotBoundary(theta,myX,myY,mylambda=0):
    xvals = linespace(-1,1.5,50)
    yvals = linespace(-1,1.5,50)
    optimizedTheta, mincost = optimizeRegularizedTheta(theta,myX,myY,mylambda)
    zvals = np.zeros((len(xvals),len(yvals)))
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            myfeaturesij = mapFeatures(np.array([xvals[i]]),np.array([yvals[j]]))
            zvals[i][j] = np.dot(optimizedTheta,myfeaturesij.T)
    zvals = zvals.transpose()
    u, v = np.meshgrid( xvals, yvals)
    mycontour = plt.contour( xvals, yvals, zvals, [0])
    #Kind of a hacky way to display a text on top of the decision boundary
    myfmt = { 0:'Lambda = %d'%mylambda}
    plt.clabel(mycontour, inline=1, fontsize=15, fmt=myfmt)
    plt.title("Decision Boundary")

plt.figure(figsize=(12,10))
plotData2()
plotBoundary(optimizedTheta,mapX,Y2,1)
plt.show()
