import numpy as np 
from scipy import optimize
import scipy.io
import matplotlib.pyplot as plt

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
thetaM = myTheta.shape[0]


def flattenTheta(thetaArray):
    return thetaArray.flatten()

myflattenThetas = flattenTheta(myTheta)
def reshapeTheta(flattenThetas,rowNum):
    return flattenThetas.reshape((rowNum,1))


def plotData(xval,yval,newY):
    plt.plot(xval[:,1],yval,'rx',ms=9)
    plt.plot(xval[:,1],newY,'')
    plt.ylabel('Water flowing out of the dam (V)')
    plt.xlabel('Change in Water Level (X)')
    plt.xticks(np.arange(-50,40,10))
    plt.show()

def plotDataog(xval,yval):
    plt.plot(xval[:,1],yval,'rx',ms=9)
    plt.ylabel('Water flowing out of the dam (V)')
    plt.xlabel('Change in Water Level (X)')
    plt.xticks(np.arange(-50,40,10))
    plt.show()

def linear(theta,xval):
    return np.dot(xval,theta)

def computeCost(flattenThetas,xval,yval,lambda1=0.):
    m = xval.shape[0]
    n = xval.shape[1]
    theta = reshapeTheta(flattenThetas,n)
    h = linear(theta,xval)
    
    leftTerm = float((1./(2*m))*np.dot(( h - yval).T,( h - yval)))
    regularizedTerm = float(lambda1)/(2*m)*float(np.dot(theta[1:].T,theta[1:]))
    cost = (leftTerm + regularizedTerm)
    return cost

def costGradient(flattenThetas,xval,yval,lambda1=0.):
    m = xval.shape[0]
    n = xval.shape[1]
    theta = reshapeTheta(flattenThetas,n)
    diff = linear(theta,xval)-yval
    leftTerm = (1./float(m))*xval.T.dot((diff))
    rightTerm = float(lambda1)/m*theta
    rightTerm[0]=0.
    answer = (leftTerm+rightTerm)
    return answer


def computeGradientFlattened(theta,xval,yval,lambda1=0.):
    return costGradient(theta,xval,yval,lambda1).flatten()

def optimizeTheta(flattenThetas,xval,yval,lambda1):
    result = scipy.optimize.fmin_cg(computeCost,x0=flattenThetas,fprime=computeGradientFlattened,args=(xval,yval,lambda1),maxiter=1000,disp=True,epsilon=1.49e-12)
    return result

# print(myTheta.shape)
print(computeCost(myTheta,X,Y,1))
print(costGradient(myTheta,X,Y,1.))
optimizedThetas = optimizeTheta(myflattenThetas,X,Y,0.)
print(optimizedThetas)
# print(X.shape,Y.shape)

def predictlinearY(flattenThetas,xval):
    n = xval.shape[1]
    theta = reshapeTheta(flattenThetas,n)
    Y = linear(theta,xval)
    return Y

# predictedY = predictlinearY(optimizedThetas,X)
# plotData(X,Y,predictedY)

def learningCurve(myflattenedTheta,xval,yval,xcross,ycross,lambda1=0,):
    m = xval.shape[0]
    error_train = []
    error_crossVal = []
    num_trainSet = []
    for i in range(1,m+1,1):
        trainingSubsetX = xval[:i,:]
        trainingSubsetY = yval[:i]
        optimizedTrainedThetas= optimizeTheta(myflattenedTheta,trainingSubsetX,trainingSubsetY,0.)
        print(optimizedTrainedThetas)
        error_trainVal = computeCost(optimizedTrainedThetas.flatten(),trainingSubsetX,trainingSubsetY,0.)
        error_train.append(error_trainVal)
        error_cross_validation = computeCost(optimizedTrainedThetas.flatten(),xcross,ycross,0.)
        print(error_cross_validation)
        print(error_trainVal)
        error_crossVal.append(error_cross_validation)
        num_trainSet.append(i)
    plt.plot(num_trainSet,error_train,label='Train')
    plt.plot(num_trainSet,error_crossVal,label='Cross Validation')
    plt.legend()
    plt.title('Learning Curve for Linear Regression')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Error')
    plt.plot(num_trainSet,error_train)
    plt.plot(num_trainSet,error_crossVal)
    plt.ylim(0,50)
    plt.show()

learningCurve(myflattenThetas,X,Y,Xcrossval,Ycrossval)    

def generatePolyFeatures(myX,p):
    newX = myX.copy()
    for i in range(p):
        dim = i+2
        newX = np.insert(newX,newX.shape[1],np.power(newX[:,1],dim),axis=1)
    return newX

def featureNormalize(xval):
    xnorm = xval.copy()
    column_mean = np.mean(xnorm,axis=0)
    xnorm[:,1:] = xnorm[:,1:]-column_mean[1:]
    print(xnorm.shape)
    stored_feature_stds = np.std(xnorm,axis=0,ddof=1)
    print(xnorm.shape)
    xnorm[:,1:] = xnorm[:,1:] / stored_feature_stds[1:]
    return xnorm, column_mean, stored_feature_stds



def plotFit(thetaas,means,stds):
    npoints = 50
    xvals = np.arange(-55,50,(50+55)/npoints)
    xmat = np.ones((npoints,1))
    xmat = np.insert(xmat,1,xvals.T,axis=1)
    xmat = generatePolyFeatures(xmat,len(thetaas)-2)
    xmat[:,1:] = xmat[:,1:] - means[1:]
    xmat[:,1:] = xmat[:,1:] / stds[1:]
    plt.plot(xvals,linear(thetaas,xmat),'b--')
    plotDataog(X,Y)
    plt.show()

# plotFit(newOptimizedTheta,stored_means,stored_stds)
computeCost(myflattenThetas,X,Y)
# print(myflattenThetas.shape, X.shape, Y.shape)
# print(newOptimizedTheta.shape, newCrossX_norm.shape,Ycrossval.shape)
# # print(newCrossX.shape, newCrossX_norm.shape)
# print(computeCost(newOptimizedTheta,newCrossX,Ycrossval))


def learningCurve2(myflattenedTheta,xval,yval,xcross,ycross,lambda1=0,):
    m = xval.shape[0]
    error_train = []
    error_crossVal = []
    num_trainSet = []
    for i in range(1,m+1,1):
        trainingSubsetX = xval[:i,:]
        trainingSubsetY = yval[:i]
        trainingSubsetX = generatePolyFeatures(trainingSubsetX,5)
        trainingSubsetX, dummy1, dummy2 =  featureNormalize(trainingSubsetX)
        optimizedTrainedThetas= optimizeTheta(myflattenedTheta,trainingSubsetX,trainingSubsetY,lambda1)
        error_trainVal = computeCost(optimizedTrainedThetas.flatten(),trainingSubsetX,trainingSubsetY,lambda1)
        error_train.append(error_trainVal)
        error_cross_validation = computeCost(optimizedTrainedThetas.flatten(),xcross,ycross,lambda1)
        error_crossVal.append(error_cross_validation)
        num_trainSet.append(i)
    plt.plot(num_trainSet,error_train,label='Train')
    plt.plot(num_trainSet,error_crossVal,label='Cross Validation')
    plt.legend()
    plt.title('Learning curve for Polynominal Regression')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Error')
    plt.plot(num_trainSet,error_train)
    plt.plot(num_trainSet,error_crossVal)
    plt.ylim(0,50)
    plt.show()


newX = generatePolyFeatures(X,5)
newX_norm, stored_means, stored_stds = featureNormalize(newX)
newThetainit = np.ones((newX_norm.shape[1],1))
newOptimizedTheta = optimizeTheta(newThetainit,newX_norm,Y,0.)
newCrossX = generatePolyFeatures(Xcrossval,5)
newCrossX_norm, storedCross_means, storedCross_stds = featureNormalize(newCrossX)
learningCurve2(newThetainit,X,Y,newCrossX_norm,Ycrossval)

fit_theta = optimizeTheta(newThetainit,newX_norm,Y,0)
fit_theta2 = optimizeTheta(newThetainit,newX_norm,Y,100)
plotFit(fit_theta,stored_means,stored_stds)
plotFit(fit_theta2,stored_means,stored_stds)
learningCurve2(fit_theta,X,Y,newCrossX_norm,Ycrossval,0)
