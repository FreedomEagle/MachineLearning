import scipy
import scipy.io
import numpy as np 
import matplotlib.pyplot as plt 
import os
from sklearn import svm

os.chdir('/home/youngson/program/machineLearning/week7/machine-learning-ex6')
os.getcwd()
data1Address = './ex6/ex6data1.mat'
data1 = scipy.io.loadmat(data1Address)

X1 = data1['X']
y1 = data1['y']

pos = np.array([X1[i] for i in range(X1.shape[0]) if y1[i]==1])
neg = np.array([X1[i] for i in range(X1.shape[0]) if y1[i]==0])

def plotData(posVal,negVal):
    plt.plot(posVal[:,0],posVal[:,1],'k+',label = 'Positive Sample')
    plt.plot(negVal[:,0],negVal[:,1],'yo',label = 'Negative Sample')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()

linear_SVM = svm.SVC(C=1, kernel = 'linear')
linear_fit = linear_SVM.fit(X1,y1.flatten())
linear_SVM2 = svm.SVC(C=100, kernel = 'linear')
linear_fit2 = linear_SVM2.fit(X1,y1.flatten())

def plotBoundary(mysvm,xmin,xmax,ymin,ymax,comment,posval,negval):
    xvals = np.linspace(xmin,xmax,100)
    yvals = np.linspace(ymin,ymax,100)
    zvals = np.zeros((len(xvals),len(yvals)))

    for i in range(len(xvals)):
        for j in range(len(yvals)):
            variable = np.array([xvals[i],yvals[j]])
            zvals[i][j] = float(mysvm.predict([variable]))
    zvals = zvals.transpose()
    xx, yy = np.meshgrid(xvals,yvals)
    plotData(posval,negval)
    plt.contour(xx,yy,zvals)
    plt.title('Decision Boundary (C) = ' + comment)
    plt.show()

plotBoundary(linear_fit,0,4.5,1.5,5,'1  ; Outlier Fails',pos,neg)
plotBoundary(linear_fit2,0,4.5,1.5,5,'100 ; Outlier Contained',pos,neg)

def gaussianKernel(x1,x2,sigma):
    sigmaSquared = np.power(sigma,2)
    diffSquared = (abs(x1-x2)).T.dot(abs(x1-x2))
    return np.exp(-(diffSquared/(2*sigmaSquared)))

# print(gaussianKernel(np.array([1, 2, 1]),np.array([0, 4, -1]), 2.))

data2Address = './ex6/ex6data2.mat'
data2 = scipy.io.loadmat(data2Address)
X2 = data2['X']
y2 = data2['y']
pos2 = np.array([X2[i] for i in range(X2.shape[0]) if y2[i] ==1])
neg2 = np.array([X2[i] for i in range(X2.shape[0]) if y2[i] ==0])


def gaussianWithDiffGamma():
    sigma = .1
    gammaArray = [-3,-2,-1,0]
    for i in range(len(gammaArray)):
        gamma = np.power(sigma,gammaArray[i])
        gaussian_method = svm.SVC(kernel='rbf',gamma=gamma)
        gaussian_method.fit(X2,y2.flatten())
        plotBoundary(gaussian_method,0,1,.4,1.0,'GaussianMethod  C = 1, Gamma = '+ str(int(gamma)),pos2,neg2)
        
# gaussianWithDiffGamma()

def gaussianWithDiffC():
    CArray = [10, 10000]
    for i in range(len(CArray)):
        gaussian_method = svm.SVC(C=CArray[i] ,kernel='rbf',gamma=100)
        gaussian_method.fit(X2,y2.flatten())
        plotBoundary(gaussian_method,0,1,.4,1.0,'GaussianMethod  C = '+ str(CArray[i])+ '; Gamma = 100',pos2,neg2)
        

data3Address = './ex6/ex6data3.mat'
data3 = scipy.io.loadmat(data3Address)

X1 = data1['X']
y1 = data1['y']

pos = np.array([X1[i] for i in range(X1.shape[0]) if y1[i]==1])
neg = np.array([X1[i] for i in range(X1.shape[0]) if y1[i]==0])

def plotData(posVal,negVal):
    plt.plot(posVal[:,0],posVal[:,1],'k+',label = 'Positive Sample')
    plt.plot(negVal[:,0],negVal[:,1],'yo',label = 'Negative Sample')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()

linear_SVM = svm.SVC(C=1, kernel = 'linear')
linear_fit = linear_SVM.fit(X1,y1.flatten())
linear_SVM2 = svm.SVC(C=100, kernel = 'linear')
linear_fit2 = linear_SVM2.fit(X1,y1.flatten())

def plotBoundary(mysvm,xmin,xmax,ymin,ymax,comment,posval,negval):
    xvals = np.linspace(xmin,xmax,100)
    yvals = np.linspace(ymin,ymax,100)
    zvals = np.zeros((len(xvals),len(yvals)))

    for i in range(len(xvals)):
        for j in range(len(yvals)):
            variable = np.array([xvals[i],yvals[j]])
            zvals[i][j] = float(mysvm.predict([variable]))
    zvals = zvals.transpose()
    xx, yy = np.meshgrid(xvals,yvals)
    plotData(posval,negval)
    plt.contour(xx,yy,zvals)
    plt.title( comment)
    plt.show()

plotBoundary(linear_fit,0,4.5,1.5,5,'Decision Boundary (C) = 1  ; Outlier Fails',pos,neg)
plotBoundary(linear_fit2,0,4.5,1.5,5,'Decision Boundary (C) = 100 ; Outlier Contained',pos,neg)

def gaussianKernel(x1,x2,sigma):
    sigmaSquared = np.power(sigma,2)
    diffSquared = (abs(x1-x2)).T.dot(abs(x1-x2))
    return np.exp(-(diffSquared/(2*sigmaSquared)))

# print(gaussianKernel(np.array([1, 2, 1]),np.array([0, 4, -1]), 2.))

data2Address = './ex6/ex6data2.mat'
data2 = scipy.io.loadmat(data2Address)
X2 = data2['X']
y2 = data2['y']
pos2 = np.array([X2[i] for i in range(X2.shape[0]) if y2[i] ==1])
neg2 = np.array([X2[i] for i in range(X2.shape[0]) if y2[i] ==0])


def gaussianWithDiffGamma():
    sigma = .1
    gammaArray = [-3,-2,-1,0]
    for i in range(len(gammaArray)):
        gamma = np.power(sigma,gammaArray[i])
        gaussian_method = svm.SVC(kernel='rbf',gamma=gamma)
        gaussian_method.fit(X2,y2.flatten())
        plotBoundary(gaussian_method,0,1,.4,1.0,'GaussianMethod  C = 1, Gamma = '+ str(int(gamma)),pos2,neg2)
        
# gaussianWithDiffGamma()

def gaussianWithDiffC():
    CArray = [10, 10000]
    for i in range(len(CArray)):
        gaussian_method = svm.SVC(C=CArray[i] ,kernel='rbf',gamma=100)
        gaussian_method.fit(X2,y2.flatten())


X3 = data3['X']
y3 = data3['y']
Xval3 = data3['Xval']
yval3 = data3['yval']
pos3val = np.array([Xval3[i] for i in range(Xval3.shape[0]) if yval3[i] ==1])
neg3val = np.array([Xval3[i] for i in range(Xval3.shape[0]) if yval3[i] ==0])

Cvalues = (0.01,0.03,0.1,0.3,1.,3.,10.,30.)
sigmaValues = Cvalues
best_pair, best_score = (0,0) , 0

for i in Cvalues:
    for j in sigmaValues:
        gamma = np.power(j,-2)
        gaussian_svm = svm.SVC(C = i, kernel = 'rbf',gamma=gamma)
        gaussian_svm.fit(X3,y3.flatten())
        this_score = gaussian_svm.score(Xval3,yval3)
        if this_score > best_score:
            best_score = this_score
            best_pair = (i,j)

print("Best C, sigma pair is (%f, %f) with a score of %f."%(best_pair[0],best_pair[1],best_score))

gaussian_svm = svm.SVC(C=best_pair[0], kernel='rbf', gamma= np.power(best_pair[1],-2))
gaussian_svm.fit(X3,y3.flatten())
plotBoundary(gaussian_svm,-.5,.3,-.8,.6,'Gaussian Support Vector Machine, Optimized C = '+ str(best_pair[0])+' Optimized Sigma = '+ str(best_pair[1]),pos3val,neg3val)