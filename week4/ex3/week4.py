import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import random
import matplotlib.cm as cm 
from PIL import Image
import scipy
import scipy.io
from scipy import optimize
from scipy.special import expit 

dataFile = './ex3/ex3data1.mat'
data = scipy.io.loadmat(dataFile)
X = data['X']
Y = data['y']
X = np.insert(X,0,1,axis=1)

def getDatumImg(row):
    width, height = 20, 20
    square = row[1:].reshape(width,height)
    return square.T

def displayData(indices = None):
    width, height = 20, 20
    nrows, ncols = 10, 10
    if not indices:
        indices = random.sample(range(X.shape[0]),nrows*ncols)
    big_picture = np.zeros((nrows*width , ncols*height))
    irow, icol =0,0
    for idx in indices:
        if icol == ncols:
            irow += 1
            icol = 0
        iimg = getDatumImg(X[idx]*255)
        big_picture[irow*height:irow*height+iimg.shape[0],icol*width:icol*width+iimg.shape[1]]= iimg
        icol += 1
    plt.figure(figsize =(6,6))
    img = Image.fromarray(big_picture)

    cmapChoice = cm.get_cmap("Greys")
    plt.imshow(img,cmap = cmapChoice)
    plt.show()

def h(theta,myX):
    return expit(np.dot(myX,theta))

def J(theta,myX,myY,mylambda=0):
    m = myX.shape[0]
    term1 = np.log(h(theta,myX)).dot(-myY.T)
    term2 = np.log(1-h(theta,myX)).dot(1-myY.T)
    print(term1.shape)
    leftTerm = (term1 - term2)/m
    rightTerm = (theta.dot(theta))*mylambda/(2*m)
    return leftTerm + rightTerm

def gradientJ(theta,myX,myY,mylambda=0):
    m = myX.shape[0]
    beta = h(theta,myX)-myY.T
    regTerm = theta[1:]*(mylambda/m)
    gradient = 1./m*np.dot(myX.T,beta)
    gradient[1:] = gradient[1:]+regTerm
    return gradient


def optimizeTheta(theta,myX,myY,mylambda=0):
    result = optimize.fmin_cg(J,fprime = gradientJ, x0 = theta,args = (myX,myY,mylambda), maxiter =50, disp=False,full_output = True)
    return result[0],result[1]

def buildTheta():
    mylambda = 0 
    initialTheta = np.zeros((X.shape[1],1)).reshape(-1)
    Theta = np.zeros((10,X.shape[1]))
    for i in range(10):
        iclass = i if i else 10
        correctY = np.array([1 if x == iclass else 0 for x in Y] )
        itheta,mincost =optimizeTheta(initialTheta,X,correctY,mylambda)
        Theta[i,:] = itheta 
    return Theta

optimalTheta = buildTheta()

def predictOneAll(theta,myX):
    optionArray = np.concatenate(([10],np.arange(1,10)),axis=None)
    answerArray = [0]*len(optionArray)
    for i in range(len(optionArray)):
        answerArray[i] = h(theta[i],myX)
    # counts = np.bincount(optionArray[np.argmax(np.array(answerArray[i]))])
    # print(np.argmax(counts))
    
    return optionArray[np.argmax(np.array(answerArray),axis=0)]

n_correct,n_total = 0, 0
n_correctIdx = []
n_correctVal = []
n_wrong = []
n_wrongVal = []

for i in range(X.shape[0]):
    n_total += 1
    if predictOneAll(optimalTheta,X[i]) == Y[i]:   
        n_correct += 1
        n_correctIdx.append(i)
        n_correctVal.append(predictOneAll(optimalTheta,X[i]))
    else:
        n_wrong.append(i)
        n_wrongVal.append(predictOneAll(optimalTheta,X[i]))
print('Accuracy is %0.1f%%'%(100*n_correct/n_total))
# print(n_correctVal[2500:2510])
# displayData(n_correctIdx[2500:2510])

dataFile2 = './ex3/ex3weights.mat'
mat = scipy.io.loadmat(dataFile2)
Theta1, Theta2 = mat['Theta1'], mat['Theta2']
Thetas = [Theta1,Theta2]

def propogateForward(row,thetas):
    features = row
    for i in range(len(thetas)):
        theta = thetas[i]
        z = theta.dot(features)
        a = expit(z)
        if i == len(thetas)-1:
            return a
        a = np.insert(a,0,1)
        features = a

def predictCNN(row,thetas):
    optionArray = np.concatenate((np.arange(1,10),[10]),axis=None)
    
    return optionArray[np.argmax(propogateForward(row,thetas),axis=0)]

optimal_correct_number , optimal_total,optimal_wrong_number = 0,0, 0
optimal_correct_idx = []
optimal_wrong_idx = []

for i in range(X.shape[0]):
    optimal_total += 1
    if predictCNN(X[i],Thetas) == int(Y[i]):
        optimal_correct_number += 1
        optimal_correct_idx.append(i)
    else:
        optimal_wrong_idx.append(i)
        optimal_wrong_number += 1

for i in range(5):
    i = random.choice(optimal_correct_idx)
    fig = plt.figure(figsize=(3,3))
    img = Image.fromarray(getDatumImg(X[i]*255))
    plt.imshow(img, cmap = "Greys")
    predicted_val = predictCNN(X[i],Thetas)
    predicted_val = 0 if predicted_val == 10 else predicted_val
    fig.suptitle('Correctly Predicted: %d' %predicted_val, fontsize = 14, fontweight= 'bold')
    plt.show()
    
for i in range(5):
    i = random.choice(optimal_wrong_idx)
    fig = plt.figure(figsize=(3,3))
    img = Image.fromarray(getDatumImg(X[i]*255))
    plt.imshow(img, cmap = "Greys")
    predicted_val = predictCNN(X[i],Thetas)
    predicted_val = 0 if predicted_val == 10 else predicted_val
    fig.suptitle('Wrongfully Predicted: %d' %predicted_val, fontsize = 14, fontweight= 'bold')
    plt.show()
