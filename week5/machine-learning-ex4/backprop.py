import scipy
import numpy as np
from scipy.special import expit 
import scipy.io
from scipy import optimize
import pandas as pd
import random
import matplotlib.cm as cm 
from PIL import Image
import matplotlib.pyplot as plt
import itertools

dataFile = ('./ex4/ex4data1.mat')
data = scipy.io.loadmat(dataFile)
X = data['X']
Y = data['y']
X = np.insert(X,0,1,axis=1)
thetaFile = ('./ex4/ex4weights.mat')
thetaData = scipy.io.loadmat(thetaFile)
theta1 = thetaData['Theta1']
theta2 = thetaData['Theta2']
thetaArray = [theta1, theta2]
correctidx = []
wrongidx = []


def sigmoid(z):
    return expit(z)


def getDatumImg(row):
    width,height = 20, 20
    square = row[1:].reshape(width,height)

    return square.T
    
def getDatumImg2(row):
    width,height = 5, 5
    square = row[1:].reshape(width,height)

    return square.T

def displayData(indices = None):
    width,height = 20,20
    nrows,ncols = 10,10
    if not indices:
        indices = random.sample(range(X.shape[0]),nrows*ncols)
    bigPicture = np.zeros((height*nrows,width*ncols))
    irow, icol = 0,0
    for idx in indices:
        if icol == ncols:
            icol = 0
            irow += 1
        iimg = getDatumImg(X[idx])
        bigPicture[irow*height:irow*height+iimg.shape[0],icol*width:icol*width+iimg.shape[1]] = iimg
        icol += 1
    fig = plt.figure(figsize=(6,6))
    img = Image.fromarray(bigPicture*255)
    cmapChoice = cm.get_cmap('Greys')
    plt.imshow(img, cmap = cmapChoice)
    plt.show()

def correctDataShow(correctidx,thetaas):
    fig = plt.figure(figsize=(7,7))
    x = random.choice(correctidx)
    iimg = Image.fromarray(getDatumImg(X[x]*255))
    plt.imshow(iimg,"Greys")
    correctValue = predictY(X[x],thetaas)
    correctValue = 0 if correctValue == 10 else correctValue
    fig.suptitle('Youngs Program \n Correctly Predicted from this image \n the number to be :%d ' %correctValue, fontsize = 14, fontweight = 'bold')
    plt.show()

def wrongDataShow(wrongidx,thetaas):
    fig = plt.figure(figsize=(7,7))
    x = random.choice(wrongidx)
    iimg = Image.fromarray(getDatumImg(X[x]*255))
    plt.imshow(iimg,"Greys")
    wrongvalue = predictY(X[x],thetaas)
    wrongvalue = 0 if wrongvalue == 10 else wrongvalue
    fig.suptitle('Youngs Program \n Wrongfully Predicted from this image \n the number to be: %d' %wrongvalue, fontsize = 14, fontweight = 'bold')
    plt.show()


def flattendThetas  (thetaas):
    flattendList = [eachTheta.flatten() for eachTheta in thetaas]
    combinedList = list(itertools.chain.from_iterable(flattendList))
    return np.array(combinedList).reshape((len(combinedList),1))

def reshapeThetas(flattendThetas):
    newTheta1 = flattendThetas[:25*401].reshape(25,401)
    newTheta2 = flattendThetas[25*401:].reshape(10,26)
    return [newTheta1, newTheta2]

def flattenX(xlist):
    flattenedX = xlist.flatten()
    return flattenedX.reshape(5000*401,1)

xlist = flattenX(X)
flatThetas = flattendThetas(thetaArray)

def reshapeX(flattenedX):
    return flattenedX.reshape(5000,401)

def computeCost(flattenedTheta,flatX,yval,lambda1 = 0):
    xval = reshapeX(flatX)
    thetaas = reshapeThetas(flattenedTheta)
    thetaa1 = thetaas[0]
    thetaa2 = thetaas[1]
    totalCost = 0
    m = X.shape[0]
    for i in range(len(xval)):
        row = xval[i]
        zAndA = propogateForward(row,thetaas)
        outputA = zAndA[-1][1]
        correctY = np.zeros((10,1))
        correctY[yval[i]-1] = 1
        costleft = -(correctY.T).dot(np.log(outputA))
        costright = (1-correctY.T).dot(np.log(1-outputA))
        cost = (costleft-costright)    
        totalCost += cost
    totalCost = float(totalCost)/m
    total_reg = 0.
    total_reg += (lambda1/(2*m))*np.sum(theta1*theta1) #element-wise multiplication
    total_reg += (lambda1/(2*m))*np.sum(theta2*theta2) #element-wise multiplication
    finalCost = totalCost + total_reg 
    return finalCost

def propogateForward(row,thetaas):
    thetaa1 = thetaas[0]
    thetaa2 = thetaas[1]
    zAndA = []
    z2 = thetaas[0].dot(row).reshape((25,1))
    a2 = sigmoid(z2)
    zAndA.append((z2,a2))
    a2 = np.insert(a2,0,1)
    z3 = thetaa2.dot(a2).reshape(10,1)
    a3 = sigmoid(z3)
    zAndA.append((z3,a3))
    return np.array(zAndA)


def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))

print(computeCost(flatThetas,xlist,Y))

def randomThetas():
    theta1Shape = (25,401)
    theta2Shape = (10,26)
    initEpsilon=0.12
    newTheta1 = np.random.rand(*theta1Shape)*(2*initEpsilon)-initEpsilon
    newTheta2 = np.random.rand(*theta2Shape)*(2*initEpsilon)-initEpsilon
    thetaArray = [newTheta1, newTheta2]
    return thetaArray

def backPropogation(flattenTheta,flatX,yval,lambda1 = 0):
    thetaas = reshapeThetas(flattenTheta)
    xval = reshapeX(flatX)
    thetaa1 = thetaas[0]
    thetaa2 = thetaas[1]
    Delta1 = np.zeros((25,401))
    Delta2 = np.zeros((10,26))
    m = xval.shape[0]
    for i in range(m):
        row = xval[i]
        a1 = row.reshape(row.shape[0],1)
        correctY = np.zeros((10,1))
        correctY[yval[i]-1]=1
        zAndA = propogateForward(row,thetaas)
        z2 = zAndA[0][0]
        a2 = zAndA[0][1]
        z3 = zAndA[1][0]
        a3 = zAndA[-1][1] 
        delta3 = a3-correctY
        delta2 = thetaa2.T[1:,:].dot(delta3)*sigmoidGradient(z2)
        a2 = np.insert(a2,0,1).reshape(26,1)
        Delta1 += delta2.dot(a1.T)
        Delta2 += delta3.dot(a2.T)
    D1 = Delta1/float(m)
    D2 = Delta2/float(m)
    D1[:,1:] = D1[:,1:] + (float(lambda1)/m)*thetaa1[:,1:]
    D2[:,1:] = D2[:,1:] + (float(lambda1)/m)*thetaa2[:,1:]
    Ds = [D1,D2]
    return flattendThetas(Ds).flatten()

D1 = reshapeThetas(backPropogation(flatThetas,xlist,Y))[0]
D2 = reshapeThetas(backPropogation(flatThetas,xlist,Y))[1]
Ds = reshapeThetas(backPropogation(flatThetas,xlist,Y))

def checkGradient(flattenThetas,Ds,flatX,yval,lambda1 = 0):
    thetaas = reshapeThetas(flattenThetas)
    xval = reshapeX(flatX)
    thetaa1 = thetaas[0]
    thetaa2 = thetaas[1]
    myeps = 0.0001
    numberE = 25*401+10*26
    for i in range(10):
        x = int(np.random.rand()*(numberE))
        if x <= (25*401):
            epsvec = np.zeros((25,401))
            rowNum = int(x/401)
            columnNum = x - 401*rowNum
            epsvec[rowNum,columnNum] = myeps
            highThetaArray = flattendThetas(np.array((thetaa1+epsvec,thetaa2)))
            lowThetaArray = flattendThetas(np.array((thetaa1-epsvec,thetaa2)))
            cost_high = computeCost(highThetaArray,flatX,yval,lambda1)
            cost_low = computeCost(lowThetaArray,flatX,yval,lambda1)
            DtoShow = D1[rowNum,columnNum]
        else:
            epsvec = np.zeros((10,26))
            rowNum = int((x-(25*401))/26)
            columnNum = (x-(25*401)) - 26*rowNum
            epsvec[rowNum,columnNum] = myeps
            highThetaArray = flattendThetas(np.array((thetaa1,thetaa2+epsvec)))
            lowThetaArray = flattendThetas(np.array((thetaa1,thetaa2-epsvec)))
            cost_high = computeCost(highThetaArray,flatX,yval,lambda1)
            cost_low = computeCost(lowThetaArray,flatX,yval,lambda1)
            DtoShow = D2[rowNum,columnNum]
        mygrad = (cost_high - cost_low) / float(2*myeps)
        print("Element: %d. Numerical Gradient = %f. BackProp Gradient = %f."%(x,mygrad,DtoShow))

# checkGradient(flatThetas,Ds,xlist,Y)


def optimizeThetas(lambda1=0):
    randomThetaArray = flattendThetas(randomThetas()) 
    print(randomThetaArray.shape)
    result = scipy.optimize.fmin_cg(computeCost,x0=randomThetaArray,fprime=backPropogation,args=(xlist,Y,lambda1),maxiter=50,disp=True,full_output=True)
    return result

optimizedThetaas = optimizeThetas()[0]

def predictY(row,thetaas):
    classes= np.concatenate((np.arange(1,10),[10]),axis=None)
    sigVals = propogateForward(row,thetaas)[-1][1]
    return classes[np.argmax(sigVals)]

def computeAccuracy(flatthetas,flatX,yval,lambda1=0):
    xval = reshapeX(flatX)
    thetaas = reshapeThetas(flatthetas)
    correctAmount = 0
    for i in range(len(xval)):
        row = xval[i]
        if predictY(row,thetaas) == yval[i]:
            correctAmount += 1
            correctidx.append(i)
        else:
            wrongidx.append(i)
    for i in range(3):
        correctDataShow(correctidx,thetaas)
    for i in range(1):
        wrongDataShow(wrongidx,thetaas)
    percentcorrect = correctAmount/(len(xval))*100
    return percentcorrect, correctAmount

print('From 5000 sample images, %d has been correctly assessed. Accuracy is %0.1f%%'%(computeAccuracy(optimizedThetaas,xlist,Y)[1], computeAccuracy(optimizedThetaas,xlist,Y)[0]))

def showHiddenLayer(hiddenlayerThetas):
    width, height = 20,20
    ncol, nrow = 5,5
    bigpicture = np.zeros((height*nrow,width*ncol))
    hiddenLayerNoBias = hiddenlayerThetas[:,1:]
    irow , icol = 0,0
    for row in hiddenLayerNoBias:
        if icol == ncol:
            icol = 0
            irow += 1
        iimg = getDatumImg(np.insert(row,0,1))
        bigpicture[height*irow:height*irow+height,width*icol:width*icol+width]= iimg
        icol += 1
    makeimg = Image.fromarray(bigpicture*255)
    cmapChoice = cm.get_cmap('Greys_r')
    plt.imshow(makeimg,cmap = cmapChoice)
    plt.show()

showHiddenLayer(reshapeThetas(optimizedThetaas)[0])