import random
import numpy as np  
import matplotlib.pyplot as plt
import scipy.io 
from scipy.spatial import distance
dataFile = 'ex7/ex7data2.mat'
data = scipy.io.loadmat(dataFile)
X = data['X']
global kMeans 

def distanceCalc(array1,array2):
    diff = 0
    for i in range(len(array1)):
        diff += (array1[i]-array2[i])**2
    return diff

def findDistance(centroid,ogArray,k, vectorsize):
    ogArray[:][ogArray.shape[1]-1] = 100000000
    rows = ogArray.shape[0]
    diff = np.zeros((rows*k,1))
    print("Centroid!")
    print(centroid)
    for j in range(centroid.shape[0]):
        for i in range(rows): 
            diff[rows*j+i] = distanceCalc(ogArray[i,:vectorsize],centroid[j,:vectorsize])
    distanceArray = diff[:,0]
    print('Distance!')
    print(distanceArray[0:10])
    print(distanceArray[rows:rows+10])
    print(distanceArray[rows*2:rows*2+10])
    return centroidTeam(distanceArray,k,ogArray,rows)

def centroidTeam(distancearray,k,idxArray,rows):
    setRepeat = np.linspace(0,k-1,k)
    setNum = int(distancearray.shape[0]/k)
    distancearray.reshape((distancearray.shape[0],1))
    print("INDEX")
    print(idxArray[:10])
    for j in setRepeat:
        for i in range(rows):
            position = int(rows*j+i)
            if distancearray[position] <= idxArray[i][idxArray.shape[1]-1]:
                 idxArray[i][idxArray.shape[1]-1] = distancearray[position]
                 idxArray[i][idxArray.shape[1]-2] = j
    print(idxArray[:10])
    return idxArray

def plotGraph(points, centroid,k):
    colorset = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    setRepeat = np.linspace(0,k-1,k)
    for i in range(centroid.shape[0]):
        plt.scatter(centroid[i,0],centroid[i,1],c=colorset[i],s=500)
        xval,yval = np.array([]), np.array([])
        clusterArray = [[]]
        for j in range(points.shape[0]):
            clusterArray.append([])
            if points[j][points.shape[1]-2] == i:
                xval = np.append(xval,points[j,0])
                yval = np.append(yval,points[j,1])
                clusterArray[j].append([points[j,0],points[j,1]])
        plt.scatter(xval,yval,c=colorset[i])
    plt.show()
   

# def changeCentroid(idxArray,centroid,k,vectorsize):
#     setRepeat = np.linspace(0,k-1,k)
#     kCounter = np.zeros((len(setRepeat),1))
#     kMeans = np.zeros((centroid.shape[0],1))
#     KArray = np.zeros((centroid.shape[0],centroid.shape[1]))
#     for i in range(len(setRepeat)):
#         for j in range(idxArray.shape[0]):
#             if idxArray[j][idxArray.shape[1]-2]== setRepeat[i]:
#                 kMeans[i] += idxArray[j][idxArray.shape[1]-1] 
#                 kCounter[i] += 1
#                 for l in range(vectorsize):
#                     KArray[i][l] += centroid[i][l]-idxArray[j][l]
#     for idx in range(len(setRepeat)):
#         kMeans[idx] = kMeans[idx]/kCounter[idx]
#         KArray[idx] = KArray[idx]/kCounter[idx]
#     centroid = centroid[:,:vectorsize] - KArray[:,:vectorsize]  
#     return centroid

def changeCentroid(idxArray,centroid,k,vectorsize):
    setRepeat = np.linspace(0,k-1,k)
    newCentroid = np.zeros((centroid.shape[0],vectorsize))
    print(newCentroid.shape) 
    for i in range(len(setRepeat)):
        tempArray = []
        for j in range(idxArray.shape[0]):
            if idxArray[j][idxArray.shape[1]-2] == i :
                tempArray.append(np.array((idxArray[j][:vectorsize])))
        print('O')
        print(tempArray)
        newCentroid[i] = np.mean(tempArray,axis=0)
    return newCentroid

def findCentroid(idxArray,k):
    initialValue = 100000
    initialidxArray = np.insert(idxArray,idxArray.shape[1],0,axis=1)
    initialidxArray = np.insert(initialidxArray,initialidxArray.shape[1],initialValue,axis=1)
    vectorsize = idxArray.shape[1]
    maxIter = 10
    centroidArray = np.zeros((k*maxIter,idxArray.shape[1]))
    totalMeansSize = int(centroidArray.shape[0]/k)
    totalMeans = np.zeros((totalMeansSize,1))
    for iterNum in range(maxIter):
        initialCentroid = np.zeros((k,idxArray.shape[1]))
        for i in range(k):
            for column in range(idxArray.shape[1]):
                vectorMin = initialidxArray[:,column].min(axis=0)
                vectorMax = initialidxArray[:,column].max(axis=0)
                vectorElem = random.randint(int(vectorMin),int(vectorMax))
                initialCentroid[i][column] = vectorElem
        idxArray = findDistance(initialCentroid,initialidxArray,k,vectorsize)
        print(idxArray.shape)
        newcentroid = changeCentroid(initialidxArray,initialCentroid,k,vectorsize)
        for iterations in range(5):
            idxArray = findDistance(newcentroid,idxArray,k,vectorsize)
            newcentroid = changeCentroid(idxArray,initialCentroid,k,vectorsize)
        centroidArray[iterNum*(initialCentroid.shape[0]):iterNum*(initialCentroid.shape[0])+(initialCentroid.shape[0]),:vectorsize] = newcentroid[:,:vectorsize]
        totalMeans[length] += float(centroidArray[length*initialCentroid.shape[0]+ksize][-1])
    bestIdx = int(np.where(np.isclose(totalMeans,totalMeans.min(axis=0)))[0][0])
    finalCentroid = np.zeros((initialCentroid.shape[0],initialCentroid.shape[1]))
    for element in range(k):
        finalCentroid[element,:vectorsize] = centroidArray[bestIdx*k+element][:vectorsize]
    print(finalCentroid)
    # print(totalMeans[bestIdx])
    idxArray = findDistance(finalCentroid[:,:vectorsize],idxArray,k,vectorsize)
    plotGraph(idxArray,finalCentroid[:,:vectorsize],k)
    return idxArray , finalCentroid[:,:vectorsize], totalMeans[bestIdx]


testArray = np.array([[1,2],[5,6],[9,10],[3,3],[6,7],[1,3],[4,5],[6,6],[7,1]])
# dataArray, centroidAnswers, finalMean = findCentroid(X,1)
# dataArray, centroidAnswers, finalMean = findCentroid(X,2)
dataArray, centroidAnswers, finalMean = findCentroid(X,3)
# dataArray, centroidAnswers, finalMean = findCentroid(X,6)

# print('Data was \n' + str(dataArray))
# print(' final Centroid was \n ' + str(centroidAnswers))
# print(" final Mean being "+ str(finalMean))

imgDataAddress = 'ex7/bird_small.mat'
imgData = scipy.io.loadmat(imgDataAddress)['A']

def plotimg(data):
    colorset = ['r','g','b',  'c', 'm', 'y', 'k', 'w']
    for i in range(3):
        xval,yval = np.array([]), np.array([])
        clusterArray = [[]]
        for j in range(data.shape[0]):
            clusterArray.append([])
            if data[j][2] == i:
                xval = np.append(xval,data[j,0])
                yval = np.append(yval,data[j,1])
                clusterArray[j].append([data[j,0],data[j,1]])
        plt.scatter(xval,yval,c=colorset[i])
    plt.show()

# findCentroid(imgData[0],16)
