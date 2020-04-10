import random
from random import sample
import numpy as np  
import matplotlib.pyplot as plt
import imageio
import scipy.io 
from PIL import Image
import matplotlib.cm as cm
from scipy.spatial import distance

dataFile = 'ex7/ex7data2.mat'
data = scipy.io.loadmat(dataFile)
X = data['X']

#Choose the number of centroids... K = 3
K = 3
#Choose the initial centroids matching ex7.m assignment script
initial_centroids = np.array([[3,3],[6,2],[8,5]])


def distanceBetween(point,centroid):
    assert(point.shape == centroid.shape)
    return np.sum(np.square(centroid-point))

def findClosestCentroids(myX,centroids):
    idx = np.zeros((myX.shape[0],1))
    for x in range(myX.shape[0]):
        point = myX[x]
        minDistance = 99999999999
        for i in range(centroids.shape[0]):
            distance = distanceBetween(point,centroids[i]) 
            if distance < minDistance:
                idx[x] = i
                minDistance = distance
    return idx

def computeMeans(myX, myidx):
    subX = []
    for x in range(len(np.unique(myidx))):
        subX.append(np.array([myX[i] for i in range(myX.shape[0]) if myidx[i] == x ]))
    newMeanArray = []
    for i in range(len(subX)):
        if subX[i].size == 0:
            # print("caught" + str(subX[i]))
            subX[i] = np.array([[np.random.randint(low = myX[:,0].min(), high = myX[:,0].max()), np.random.randint(low = myX[:,1].min(), high = myX[:,1].max())]])      
            # print(subX[i])
            # print(subX[i-1])
    return np.array([np.mean(subX[i],axis=0) for i in range(len(subX))])


def runKtimes(myX, centroids, myK ,iterations):
    centroid_history = []
    centroid_history.append(centroids)
    for iterations in range(iterations):
        idx = findClosestCentroids(myX,centroids)
        centroids = computeMeans(myX,idx)
        centroid_history.append(centroids)
    # print(centroid_history[-1])
    return centroid_history , idx
           
example1 = np.array([[1,2],[0,0]])
example2 = np.array([[3,4],[5,6],[2,3]])

def plotData(centroid_history, myX, myidx=None):
    colors = ['b','g','gold','darkorange','salmon','olivedrab','black','white','pink','violet']
    for i in range(len(np.unique(myidx))):
        tempX = [] 
        tempY = []           
        for x in range(myX.shape[0]):
            if(myidx[x]==i):
                tempX.append(myX[x][0])
                tempY.append(myX[x][1])
        plt.plot(tempX,tempY,'o',color= colors[i])
        
    for j in range(centroid_history[0].shape[0]): 
        tempX = [] 
        tempY = []  
        for k in range(len(centroid_history)):      
            # print(k)
            # print(j)
            print(centroid_history[k])
            print(centroid_history[0])
            # print(centroid_history[0].shape[0])
            if j <len(centroid_history[k]): 
                tempX.append(centroid_history[k][j][0])
                tempY.append(centroid_history[k][j][1])
            plt.plot(tempX,tempY,'rx--')
    plt.show()


def randomCentroids(myX, myK):
    rand_indices = sample(range(0,myX.shape[0]),myK)
    return np.array([myX[i] for i in rand_indices])

def kCluster(myX, K , correctionaliterations, newCentroidIterations):
    for iteration in range(newCentroidIterations):
        for i in range(K):
            randomcentroids = randomCentroids(myX,i)
            finalCentroidArray, finalidx = runKtimes(myX, randomcentroids, i, correctionaliterations)
            plotData(finalCentroidArray,myX,finalidx)

# kCluster(X,10,3,1)

imgFile = 'ex7/bird_small.png'
A = imageio.imread(imgFile)
A = A/255
A = A[:,:,:3]
A = A.reshape(-1,3)
kPoints = 1
initialCentroids = randomCentroids(A,kPoints)
centroidHistory, newIdx = runKtimes(A, initialCentroids , kPoints ,10)
newIdx = findClosestCentroids(A,centroidHistory[-1])
finalCentroid = centroidHistory[-1]
final_image = np.zeros((newIdx.shape[0],3))
for i in range(final_image.shape[0]):
    final_image[i] = finalCentroid[int(newIdx[i])]
plt.imshow(final_image.reshape(128,128,3))
plt.show()


pcaDataFile = 'ex7/ex7data1.mat'
pcaData = scipy.io.loadmat(pcaDataFile)
pcaData = pcaData['X']
print(pcaData)
plt.figure(figsize=(7,5))
plot = plt.scatter(pcaData[:,0],pcaData[:,1])
plt.grid(True)

def featureNormalize(myX):
    means = np.mean(myX,axis=0)
    myX_norm = myX - means
    stds = np.std(myX_norm, axis = 0)
    myX_norm = myX_norm/ stds
    return means, stds, myX_norm

def getUSV(myX_norm):
    #covariance matrix
    cov_matrix = myX_norm.T.dot(myX_norm)/myX_norm.shape[0]
    U,S,V = scipy.linalg.svd(cov_matrix, compute_uv = True, full_matrices = True)
    return U,S,V

# Getting the USV value of example data 
means, stds, pcaData_norm = featureNormalize(pcaData)
U, S, V = getUSV(pcaData_norm)
# print("PCA")
# print(pcaData.shape)

# print(getUSV(pcaData_norm)[0])
# print(getUSV(pcaData_norm)[1])
# print(getUSV(pcaData_norm)[2])

def plotPCA(data, dataMeans, Ueigenvector ,Seigenvalue):
    plt.scatter(data[:,0],data[:,1], facecolor='white' , edgecolor='b')
    
    plt.plot([dataMeans[0],dataMeans[0]+S[0]*U[0,0]],
            [dataMeans[1],dataMeans[1]+S[0]*U[0,1]],
            color = "red" ,  linewidth = '4',
            label = "Principal Component"
            )
    plt.plot([dataMeans[0],dataMeans[0]+S[1]*U[1,0]],
            [dataMeans[1],dataMeans[1]+S[1]*U[1,1]],
            color="black", linewidth='2',
            label =' Second Principal Component')
    plt.grid(True)
    plt.legend()
    plt.show()

    
#plotPCA(pcaData,means,U,S)
  
#data * eigenvalue
def dataProjection (data, Ueigenvector, Kindex):
    Ureduced = Ueigenvector[:,: Kindex]
    projection =data.dot(Ureduced)
    return projection
print("Projecting Data")
dataProjected =  dataProjection(pcaData_norm, U, 1)
#print(dataProjected)

def recoverData( projectedData, Ueigenvector, K):
    Ureduced = Ueigenvector[:,:K]
    approxData = projectedData.dot(Ureduced.T)
    return approxData

approxData = recoverData(dataProjected, U, 1)
print(approxData)

def plotrecoveredData(orgData,recoveredData):
    plt.scatter(orgData[:,0], orgData[:,1], facecolor = 'white' ,edgecolor = 'red', label = 'Original Data')
    plt.scatter(recoveredData[:,0], recoveredData[:,1], facecolor = 'white', edgecolor= 'blue', label = 'Reconstrued')
    for x in range(orgData.shape[0]):
        plt.plot([orgData[x,0],recoveredData[x,0]],[orgData[x,1],recoveredData[x,1]],color = 'black', linestyle = '--',)
    plt.legend()
    plt.title('Data Reconstructed from 1st PCA')
    plt.grid(True)
    plt.show()

# Showing the Recovered Data from the First Principal Component
#plotrecoveredData(pcaData_norm, approxData)

#2.4 Remaking Faces
faceDataAddress = 'ex7/ex7faces.mat'
faceData = scipy.io.loadmat(faceDataAddress)
faceData = faceData['X']
faceMeans, faceSTD, face_norm = featureNormalize(faceData)

def constructFaces(row):
    width, height= 32, 32
    square = row.reshape(width,height)
    return square.T
face1 = constructFaces(faceData[63])
plt.imshow(face1)
plt.show()

def displayData(data, nrows = 10, ncols = 10):
    width = 32
    height = 32
    bigGrid = np.zeros((height*nrows,width*ncols))
    irow, icol = 0 , 0
    for index in range(nrows*ncols):
        if icol == ncols :
            irow += 1
            icol = 0
        picture = constructFaces(data[index])
        bigGrid[irow*height:irow*height+picture.shape[0],icol*width:icol*width+picture.shape[1]] = picture
        icol += 1
    fig = plt.figure(figsize = (30,30))
    finalImg = Image.fromarray(bigGrid)
    plt.imshow(finalImg, cmap = cm.hot) 
    plt.show()
        
#displayData(faceData)
faceU, faceS, faceV = getUSV(face_norm)
print(faceU.shape)
faceU, faceV = faceU*255, faceV*255
#displayData(faceU[:,:1024].T,10,10)

mold = dataProjection(face_norm,faceU,36)
facemold = recoverData(mold,faceU,36)
#displayData(facemold)

def both(data1,data2,k,l):
    for i in range(k):
        picture1 = constructFaces(data1[l+i])
        picture2 = constructFaces(data2[l+i])
        img1 = Image.fromarray(picture1)
        img2 = Image.fromarray(picture2)
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2,cmap = cm.Greys_r)
        plt.show()
both(faceData,faceU[:,:1000].T,6,56)

finalmeans, finalstds, final_norm = featureNormalize(A)
U,S,V = getUSV(final_norm)
z = dataProjection(final_norm,U,2)
subX = []


Address = 'ex7/bird_small.png'
A = imageio.imread(Address)
print(A.shape)
A = A/255
A = A.reshape(-1,3)
print(A.shape)
initialCentroid = randomCentroids(A, 16)
centroidHistory, newIdx = runKtimes(A,initialCentroid, 16,10)
mean, std, A_norm = featureNormalize(A)
U,S,V = getUSV(A_norm)
print(U.shape)
z = dataProjection(A,U,2)
print(z.shape)
figure = plt.figure(figsize=(8,8))
plt.plot(z[:,0],z[:,1],'.')
plt.show()
for x in range(len(np.unique(newIdx))):
    subX.append(np.array([z[i] for i in range(A.shape[0]) if newIdx[i] == x]))
    print(x)

for x in range(len(subX)):
    newX= []
    colors = ['b','g','gold','darkorange','salmon','olivedrab','black','gold','pink','violet','ivory','navy','yellow','tomato','orchid','maroon','lime','khaki']
    newX = subX[x]
    plt.plot(newX[:,0],newX[:,1],'.',color = colors[x],alpha = 0.3)

plt.xlabel('z1', fontsize = 14)
plt.ylabel('z2', fontsize =14)
plt.title('PCA Projection Plot', fontsize = 10)
plt.grid(True) 
plt.show()


