import numpy as np
import math
import mdp
import re
import random
import matplotlib.pyplot as plt
import dtw
import fastdtw
import time
import random
import pickle


def coswave(length, reps, noise=0.01):
    '''

    :param length: Amount of points that the wave should consist of
    :param reps: Amount of periods in the wave.
    :param noise: Standard deviation of gaussian noise added to the wave. Default is 0.01 instead of 0, because np.random.normal crashes on 0 standard deviation.
    :return: A sine wave starting at 0 and ending at 0 consisting of length points, reps periods and with $noise$ gaussian noise. The points of the wave are two dimensional, a simple axis and the actual sine value
    '''
    end=2*reps*math.pi
    stepsize=end/(length-1.)
    wave=[]
    for i in range(length):
        wave.append((i,-np.cos(i*stepsize)*0.5+0.5+np.random.normal(0,noise)))
    return wave

def trianglewave(length,reps,noise=0.01):
    '''

    :param length: Amount of points that the wave should consist of
    :param reps: Amount of periods in the wave.
    :param noise: Standard deviation of gaussian noise added to the wave. Default is 0.01 instead of 0, because np.random.normal crashes on 0 standard deviation.
    :return: A triangle wave starting at 0 and ending at 0 consisting of length points, reps periods and with $noise$ gaussian noise. The points of the wave are two dimensional, a simple axis and the actual triangle wave value.
    '''
    end=reps
    wave=[]
    stepsize=end/(length-1.)
    for i in range(length):
        wave.append((i,1.-abs(1-2*i*stepsize%2)+np.random.normal(0,noise)))
    return wave

def zigzagwave(length,reps,noise=0.01,wavefactor=0.3):
    minilength=length/reps
    miniwave=trianglewave(length,reps*7,noise=0.0001)
    mainwave=trianglewave(length,reps,noise=noise)
    output=[miniwave[i][1]*wavefactor+mainwave[i][1] for i in range(len(miniwave))]
    return output

def stepwave(length,reps,noise=0.01):
    stepamount=3
    listerino=[]
    for i in range(length):
        value=float((reps*i)%length)/length

        point=round(value*3,0)/3.+np.random.normal(0,noise)
        listerino.append(point)
    return listerino


def rectwave(length,reps,noise=0.01):
    '''

    :param length: Amount of points that the wave should consist of
    :param reps: Amount of periods in the wave.
    :param noise: Standard deviation of gaussian noise added to the wave. Default is 0.01 instead of 0, because np.random.normal crashes on 0 standard deviation.
    :return: A rectangular wave starting at 0 and ending at 0 consisting of length points, reps periods, and with $noise$ gaussian noise. Note that in the rising sections it isn't vertical, but sloped.
    '''
    end=reps
    wave=[]
    stepsize=end/(length-1.)
    for i in range(length):
        current=i*stepsize%1
        if current <0.5: #going straight up
            wave.append((i,np.random.normal(0,noise))) #going up
        else:
            wave.append((i,1+np.random.normal(0,noise))) #straight at 1

    return wave

def multiblock(typeslist,length,reps,noise=0.01):
    '''
    :param typeslist: What types of waves are represented. List in the format ['triangle', 'sine', 'rect']
    :param length: Amount of points
    :param reps: Amount of periods
    :param noise: Gaussian noise added.
    :return: A list of multidimensional data points. First point corresponds to point on a wave of type specified by typeslist[0], second points a wave specified by typeslist[1] and so on and so forth
    '''
    typelists=[]
    for i in typeslist:
        if i=='rect':
            typelists.append([i[1] for i in rectwave(length,reps,noise)])
        elif i=='triangle':
            typelists.append([i[1] for i in trianglewave(length, reps,noise)])
        elif i=='sine':
            typelists.append([i[1] for i in coswave(length, reps, noise)])
        elif i=='zigzag':
            typelists.append([i for i in zigzagwave(length,reps,noise)])
        elif i=='step':
            typelists.append([i for i in stepwave(length,reps,noise)])
        else:
            print('type '+ i + ' not supported')
    finallist=[]
    for i in range(length):
        finallist.append([])
        for x in range(len(typeslist)):
            finallist[-1].append(typelists[x][i])
    return finallist

def toydata2(noise=0.01,numblocks=10,blocksize=500,variance=0.1,periodicity=3,periodicityvariance=1,numfeatures=20):
    testdata=[]
    realboundaries=[0]
    for i in range(numblocks):
        size=int(blocksize*math.exp(-variance+2*variance*random.random()))
        realboundaries.append(size+realboundaries[-1])
        periods=int(periodicity-periodicityvariance+2*periodicityvariance*random.random())
        for j in multiblock([random.choice(toydatashapes) for k in range(numfeatures)],size,periods,noise):
            testdata.append(j)
    testdata=np.array(testdata)
    return testdata,realboundaries

def toydata1(noise,expansion):
    '''
    A predefined block of toydata. Consists of 3 segements, one of length 120 with 4 repetitions, next with 80 points and 2 periods, and then one with 280 length and 6 periods.
    :param noise: Gaussian noise
    :param expansion: Expansion parameter for toydata.
    :return: List of threedimensional points. in total 480 points
    '''
    testdata=[]
    block1 = multiblock(['rect', 'triangle', 'sine'], 120, 4, noise)
    block2 = multiblock(['triangle', 'sine', 'rect'], 80, 2, noise)
    block3 = multiblock(['sine', 'rect', 'triangle'], 280, 6, noise)
    for i in block1:
        testdata.append(i)
    for i in block2:
        testdata.append(i)
    for i in block3:
        testdata.append(i)
    testdata=np.array(testdata)
    expander=mdp.nodes.PolynomialExpansionNode(expansion)#We use polynomials of order 2 to get any nonlinear relationships
    testdata=expander.execute(testdata)
    return testdata

def readSample(number):
    """
    Reads from the samples from 86, only enter the correct number.
    """
    string=""
    if number<10:
        string= '86/86_0' + str(number) + ".amc"
    else:
        string="86/86_"+str(number)+".amc"
    f=open(string,'r')
    f.readline() #A bit of a hack, but I don't want to deal with the first three lines here.
    f.readline()
    f.readline()
    container=[]
    for line in f:
        asdf=line.rstrip()
        if asdf.isdigit():
            container.append([])
        else:
            for x in asdf.split(" "):
                if re.search('[0-9]',x):
                    container[-1].append(float(x))
    return container

def parse_segmentation(filename):
    '''
    Takes a filename such as mocap.txt and returns a dictionary with the transition points
    :param filename:
    :return:
    '''
    subsections=[[]]
    for line in open(filename,'r'):
        linestripped=line.strip()
        if linestripped=='':
            subsections.append([])
        else:
            subsections[-1].append(linestripped)
    dicterino={}
    for subsegmentation in subsections:
        dicterino[subsegmentation[0]]=[]#The first element in our subsections is the name. We use it as a dictionary key
        for line in subsegmentation[1:]:
            name,lower,middle,upper=line.split(' ')
            dicterino[subsegmentation[0]].append({'name':name, 'lower':int(lower), 'middle': int(middle), 'upper':int(upper)})
    return dicterino


segmentations=parse_segmentation('86/mocap.txt')

def simmatrix1(numfeatures, testerino, timesteps):
    '''
    Makes a similarity matrix using inner product from slow feature data, using $numfeatures$ features, and $timesteps$ timesteps to make comparison vectors
    :param numfeatures: Amount of features to be used for similarity matrix
    :param testerino: Calculated slow features
    :param timesteps: Amount of timesteps to be compared for similarity matrix
    :return: Similarity matrix based on inner product
    '''
    vecmat = vectormatrix(numfeatures, testerino, timesteps)
    selfsimmatrix = vecmat * vecmat.transpose()
    return selfsimmatrix


def simmatrix2(numfeatures,testerino,timesteps):
    '''
    Makes a similarity matrix using normalized euclidean distance from slow feature data, using $numfeatures$ features, and $timesteps$ timesteps to make comparison vectors
    :param numfeatures: Amount of features to be used for similarity matrix
    :param testerino: Calculated values of slow features
    :param timesteps: Amount of timesteps for comparison in similarity matrix
    :return: Similarity matrix with similarity calculated using e^{-distance/delta}. Atm delta is set manually. Possibly can be changed to computed from data
    '''
    vecmat=vectormatrix(numfeatures,testerino,timesteps)
    delta=0.4
    distmatrix=[]
    for row1 in vecmat:
        distmatrix.append([])
        for row2 in vecmat:
            distmatrix[-1].append(expdistance(row1,row2,delta))
    return np.matrix(distmatrix)

def slowestdtwmatrix(numfeatures,testerino,timesteps):
    matrixerino=[]
    finalmatrix=[]
    for i in range(len(testerino[:-timesteps])):
        matrixerino.append([])
        for y in range(numfeatures):
            helpvector=[]
            for j in range(timesteps):
                helpvector.append(testerino[i+j][y])
            matrixerino[-1].append(np.matrix(np.array(helpvector).reshape(-1,1)))
    for i in range(len(matrixerino)):
        vectors1=matrixerino[i]
        finalmatrix.append([])
        print(str(i)+'/'+str(len(matrixerino)))
        for j in range(len(matrixerino)):
            if j%100==0:
                print(j)
            vectors2=matrixerino[j]
            if i<=j:
                dtwsum=0
                for k in range(len(vectors1)):
                    dtwsum+=dtw.dtw(vectors1[k],vectors2[k],dist=lambda x,y:np.linalg.norm(x-y))[0]
                finalmatrix[-1].append(math.exp(-dtwsum)) #setting finalmatrix[i][j]
            else:
                finalmatrix[-1].append(finalmatrix[j][i])#We know this already exists, since j<=i and we've done all the [i][:]
    return np.matrix(finalmatrix)

def fastdtwmatrix(numfeatures,testerino,timesteps):
    matrixerino=[]
    finalmatrix=[]
    for i in range(len(testerino[:-timesteps])):
        matrixerino.append([])
        for y in range(numfeatures):
            helpvector=[]
            for j in range(timesteps):
                helpvector.append(testerino[i+j][y])
            matrixerino[-1].append(np.matrix(np.array(helpvector).reshape(-1,1)))
    for i in range(len(matrixerino)):
        vectors1=matrixerino[i]
        finalmatrix.append([])
        print(str(i)+'/'+str(len(matrixerino)))
        for j in range(len(matrixerino)):
            if j%100==0:
                print(j)
            vectors2=matrixerino[j]
            if i<=j:
                dtwsum=0
                for k in range(len(vectors1)):
                    dtwsum+=fasdtw.dtw(vectors1[k],vectors2[k],dist=lambda x,y:np.linalg.norm(x-y))[0]
                finalmatrix[-1].append(dtwsum) #setting finalmatrix[i][j]
            else:
                finalmatrix[-1].append(finalmatrix[j][i])#We know this already exists, since j<=i and we've done all the [i][:]
    return np.matrix(finalmatrix)

def hybriddtwmatrix(numfeatures,testerino,timesteps,windowsize):
    matrixerino=[]
    delta=0.4
    finalmatrix=[]
    for i in range(len(testerino[:-timesteps])):
        matrixerino.append([])
        for y in range(numfeatures):
            helpvector=[]
            for j in range(timesteps):
                helpvector.append(testerino[i+j][y])
            matrixerino[-1].append(np.matrix(np.array(helpvector).reshape(-1,1)))
    for i in range(len(matrixerino)):
        vectors1=matrixerino[i]
        finalmatrix.append([])
        print(str(i)+'/'+str(len(matrixerino)))
        for j in range(len(matrixerino)):
            vectors2=matrixerino[j]
            if i<=j:
                if i+windowsize<=j:
                    newvector1=[vector.tolist() for vector in vectors1]
                    newvector2=[vector.tolist() for vector in vectors2]
                    newvector1=[item for sublist in newvector1 for item in sublist]
                    newvector2 = [item for sublist in newvector2 for item in sublist]
                    newvector1=np.array(newvector1)
                    newvector2=np.array(newvector2)
                    finalmatrix[-1].append(expdistance(newvector1,newvector2,delta))
                else:
                    dtwsum=0
                    for k in range(len(vectors1)):
                        dtwsum+=dtw.dtw(vectors1[k],vectors2[k],dist=lambda x,y:np.linalg.norm(x-y))[0]
                    finalmatrix[-1].append(math.exp(-dtwsum)) #setting finalmatrix[i][j]
            else:
                finalmatrix[-1].append(finalmatrix[j][i])#We know this already exists, since j<=i and we've done all the [i][:]
    return np.matrix(finalmatrix)
def expdistance(vector1,vector2,delta):
    '''
    Calculates distance between two vectors, using formula e^{-(vector1-vector2)/delta)}
    :param vector1:
    :param vector2:
    :param delta:
    :return:
    '''
    distance=np.linalg.norm(vector1-vector2)
    return np.exp(-distance/2*delta**2)

def vectormatrix(numfeatures, testerino, timesteps):
    '''
    Takes testdata, timesteps, and the number of features and makes a matrix of timeseries vectors. If numfeatures=2, timesteps=2, first row would be [firstfeature[0],secondfeature[0],firstfeature[1],secondfeature[1]]
    :param numfeatures: Number of features
    :param testerino: Slow feature data
    :param timesteps: Amount of timesteps considered for vector data
    :return: Matrix of timeseries vectors. Should be len(testdata)-timesteps x timesteps*numfeatures size
    '''
    vectorized = []
    offset = timesteps * numfeatures
    for i in range(len(testerino[:-timesteps])):
        vectorized.append([])
        for j in range(timesteps):
            for y in range(numfeatures):
                vectorized[-1].append(testerino[i + j][y])
    vectormatrix = np.matrix(vectorized)
    return vectormatrix

def reorder(intlist):
    '''
    Takes a list of markers referring to groups, and renames the clusters, so that cluster 1 has name 1 etc.
    :param intlist: List of integers from 0 to n
    :return: renamed list
    '''
    translator={}
    elements=list(set(intlist))#Removing duplicates
    indexcount={}
    for i in elements:
        indexcount[i]=[0,0]
    for i in range(len(intlist)):
        current=indexcount[intlist[i]]
        indexcount[intlist[i]]=[current[0]+1.,current[1]+i]
    indexaverage={}
    for key in indexcount:
        indexaverage[key]=indexcount[key][1]/indexcount[key][0]
    counter=0
    sortedlist=sorted(indexaverage.keys(), key=lambda a:indexaverage[a])
    for key in sortedlist:
        translator[key]=counter
        counter+=1
    newintlist=[]
    for i in intlist:
        newintlist.append(translator[i])
    return newintlist

def evaluate_boundary(boundaries,binarylist):
    '''
    Takes a list of 0s and 1s, and a boundary value, and returns the percentage of errors, if we assume that the list is split according to these boundaries
    :param boundary: integer of where to place boundary
    :param list: List of class assignments. Note: It has to have reorder run over it beforehand.
    :return: error percentage
    '''
    boundarylower=[0]+boundaries
    boundaryupper=boundaries+[len(binarylist)]
    length=len(binarylist)
    errors=0
    currentgroup=0
    for j in range(len(boundaries)+1):
        for i in binarylist[boundarylower[j]:boundaryupper[j]]:
            if not(i==j):
                errors+=1.
    return errors/len(binarylist)


def find_boundary(binarylist):
    bestboundary=0
    bestboundaryvalue=100000000000
    for i in range(len(binarylist)):
        boundaryvalue=evaluate_boundary([i],binarylist)
        if boundaryvalue<bestboundaryvalue:
            bestboundaryvalue=boundaryvalue
            bestboundary=i
    return bestboundary,bestboundaryvalue

def find_boundaries2(binarylist):
    '''
    Find double boundaries. Don't htink we will ever do quadruple boundaries so ehh.
    :param binarylist:
    :return:
    '''
    bestboundaryvalue=2
    bestboundary=[0,0]

    for i in range(len(binarylist)-1):
        newi=i+1
        for j in range(len(binarylist[newi:])-1):
            newj=newi+j+1
            boundaryvalue=evaluate_boundary([newi,newj],binarylist)
            if boundaryvalue<bestboundaryvalue:
                bestboundaryvalue=boundaryvalue
                bestboundary=[newi,newj]
    return (bestboundary,bestboundaryvalue)

def find_boundariesold(binarylist):
    '''
    Same thing as find_boundaries2, but it can also find boundaries that correspond to two groups.
    '''
    bestboundaryvalue=2
    bestboundary=[0,0]

    for i in range(len(binarylist)):
        for j in range(len(binarylist[i:])+1):
            boundaryvalue=evaluate_boundary([i,i+j],binarylist)
            if boundaryvalue<bestboundaryvalue:
                bestboundaryvalue=boundaryvalue
                bestboundary=[i,i+j]
    return (bestboundary,bestboundaryvalue)

def find_boundarieskmeans(binarylist,numclusters,clusterimportance):
    #We're making a custom k-means clustering
    def customdifference(point1,point2):
        if point1[0]==point2[0]:
            modifier=0
        else:
            modifier=clusterimportance
        return abs(point1[1]-point2[1])+modifier

    size=len(binarylist)/numclusters
    centers=[(i,size*i+size/2) for i in range(numclusters)] #Initial guess is evenly spread centroids. This should conform to our expectation
    for i in range(10):
        clusters = [([],centers[i]) for i in range(numclusters)]
        for j in range(len(binarylist)):
            value=binarylist[j]
            point=[value,j]
            bestcenter=min(centers,key=lambda x:customdifference(point,x)) #Allocation step
            for cluster,center in clusters:
                if center==bestcenter:
                    cluster.append(point)
            #Updating center
        centers=[]
        for cluster,center in clusters:
            if len(cluster)>0:
                clustervalue=np.mean([x[0] for x in cluster])
                numericcenter=np.mean([x[1] for x in cluster])
            else:
                clustervalue=int(round(random.random()*numclusters))
                numericcenter=random.random()*len(binarylist)
            centers.append([int(round(clustervalue)),numericcenter])
        #Here we get boundaries from the centers
        boundaries=boundariesfromcenters(clusters)
    return boundaries
def boundariesfromcenters(clusterslist):
    '''
    Takes a clusterslist as defined in findboundarieskmeans, and gives out actual decision boundaries. We don't simply choose the middle, but instead choose a point based on the size of
    of the clusters, with the boundary being closer to the smaller clustercenter. The idea is to better represent situations like: [1,1,2,2,2,2,2,2,2,2,2,2,2] If we simply picked the middle between the two
    centers there, we would misclassify some 2s.
    :param clusterslist: clusterslist as in boundarieskmeans
    :return: List of decisionboundaries.
    '''
    boundaries=[]
    sortedclusterslist=sorted(clusterslist, key=lambda x:x[0][1])
    for i,(cluster,center) in enumerate(sortedclusterslist[:-1]):
        j=i+1
        lengthi=len(cluster)
        lengthj=len(sortedclusterslist[j][0])
        boundarypoint=(lengthj*center[1]+lengthi*sortedclusterslist[j][1][1])/(lengthj+lengthi)
        boundaries.append(boundarypoint)
    return boundaries

def clusteringheuristic1(binarylist,windowsize):
    '''
    Takes a binarylist of clusterings, and then does some cheap improvements to get them to more closely align with sharp boundary points. This should eliminate the easy points, that are probalby outliers.
    :param binarylist: A list of integers. Should be 0-x
    :return: Another binarylist, with some values chane
    '''
    originallist=[x for x in binarylist] #Just copying so no python = list shenanigans can happen
    threshold=.95 #If this threshold of numbers in your area area are of a certain cluster, you probably are too.
    newlist=[x for x in binarylist]
    clusters=list(set(originallist))
    for point in range(len(originallist)-windowsize):
        consideredlist=originallist[point:point+windowsize]
        centerpoint=windowsize/2+point
        mostcommon=max(clusters,key=lambda x: consideredlist.count(x))
        if consideredlist.count(mostcommon)>windowsize*threshold:
            newlist[centerpoint]=mostcommon
    mostcommonstart=max(clusters,key=lambda x: originallist[:windowsize].count(x))
    if originallist[:windowsize].count(mostcommonstart)>windowsize*threshold:
        changestart=True
    else:
        changestart=False #I know I could do this all in one line
    mostcommonend = max(clusters, key=lambda x: originallist[-windowsize:].count(x))
    if originallist[-windowsize:].count(mostcommonend)>windowsize*threshold:
        changeend=True
    else:
        changeend=False #I know I could do this all in one line

    for point in range(windowsize/2):
        if changestart:
            newlist[point]=mostcommonstart
        if changeend:
            newlist[-point]=mostcommonend
    return newlist

def clusteringheuristic2(binarylist,windowsize,simmatrix):
    '''
    Supposed to catch the trickier cases of like 1,1,1,1,2,3,3,3,3,3,3 where to assign that 2. We take the distance in the similarity matrix
    :param binarylist: A binarylist of clusterings
    :param windowsize:
    :return:
    '''
    exclusionthreshold=0.2 #If you're not more than 0.2 times the expected windowsize, you're probably badly clasified in terms of decision boundaries
    inclusionthreshold=0.3
    potentials=[]
    clusters=list(set(binarylist))
    newlist=[i for i in binarylist]
    for point in range(len(binarylist)-windowsize):
        actualpoint=point+windowsize/2
        if binarylist[point:point+windowsize].count(binarylist[actualpoint])<windowsize*exclusionthreshold:
            sortbinary=sorted(clusters,key=lambda x:binarylist[point:point+windowsize].count(x))
            highest=sortbinary[-1]
            secondhighest=sortbinary[-2]
            #TODO make distance in simmatrix and just assign the point to closest and next closest, as long as those two are reasonably big.
            if binarylist[point:point+windowsize].count(highest)>windowsize*inclusionthreshold and binarylist[point:point+windowsize].count(secondhighest)>windowsize*inclusionthreshold: #Both other options are reasonably big
                highestclusterino=[i for i,j in enumerate(binarylist) if j==highest]
                secondhighestclusterino=[i for i,j in enumerate(binarylist) if j==secondhighest]
                highestdistance=0
                secondhighestdistance=0
                pointline=simmatrix[actualpoint]
                for line in highestclusterino:
                    matrixline=simmatrix[line]
                    highestdistance+=np.linalg.norm(matrixline-pointline)
                highestdistance/=float(len(highestclusterino))
                for line in secondhighestclusterino:
                    matrixline=simmatrix[line]
                    secondhighestdistance+=np.linalg.norm(matrixline-pointline)
                secondhighestdistance/=float(len(secondhighestclusterino))
                if highestdistance>=secondhighestdistance:
                    newlist[actualpoint]=highest
                else:
                    newlist[actualpoint]=secondhighest
    return newlist


def smoldata(sample):
    '''
    Takes a sample number and returns the data from CMU dataset, but cut down to a quarter of the size. Simply takes every 4th point.
    :param sample: Sample number
    :return:
    '''
    data=readSample(sample)
    length=len(data)/4
    realdata=[]
    for i in range(length):
        realdata.append(data[4*i])
    return np.array(realdata)


def findbestcenterapproach(binarylist,numclusters,windowsize):
    centers=[]
    for i in range(numclusters):

        bestcenter=0
        bestcentervalue=0
        for j in range(len(binarylist)-windowsize):
            centervalue=binarylist[j:j+windowsize].count(i)
            if centervalue>bestcentervalue:
                bestcentervalue=centervalue
                bestcenter=j
        centers.append((bestcenter,bestcentervalue))
    return centers

# NOTE THIS IS NOT HACA
othersegmentationstuff=[[137,145,285,314,488,534,615,628,817 ,1018]
,[ 247,273,457,480,654,673,789,1178,1491,1510,1631,1803,1881,1974,2067,2212,2220,2384,2414],
[10,233,471,487,609,618,847,877,971,1172,1339,1529,1571,1742,1779],
[248,  572,  580,  834,  849, 1035, 1258, 1469, 1476, 1664, 1745, 1790, 1891, 1990, 2054, 2260, 2288],
    [ 191,  202,  373,  416,  567,  818,  845,  971,  986, 1126 ,1140, 1292, 1459, 1639, 1837, 1853],
    [6,  251,  399,  414,  624,  637,  786,  794,  970,  977, 1144, 1376, 1540, 1721, 1738, 1896, 1984, 2230, 2238],
    [260,  287,  462,  479,  638,  650,  903,  915, 1103, 1117, 1287, 1312, 1417, 1435, 1517, 1741, 1945],
[7,  272,  293,  464,  479,  663,  681,  826,  833,  987, 1195, 1221, 1402, 1417, 1586, 1789, 1800, 2027, 2047],
 [   248 ,  347 ,  526   ,538  , 714 ,  738  , 901   ,925]] #Cuts performed on samples 1-9 by HACA stuff 
toydatashapes=['rect', 'triangle', 'sine','zigzag','step']
# testdata=[0 for i in range(500)]+[int(random.random()*2) for i in range(200)]+[1 for i in range(500)]+[2 for i in range(500)]
# print(find_boundarieskmeans(testdata,3,600))
# asdf=pickle.load(open('Logs/manywindows32','rb'))
# asdf=reorder(asdf[3])
# boundaryscenter=findbestcenterapproach(asdf,10,250)
# for point,quality in boundaryscenter:
#     plt.plot([point,point],[0,10],c='red')
# centers=find_boundarieskmeans(asdf,10,250 )
# centers.sort(key=lambda x:x[1])
# actualboundaries=[(centers[i][1]+centers[i+1][1])/2. for i in range(len(centers[1:]))]
# for boundary in actualboundaries:
#     plt.plot([boundary,boundary],[0,10],c='blue')
# plt.plot(range(len(asdf)),asdf,'x')
# plt.show()
# testtestdaata=[1 for x in range(500)]+[3 for x in range(51)]+[1 for x in range(500)]+[2 for x in range(1000)]
# asdf=clusteringheuristic1(testtestdaata,1000)
# print(asdf[450:650])


