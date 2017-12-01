import generate_data
import sklearn.cluster as cluster
import mdp
import numpy as np
import math
import random
import time
import dtw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def logcounter(filename):
    file = open('Logs/'+filename, 'r')
    counter = int(file.read())
    file.close()
    file = open('Logs/'+filename, 'w')
    file.write(str(counter + 1))
    file.close()
    return counter

def GetSFAValuesFromData(data,numfeatures=3, reducenumber=10):
    testdata=pca_reduce(data,reducenumber)
    sfanode=mdp.nodes.SFANode()
    expansionnode = mdp.nodes.PolynomialExpansionNode(2)
    testdata = expansionnode.execute(testdata)
    print(testdata.shape)
    sfanode.train(testdata)
    sfanode.stop_training()
    feature_values = sfanode.execute(testdata, n=numfeatures)
    return feature_values

def GetClusteringFromData(data, numclusters=2,numfeatures=3, reducenumber=10, timesteps=20):
    '''
    Same as GetClustering, except inputs arbitrary data. Has to be formatted correctly though, so big np.array() with rows being data points
    '''
    featurevalues=GetSFAValuesFromData(data,numfeatures,reducenumber)
    distmatrix = generate_data.simmatrix2(numfeatures, featurevalues, timesteps)
    clusterer = cluster.SpectralClustering(n_clusters=numclusters, affinity='precomputed')
    binarylist = clusterer.fit_predict(distmatrix)
    binarylist = generate_data.reorder(binarylist)
    if numclusters == 2:
        boundary = generate_data.find_boundary(binarylist)
    else:
        boundary = generate_data.find_boundariesold(binarylist)
    return boundary, binarylist, featurevalues

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
    for i in range(25):
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

def FullBatch(testdata,numclusters,w, numfeatures=5,reducenumber=10,timesteps=20):
    print('Dimensions of data are: '+str(testdata.shape))
    testdata=pca_reduce(testdata,reducenumber)
    sfanode=mdp.nodes.SFANode()
    expansionnode = mdp.nodes.PolynomialExpansionNode(2)
    testdata = expansionnode.execute(testdata)
    print('After expansion of data, we have dimensions: '+str(testdata.shape))
    sfanode.train(testdata)
    sfanode.stop_training()
    print('Constructing Feature Vectors')
    feature_values=sfanode.execute(testdata,n=numfeatures)
    print('Constructing Simlarity Matrix')
    S=MakeSimilarityMatrix(feature_values,w,config='euc0s',numfeatures=numfeatures,timesteps=timesteps)

    clusterer = cluster.SpectralClustering(n_clusters=numclusters, affinity='precomputed')
    print('Clustering')
    clusterlabellist = clusterer.fit_predict(S)
    clusterlabellist = generate_data.reorder(clusterlabellist)
    clusterlabellist = clusteringheuristic1(clusterlabellist,int(w))
    clusterlabellist = clusteringheuristic2(clusterlabellist,int(w),S)
    print('Finding boundaries')
    boundaries=find_boundarieskmeans(clusterlabellist,numclusters,w/2)

    return boundaries

def noSFAFullBatch(testdata,numclusters,w, numfeatures=5,reducenumber=20,timesteps=20):
    print('Dimensions of data are: '+str(testdata.shape))
    feature_values=pca_reduce(testdata,5)

    print('Constructing Similarity Matrix')
    S=MakeSimilarityMatrix(feature_values,w,config='euc0s',numfeatures=numfeatures,timesteps=timesteps,Delta=30)

    clusterer = cluster.SpectralClustering(n_clusters=numclusters, affinity='precomputed')
    print('Clustering')
    plt.matshow(S)

    clusterlabellist = clusterer.fit_predict(S)
    print('clustering2')
    clusterlabellist = generate_data.reorder(clusterlabellist)
    clusterlabellist = clusteringheuristic1(clusterlabellist,int(w))
    clusterlabellist = clusteringheuristic2(clusterlabellist,int(w),S)
    print('Finding boundaries')
    boundaries=find_boundarieskmeans(clusterlabellist,numclusters,w/2)

    return boundaries
def noSFAFullBatch2(testdata,numclusters,w, numfeatures=5,reducenumber=20,timesteps=20):
    print('Dimensions of data are: '+str(testdata.shape))
    feature_values=testdata

    print('Constructing Similarity Matrix')
    S=MakeSimilarityMatrix(feature_values,w,config='euc0s',numfeatures=feature_values.shape[1],timesteps=timesteps,Delta=90)

    clusterer = cluster.SpectralClustering(n_clusters=numclusters, affinity='precomputed')
    print('Clustering')
    plt.matshow(S)

    clusterlabellist = clusterer.fit_predict(S)
    clusterlabellist = generate_data.reorder(clusterlabellist)
    clusterlabellist = clusteringheuristic1(clusterlabellist,int(w))
    clusterlabellist = clusteringheuristic2(clusterlabellist,int(w),S)
    print('Finding boundaries')
    boundaries=find_boundarieskmeans(clusterlabellist,numclusters,w/2)

    return boundaries

def noSFAFullBatch3(testdata,numclusters,w, numfeatures=5,reducenumber=20,timesteps=20):
    print('Dimensions of data are: '+str(testdata.shape))
    feature_values=testdata[:,:5]

    print('Constructing Simlarity Matrix')
    S=MakeSimilarityMatrix(feature_values,w,config='euc0s',numfeatures=numfeatures,timesteps=timesteps,Delta=30)
    plt.matshow(S)

    clusterer = cluster.SpectralClustering(n_clusters=numclusters, affinity='precomputed')
    print('Clustering')
    clusterlabellist = clusterer.fit_predict(S)
    clusterlabellist = generate_data.reorder(clusterlabellist)
    clusterlabellist = clusteringheuristic1(clusterlabellist,int(w))
    clusterlabellist = clusteringheuristic2(clusterlabellist,int(w),S)
    print('Finding boundaries')
    boundaries=find_boundarieskmeans(clusterlabellist,numclusters,w/2)

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

def MakeFeatureVectors(SFAvalues,numfeatures=5,timesteps=20):
    '''
    Creates featurevectors for use in matrix construction.
    :param SFAvalues: List of slow feature values.
    :param numfeatures: Number of features in one slow feature value
    :param timesteps: Number of time steps to consider
    :return:
    '''
    featurevectors=[]
    for t in range(len(SFAvalues)-timesteps):
        featurevectors.append([])
        for i in range(numfeatures):
            featurevectors[-1].append([])
            for j in range(timesteps):
                featurevectors[-1][-1].append(SFAvalues[t+j][i])
            featurevectors[-1][-1]=np.array(featurevectors[-1][-1])
    return featurevectors

def euclidsimilarity(x,y,Delta):
    '''
    Computes the piecewise taxi cab similarity between two featurevectors x and y. Using formula e^(d(x,y)/Delta^2) where $d(x,y)=Sum |x_i-y_i|     for components x_1 of feature vectors
    :return:
    '''
    sum=0
    for i in range(len(x)):
        sum+=np.linalg.norm(x[i]-y[i])
    return math.exp((-1*sum)/Delta**2)

def dtwsimilarity(x,y,Delta):
    '''
    Computes the piecewise taxi cab similarity between two featurevectors x and y. Using formula e^(d(x,y)/Delta^2) where $d(x,y)=Sum dtw(x_i,y_i)     for components x_1 of feature vectors
    :param x:
    :param y:
    :param Delta:
    :return:
    '''
    sum = 0
    for i in range(len(x)):
        sum += dtw.dtw(x[i],y[i],dist=lambda x,y:np.linalg.norm(x-y))[0]
    return math.exp((-1 * sum) / Delta ** 2)
def MakeSimilarityMatrix(SFAvalues,w,config='euc0s',numfeatures=5,timesteps=20,Delta=3.5):
    '''
    Creates a Similarity Matrix for use in clustering.
    :param SFAvalues: List of slow feature values, starting with slowest first, then next slowest etc.
    :param w:What part of the diagonal to treat special.
    :param config: Takes one of five values, 'hybrid','dtw','Euclidean','euc0s','dtw0s' for each of the five basic types of similarity matrix
    :param numfeatures: Number of features to consider.
    :param timesteps: Number of timesteps to consider at once
    :return:
    '''
    featurevectors=MakeFeatureVectors(SFAvalues,numfeatures=numfeatures,timesteps=timesteps)
    S=[]
    if config=='hybrid':
        def diag(x,y):
            return dtwsimilarity(x,y,Delta)
        def off(x,y):
            return euclidsimilarity(x,y,Delta)
    elif config=='dtw':
        def diag(x,y):
            return dtwsimilarity(x,y,Delta)
        def off(x,y):
            return dtwsimilarity(x,y,Delta)
    elif config=='euclid':
        def diag(x,y):
            return euclidsimilarity(x,y,Delta)
        def off(x,y):
            return euclidsimilarity(x,y,Delta)
    elif config=='dtw0s':
        def diag(x,y):
            return dtwsimilarity(x,y,Delta)
        def off(x,y):
            return 0
    elif config=='euc0s':
        def diag(x,y):
            return euclidsimilarity(x,y,Delta)
        def off(x,y):
            return 0
    else:
        print('Cannot recognize the required matrix format')
        def diag(x,y):
            return euclidsimilarity(x,y,Delta)
        def off(x,y):
            return euclidsimilarity(x,y,Delta)
    for i in range(len(featurevectors)):
        if i%500==0:
            print(str(i)+'/'+str(len(featurevectors)))
        S.append([])
        for j in range(len(featurevectors)):
            if j<i: #Value has already previously been computed
                S[i].append(S[j][i])
            else:
                if j>i+w: #It's far away from the diagonal
                    S[i].append(off(featurevectors[i],featurevectors[j]))
                else:
                    S[i].append(diag(featurevectors[i],featurevectors[j]))
    return np.matrix(S)

def minibatch(data,steplength,numfeatures=5,reducenumber=10,timesteps=20,Delta=3.5):
    currentboundary=0
    currentsteplength=steplength
    boundaries=[]
    while(currentboundary+currentsteplength*0.66<len(data)):
        print('Current step length is: '+str(currentsteplength))
        currentdata=data[currentboundary:currentboundary+currentsteplength]
        currentdata = pca_reduce(currentdata, reducenumber)
        sfanode = mdp.nodes.SFANode()
        expansionnode = mdp.nodes.PolynomialExpansionNode(2)
        testdata = expansionnode.execute(currentdata)
        sfanode.train(testdata)
        sfanode.stop_training()
        feature_values = sfanode.execute(testdata, n=numfeatures)
        S=MakeSimilarityMatrix(feature_values,100,config='euclid',numfeatures=numfeatures,timesteps=timesteps,Delta=Delta)
        clusterer = cluster.SpectralClustering(n_clusters=2, affinity='precomputed')
        clusterlabellist = clusterer.fit_predict(S)
        clusterlabellist = generate_data.reorder(clusterlabellist)
        boundary = generate_data.find_boundary(clusterlabellist)
        currentboundary+=boundary[0]
        print(boundary[0])
        if boundary[0]==0:
            print('Increasing window size by 20%')
            currentsteplength=int(1.2*currentsteplength)
        else:
            currentsteplength=steplength
        print('Found new boundary at '+str(currentboundary))
        boundaries.append(currentboundary)
    return boundaries

def noSFAminibatch(data,steplength,numfeatures=5,reducenumber=10,timesteps=20,Delta=3.5):
    currentboundary=0
    currentsteplength=steplength
    boundaries=[]
    while(currentboundary+currentsteplength*0.66<len(data)):
        print('Current step length is: '+str(currentsteplength))
        currentdata=data[currentboundary:currentboundary+currentsteplength]
        currentdata = pca_reduce(currentdata, 5)
        feature_values=currentdata
        S=MakeSimilarityMatrix(feature_values,100,config='euclid',numfeatures=numfeatures,timesteps=timesteps,Delta=Delta)
        clusterer = cluster.SpectralClustering(n_clusters=2, affinity='precomputed')
        clusterlabellist = clusterer.fit_predict(S)
        clusterlabellist = generate_data.reorder(clusterlabellist)
        boundary = generate_data.find_boundary(clusterlabellist)
        currentboundary+=boundary[0]
        print(boundary[0])
        if boundary[0]==0:
            print('Increasing window size by 20%')
            currentsteplength=int(1.2*currentsteplength)
        else:
            currentsteplength=steplength
        print('Found new boundary at '+str(currentboundary))
        boundaries.append(currentboundary)
    return boundaries

def noSFAminibatch3(data,steplength,numfeatures=5,reducenumber=10,timesteps=20,Delta=3.5):
    currentboundary=0
    currentsteplength=steplength
    boundaries=[]
    while(currentboundary+currentsteplength*0.66<len(data)):
        print('Current step length is: '+str(currentsteplength))
        currentdata=data[currentboundary:currentboundary+currentsteplength]
        feature_values=currentdata[:,:5]
        S=MakeSimilarityMatrix(feature_values,100,config='euclid',numfeatures=numfeatures,timesteps=timesteps,Delta=Delta)
        clusterer = cluster.SpectralClustering(n_clusters=2, affinity='precomputed')
        clusterlabellist = clusterer.fit_predict(S)
        clusterlabellist = generate_data.reorder(clusterlabellist)
        boundary = generate_data.find_boundary(clusterlabellist)
        currentboundary+=boundary[0]
        print(boundary[0])
        if boundary[0]==0:
            print('Increasing window size by 20%')
            currentsteplength=int(1.2*currentsteplength)
        else:
            currentsteplength=steplength
        print('Found new boundary at '+str(currentboundary))
        boundaries.append(currentboundary)
    return boundaries

def noSFAminibatch2(data,steplength,numfeatures=5,reducenumber=10,timesteps=20,Delta=10):
    currentboundary=0
    currentsteplength=steplength
    boundaries=[]
    while(currentboundary+currentsteplength*0.66<len(data)):
        print('Current step length is: '+str(currentsteplength))
        currentdata=data[currentboundary:currentboundary+currentsteplength]
        feature_values=currentdata
        S=MakeSimilarityMatrix(feature_values,100,config='euclid',numfeatures=feature_values.shape[1],timesteps=timesteps,Delta=Delta)
        clusterer = cluster.SpectralClustering(n_clusters=2, affinity='precomputed')
        clusterlabellist = clusterer.fit_predict(S)
        clusterlabellist = generate_data.reorder(clusterlabellist)
        boundary = generate_data.find_boundary(clusterlabellist)
        currentboundary+=boundary[0]
        print(boundary[0])
        if boundary[0]==0:
            print('Increasing window size by 20%')
            currentsteplength=int(1.2*currentsteplength)
        else:
            currentsteplength=steplength
        print('Found new boundary at '+str(currentboundary))
        boundaries.append(currentboundary)
    return boundaries
def minibatchold(data,steplength,numfeatures=5,reducenumber=10,timesteps=20,numclusters=2):
    currentmiddleboundary=0
    boundaries=[]
    currentsteplength=steplength
    while (currentmiddleboundary+currentsteplength*0.66<len(data)): #Aka if we still have a relatively full window at the end.
        currentdata=data[currentmiddleboundary:currentmiddleboundary+currentsteplength]
        currentdata=np.array([k for k in currentdata])
        boundary,binlist,featurevalues=GetClusteringFromData(currentdata,reducenumber=reducenumber,numfeatures=numfeatures,timesteps=timesteps,numclusters=numclusters)

        currentmiddleboundary+=boundary[0]
        if boundary[0]!=0:
            currentsteplength=steplength
        else:
            print("Didn't find a boundary, increasing search range")
            currentsteplength*=1.2
        print(currentmiddleboundary)
        print(boundary )
        boundaries.append((currentmiddleboundary,boundary[1]))
    return boundaries

def tripledouble(data,numfeatures,steplength,reducenumber,timesteps=20):
    startpoints=[]
    grace=0.25*steplength
    currentstartpoint=0
    oldhigherguess=-steplength
    while(currentstartpoint+steplength<len(data)):
        print('Starting from '+str(currentstartpoint))
        boundary1,binarylist1,featurevalues1=GetClusteringFromData(data[currentstartpoint:currentstartpoint+2*steplength],numclusters=2,numfeatures=numfeatures,reducenumber=reducenumber,timesteps=timesteps)
        boundary2, binarylist2, featurevalues2 = GetClusteringFromData(data[currentstartpoint:currentstartpoint+3 * steplength], numclusters=3, numfeatures=numfeatures, reducenumber=reducenumber, timesteps=timesteps)
        middleguess=boundary1[0]+currentstartpoint
        lowerguess=boundary2[0][0]+currentstartpoint
        higherguess=boundary2[0][1]+currentstartpoint
        print('2 Cluster guess: '+str(middleguess))
        print('3 Cluster guess 1: '+str(lowerguess))
        print('3 Cluster guess 2: '+str(higherguess))
        print('Old 3 cluster guess 2:'+str(oldhigherguess))
        distancelist=[(oldhigherguess,lowerguess),(middleguess,oldhigherguess),(middleguess,lowerguess),(higherguess,oldhigherguess),(middleguess,higherguess)]
        minimum=min(distancelist,key=lambda x:abs(x[0]-x[1]))
        print(minimum)
        if abs(minimum[0]-minimum[1])<grace and (minimum[1]+minimum[0])/2>currentstartpoint:
            currentstartpoint=(minimum[0]+minimum[1])/2
            answerkey='proper'
            print('worked well. Going with '+str(currentstartpoint))
        else:
            print('AYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY')
            currentstartpoint+=steplength
            answerkey='ayyyy'
        startpoints.append((currentstartpoint,answerkey))
        oldhigherguess=higherguess
    return startpoints
def pca_reduce(testdata,numdimensions):
    '''
    Takes testdata in the form needed for mdp nodes and then uses pca to reduce it down to numdimensions
    '''
    pcanode=mdp.nodes.PCANode(output_dim=numdimensions)
    pcanode.train(testdata)
    pcanode.stop_training()
    return pcanode.execute(testdata)

def boundaryevaluate(boundaries,truth):
    divergence=0
    missed=0
    found=0

    for boundary,range in truth:
        rangesize = range[1] - range[0]
        bestdistance=100000000000000000000000
        best=0
        for point in boundaries:
            if abs(boundary-point)<bestdistance:
                best=point
                bestdistance=abs(boundary-point)
        if best <range[0] or best >range[1]:
            if abs(range[0]-best)>4*rangesize and abs(range[1]-best)>4*rangesize:
                missed+=1
                print('MISSED ')

            else:
                divergence+=min(abs(range[0]-best),abs(range[1]-best))
                found+=1
        else:
            found+=1
    if found==0:
        print('='*50)
        print('REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
        return 10000, missed
    else:
        return divergence/float(found),missed

def boundaryevaluate2(boundaries,truth,rangesize=50):
    divergence=0
    missed=0
    found=0

    for boundary in truth:

        bestdistance=100000000000000000000000
        best=0
        for point in boundaries:
            if abs(boundary-point)<bestdistance:
                best=point
                bestdistance=abs(boundary-point)
        if best <boundary or best >boundary:
            if abs(boundary-best)>4*rangesize and abs(boundary-best)>4*rangesize:
                missed+=1
                print('MISSED ')
                print(boundary)
                print(best)
            else:
                divergence+=min(abs(boundary-best),abs(boundary-best))
                found+=1
        else:
            found+=1
    if found==0:
        print('='*50)
        print('REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
        return 10000, missed
    else:
        return divergence/float(found),missed

def showsegmentation(sevboundaries,defaultscale):
    '''
    Takes a list of boundaries with labels, and then makes  a graph showing them with boxes and shit.
    :param sevboundaries: a list of boundaries of the format (boundaries,label), where we assume that the boundaries include 0 and end at the end of data.
    :param defaultscale:
    :return:
    '''
    numsegmentations=len(sevboundaries)
    plt.figure()
    for i,(boundaries,label) in enumerate(sevboundaries):
        ax=plt.subplot(numsegmentations,1,i+1)
        for j in range(len(boundaries)-1): #We assume that boundaries start with 0, and end with the last point of the window
            if j%2==0:
                hatch='/'
                facecolor='red'
            else:
                hatch='\\'
                facecolor='blue'
            ax.add_patch(patches.Rectangle((boundaries[j]+1*defaultscale,0),(boundaries[j+1]-boundaries[j])-2*defaultscale,1*defaultscale,hatch=hatch,facecolor=facecolor))

            plt.title(label)
        ax.set_xlim(0,boundaries[-1])
        ax.set_ylim(0,defaultscale)
        ax.set_yticklabels([])
        ax.set_yticks([])

def comparetotruth(truth,severalsits):
    plt.figure()
    colors=['red','yellow','blue','orange']
    hatches=['-','+','x','o']
    ax=plt.subplot(1,1,1)
    truthlength=truth[0][-1][0]
    halfway=int(len(severalsits)/2)
    numthings=len(severalsits)
    ylabels=[]
    for i in range(halfway):
        boundaries,label=severalsits[i]
        for j in range(len(boundaries[:-1])):
            scalerino=truthlength/float(boundaries[-1])
            ax.add_patch(patches.Rectangle(((boundaries[j])*scalerino,i),(boundaries[j+1]-boundaries[j])*scalerino,1,facecolor=colors[j%4],hatch=hatches[j%4]))
        ylabels.append(label)

    truthranges,label=truth
    for i,(boundary,ranger) in enumerate(truthranges[:-1]):
        ax.add_patch(patches.Rectangle((truthranges[i][0],halfway),(truthranges[i+1][0]-boundary),1,facecolor='grey',hatch='/'))
        ax.add_patch(patches.Rectangle((ranger[0],halfway),ranger[1]-ranger[0],1,facecolor='green',hatch='*'))
    ylabels.append(label)
    for i in range(len(severalsits)-halfway):
        boundaries, label = severalsits[i+halfway]
        for j in range(len(boundaries[:-1])):
            scalerino = truthlength/float(boundaries[-1])
            ax.add_patch(patches.Rectangle(((boundaries[j]) * scalerino, i+1+halfway), (boundaries[j+1]-boundaries[j])*scalerino,1, facecolor=colors[j % 4], hatch=hatches[j % 4]))
        ylabels.append(label)
    ax.set_xlim(1,truthlength)
    ax.set_ylim(0,len(severalsits)+1)
    yticks=[x+0.5 for x in range (len(severalsits)+1)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
#
# asdf=([1,100,200,300,400],'label1')
# asdf2=([1,150,250,450,500],'label2')
# truth=([(1,(1,1)),(125,(100,150)),(225,(200,250)),(450,(450,450))],'Label of the truth')
# comparetotruth(truth,[asdf,asdf2])