import generate_data
import sklearn.cluster as cluster
import mdp
import numpy as np
import time
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

def minibatch(data,steplength,numfeatures=3,reducenumber=10,timesteps=20,numclusters=2):
    currentmiddleboundary=0
    boundaries=[]
    currentsteplength=steplength
    while (currentmiddleboundary+currentsteplength<len(data)):
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
