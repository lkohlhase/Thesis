import generate_data
import numpy as np
import mdp
import matplotlib.pyplot as plt
import sklearn.cluster as cluster


#TODO try out shit on real data
#TODO graph variance versus quality of approximation

def first_test():
    '''
    First try at working on toydata. Uses unnormalized innerproduct for similarity matrix. Uses toydata1 as data
    '''
    testdata=generate_data.toydata1(noise=0.3,expansion=2)
    sfanode=mdp.nodes.SFANode()
    sfanode.train(testdata)
    sfanode.stop_training()
    testerino=sfanode.execute(testdata,n=3)
    show_features(testdata, testerino)
    #plt.show()

    #self similarity matrix stuff.
    timesteps=10
    numfeatures=2
    simmatrix=simmatrix1(numfeatures, testerino, timesteps)
    plt.matshow(simmatrix)
    plt.show()

def second_test():
    '''
    Similar to second_test(), used e^{-distance} for similarity matrix. Resulting similarity matrix is normalized.
    '''
    testdata=generate_data.toydata1(noise=1,expansion=2)
    sfanode=mdp.nodes.SFANode()
    sfanode.train(testdata)
    sfanode.stop_training()
    testerino=sfanode.execute(testdata,n=3)
    show_features(testdata,testerino)
    timesteps=15
    numfeatures=2
    distmatrix=simmatrix2(numfeatures,testerino,timesteps)
    plt.matshow(distmatrix,cmap='gray')
    plt.show()


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

def find_boundaries(binarylist):
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


def segmentate_first_try():
    testdata = generate_data.toydata1(noise=0.04, expansion=2)  # Sometimes gets bad results for higher than this amount of noise.
    blocksize=200
    numclusters=2

    timesteps=20
    numfeatures=3
    boundaries=[]
    boundary=0
    for i in range(2): #For now we know that there are 3 clusters, so we will use this knowledge.
        sfanode = mdp.nodes.SFANode()
        sfanode.train(testdata[boundary:boundary+blocksize])
        sfanode.stop_training()
        testerino = sfanode.execute(testdata[boundary:boundary+blocksize], n=3)
        distmatrix = simmatrix2(numfeatures, testerino, timesteps)
        plt.matshow(distmatrix)
        clusterer=cluster.SpectralClustering(n_clusters=2,affinity='precomputed')
        clustering=clusterer.fit_predict(distmatrix)

        newboundary=find_boundary(clustering)[0]
        boundary+=find_boundary(clustering)[0]
        boundaries.append(boundary)
    print(boundaries)




def third_test():
    '''
    We use spectral clustering here, to see if we closely identify clusters.
    '''
    clustersize=3
    # kappatest=[x[1] for x in generate_data.coswave(30, 1, noise=0.0001)]
    # kappatest2=[x[1] for x in generate_data.trianglewave(30,1,noise=0.0001)]
    # kappatest3=[kappatest[i]-kappatest2[i] for i in range(len(kappatest))]
    # print(kappatest3)
    testdata=generate_data.toydata1(noise=0.037,expansion=2) #Sometimes gets bad results for higher than this amount of noise.
    sfanode=mdp.nodes.SFANode()
    sfanode.train(testdata[:300])
    sfanode.stop_training()
    testerino=sfanode.execute(testdata[:300],n=3)
    timesteps=20
    numfeatures=3

    if clustersize==2:
        distmatrix=simmatrix2(numfeatures,testerino[:200],timesteps)
        clusterer=cluster.SpectralClustering(n_clusters=2,affinity='precomputed')
        asdf=clusterer.fit_predict(distmatrix)
        part1=asdf[:120]
        part2=asdf[120:200]
        part3=asdf
        print(part1)
        print(part1.tolist().count(0))
        print(part1.tolist().count(1))
        print(part1.tolist().count(2))
        print(part2)
        print(part2.tolist().count(0))
        print(part2.tolist().count(1))
        print(part2.tolist().count(2))
        print(part3)
        print(part3.tolist().count(0))
        print(part3.tolist().count(1))
        print(part3.tolist().count(2))
        show_features(testdata[:200],testerino[:200])
        plt.matshow(distmatrix,cmap='gray')
        plt.show()
        print('boundary Value:' + str(find_boundary(asdf[:200])))
    else:
        distmatrix = simmatrix2(numfeatures, testerino, timesteps)
        clusterer = cluster.SpectralClustering(n_clusters=3, affinity='precomputed')
        asdf = clusterer.fit_predict(distmatrix)
        part1 = asdf[:120]
        part2 = asdf[120:200]
        part3 = asdf[200:]
        print(part1)
        print(part1.tolist().count(0))
        print(part1.tolist().count(1))
        print(part1.tolist().count(2))
        print(part2)
        print(part2.tolist().count(0))
        print(part2.tolist().count(1))
        print(part2.tolist().count(2))
        print(part3)
        print(part3.tolist().count(0))
        print(part3.tolist().count(1))
        print(part3.tolist().count(2))

        #show_features(testdata, testerino)
        plt.matshow(distmatrix, cmap='gray')
        plt.show()
        asdf=reorder(asdf)
        print(find_boundaries(asdf))


def twoversusthree():
    noises=[0.001,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.11,0.12,0.13,0.14,0.15]
    timesteps = 20
    numfeatures = 3
    midpoints2=[]
    midpoints3=[]
    for noise in noises:
        midpoints2.append([])
        midpoints3.append([])
        for i in range(5):
            print(noise)
            testdata=generate_data.toydata1(noise,expansion=2)
            #================================================== 2 cluster stuff
            sfanode2 = mdp.nodes.SFANode()
            sfanode2.train(testdata[:200])
            sfanode2.stop_training()
            testerino2 = sfanode2.execute(testdata[:200], n=3)
            distmatrix2=simmatrix2(numfeatures,testerino2,timesteps)
            clusterer2=cluster.SpectralClustering(n_clusters=2,affinity='precomputed')
            binarylist2=clusterer2.fit_predict(distmatrix2)
            binarylist2=reorder(binarylist2)
            midpoints2[-1].append(find_boundary(binarylist2))
            #================================================== 3 cluster stuff
            sfanode3 = mdp.nodes.SFANode()
            sfanode3.train(testdata[:300])
            sfanode3.stop_training()
            testerino3 = sfanode3.execute(testdata[:300], n=3)
            distmatrix3=simmatrix2(numfeatures,testerino3,timesteps)
            clusterer3 = cluster.SpectralClustering(n_clusters=3, affinity='precomputed')
            binarylist3=clusterer3.fit_predict(distmatrix3)
            binarylist3=reorder(binarylist3)
            midpoints3[-1].append(find_boundaries(binarylist3))
    midpoints2avg=[np.mean([a[1] for a in k]) for k in midpoints2]
    midpoints3avg=[np.mean([a[1] for a in k]) for k in midpoints3]
    midpoints2std=[np.std([a[1] for a in k]) for k in midpoints2]
    midpoints3std=[np.std([a[1] for a in k]) for k in midpoints3]
    plt.subplot(311)
    plt.title('Errors of the best boundary. Average of 5 runs')
    plt.errorbar((noises+noises),(midpoints2avg+midpoints3avg), yerr=(midpoints2std+midpoints3std),fmt='o')
    plt.subplot(312)
    plt.title('Average boundary point for splitting into 2. Should be 110 ish')
    boundarypoints2avg=[np.mean([a[0] for a in k]) for k in midpoints2]
    boundarypoints3avg1=[np.mean([a[0][0] for a in k]) for k in midpoints3]
    boundarypoints3avg2=[np.mean([a[0][1] for a in k]) for k in midpoints3]
    boundarypoints2std=[np.std([a[0] for a in k]) for k in midpoints2]
    boundarypoints3std1=[np.std([a[0][0] for a in k]) for k in midpoints3]
    boundarypoints3std2=[np.std([a[0][1] for a in k]) for k in midpoints3]
    plt.errorbar((noises),(boundarypoints2avg), yerr=(boundarypoints2std),fmt='o')
    plt.subplot(313)
    plt.title('Average Boundary poijnts for splitting into 3 parts. Should be 110ish and 190ish respectively')
    plt.errorbar((noises+noises),(boundarypoints3avg1+boundarypoints3avg2), yerr=(boundarypoints3std1+boundarypoints3std2),fmt='o')
    plt.show()
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


def show_features(testdata, testerino):
    '''
    Make a plot of testdata versus slow features. Shows first 3 dimensions of testdata and first 3 slow features
    :param testdata: Datapoints
    :param testerino: Calculated slow features
    :return:
    '''
    plt.subplot(231)
    plt.plot(range(len(testdata)), [i[0] for i in testdata])
    plt.subplot(232)
    plt.plot(range(len(testdata)), [i[1] for i in testdata])
    plt.subplot(233)
    plt.plot(range(len(testdata)), [i[2] for i in testdata])
    plt.subplot(234)
    plt.plot(range(len(testerino)), [i[0] for i in testerino])
    plt.subplot(235)
    plt.plot(range(len(testerino)), [i[1] for i in testerino])
    plt.subplot(236)
    plt.plot(range(len(testerino)), [i[2] for i in testerino])
