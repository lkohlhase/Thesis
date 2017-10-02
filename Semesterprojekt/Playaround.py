from incsfa import *
from Main import *
import matplotlib.pyplot as plt
import mdp

def do_1():
    readindata=readSample(1)

    expnode=mdp.nodes.PolynomialExpansionNode(degree=2)
    expanded=expnode(np.array(readindata))
    center=sum(expanded)/len(expanded)
    centereddata=[np.matrix(x-center) for x in expanded] #We center data after expansion

    incsfanode = IncSFANode(2015, 20, 20, eps=0.005)
    for x in centereddata:
        incsfanode.update(x)

    feature1=[]
    feature2=[]
    feature3=[]
    for x in centereddata:
        feature1.append((incsfanode.v[0].transpose()*np.matrix(x).transpose()).flat[0])
        feature2.append((incsfanode.v[1].transpose()*np.matrix(x).transpose()).flat[0])
        feature3.append((incsfanode.v[2].transpose() * np.matrix(x).transpose()).flat[0])
    fig=plt.figure()
    plt.plot(range(len(centereddata)),feature1,'red',label="feature 1")
    plt.plot(range(len(centereddata)),feature2,'blue',label="feature 2")
    plt.plot(range(len(centereddata)),feature3,'green',label="feature 3")
    plt.legend(loc=2)

    plt.text(x='200',y='15',s='walk')
    plt.text(x='600',y='15',s='jump')
    plt.text(x='1200',y='15', s='walk')
    plt.text(x='2000',y='15',s='punch')
    plt.text(x='2550',y='15',s='walk')
    plt.text(x='3350',y='15',s='kick')
    plt.text(x='4100',y='15',s='punch')


    #============================================ Transition points for data sample 1
    plt.axvline(x=550)
    plt.axvline(x=1150)
    plt.axvline(x=1980)
    plt.axvline(x=2500)
    plt.axvline(x=3200)
    plt.axvline(x=4065)
    plt.axvline(x=4579)
    plt.show()

def count0(list):
    counter=0
    sign=1
    positions=[]
    if list[0]>0:
        sign=1
    else:
        sign=-1
    for i,x in enumerate(list):
        if sign*x<0:
            counter=counter+1
            positions.append(i)
            sign=sign*-1
    return (counter,positions)


def do_11():
    readindata = readSample(11)

    expnode = mdp.nodes.PolynomialExpansionNode(degree=2)
    expanded = expnode(np.array(readindata))
    center = sum(expanded) / len(expanded)
    centereddata = [np.matrix(x) for x in expanded]  # We center data after expansion

    incsfanode = IncSFANode(2015, 20, 20, eps=0.05)
    for x in centereddata:
        incsfanode.update(x)

    feature1 = []
    feature2 = []
    feature3 = []
    for x in centereddata:
        feature1.append((incsfanode.v[0].transpose() * np.matrix(x).transpose()).flat[0])
        feature2.append((incsfanode.v[1].transpose() * np.matrix(x).transpose()).flat[0])
        feature3.append((incsfanode.v[2].transpose() * np.matrix(x).transpose()).flat[0])
    fig = plt.figure()
    plt.plot(range(len(centereddata)), feature1, 'red', label="feature 1")
    plt.plot(range(len(centereddata)), feature2, 'blue', label="feature 2")
    plt.plot(range(len(centereddata)), feature3, 'green', label="feature 3")
    plt.legend(loc=2)
    # ===============================================Transition points for data sample 11
    plt.axvline(x=1115)
    plt.axvline(x=1700)
    plt.axvline(x=2350)
    plt.axvline(x=2755)
    plt.axvline(x=3340)
    plt.axvline(x=4030)
    plt.axvline(x=4670)
    plt.axvline(x=5674)
    plt.text(x='500',y='15',s='walk')
    plt.text(x='1150',y='15',s='both arms rotation')
    plt.text(x='1800',y='15',s='right arm rotation')
    plt.text(x='2400 ',y='16',s='both arms')
    plt.text(x='2800',y='15',s='left arm rotation')
    plt.text(x='3400',y='15',s='right arm rotation')
    plt.text(x='4100',y='15',s='both arms rotation')
    plt.text(x='4800',y='15',s='walk')
    plt.show()


def do_1_incremental():
    readindata=readSample(1)

    expnode=mdp.nodes.PolynomialExpansionNode(degree=2)
    expanded=expnode(np.array(readindata))
    center=sum(expanded)/len(expanded)
    centereddata=[np.matrix(x-center) for x in expanded] #We center data after expansion


    feature1=[]
    feature2=[]
    feature3=[]
    timereset=800
    incsfanode = IncSFANode(2015, 20, 20, eps=0.005)
    for x in centereddata[:timereset]:
        incsfanode.update(x)
    for x in centereddata[(timereset+2):]:
        incsfanode.update(x)
        feature1.append((incsfanode.v[0].transpose()*np.matrix(x).transpose()).flat[0])
        feature2.append((incsfanode.v[1].transpose()*np.matrix(x).transpose()).flat[0])
        feature3.append((incsfanode.v[2].transpose() * np.matrix(x).transpose()).flat[0])

    fig=plt.figure()
    plt.plot(range(len(centereddata[(timereset+2):])),feature1,'red',label="feature 1")
    plt.plot(range(len(centereddata[(timereset+2):])),feature2,'blue',label="feature 2")
    plt.plot(range(len(centereddata[(timereset+2):])),feature3,'green',label="feature 3")
    plt.legend(loc=2)

    plt.text(x=(200-timereset),y='15',s='walk')
    plt.text(x=(600-timereset),y='15',s='jump')
    plt.text(x=(1200-timereset),y='15', s='walk')
    plt.text(x=(2000-timereset),y='15',s='punch')
    plt.text(x=(2550-timereset),y='15',s='walk')
    plt.text(x=(3350-timereset),y='15',s='kick')
    plt.text(x=(4100-timereset),y='15',s='punch')


    #============================================ Transition points for data sample 1
    plt.axvline(x=(550-timereset))
    plt.axvline(x=(1150-timereset))
    plt.axvline(x=(1980-timereset))
    plt.axvline(x=(2500-timereset))
    plt.axvline(x=(3200-timereset))
    plt.axvline(x=(4065-timereset))
    plt.axvline(x=(4579-timereset))
    plt.show()

do_1_incremental()