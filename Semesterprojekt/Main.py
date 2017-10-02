import re
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

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


# class CCFIPCA:
#     def __init__(self,data,l):
#         self.max=len(data[0]) #The dimensionality of the vector space.
#         self.data=data
#         self.u=data
#         self.ui=[] #ui[n][i] = u_i(n)
#         self.vi=[] #vi[n][i] = v_i[n]
#         self.l=l
#         length=len(data)
#         for n in range(length):#Going through all the datapoints
#             self.vi.append([])
#             self.ui.append([])
#             self.ui[n].append(self.u[n])#u_1(n)=u(n)
#             for i in range(min(self.max,n+1)): #Has to be n+1, since range(x) returns numbers 0 to x-1
#                 if i==n:
#                     self.vi[n].append(self.u[n])
#                 else:
#                     vileftterm=(n+1-1.-l+0.5)/(n+1.)*self.vi[n-1][i] #cheated in a +0.5 term
#                     virightterm=(0.5+l)/(n+1.)*self.ui[n][i]*(self.ui[n][i].transpose()*self.vi[n-1][i]/np.linalg.norm(self.vi[n-1][i]))
#                     newvi=vileftterm+virightterm
#                     self.vi[n].append(newvi)
#                     newui=self.ui[n][i]-(self.ui[n][i].transpose()*self.vi[n][i]/np.linalg.norm(self.vi[n][i])).flat[0]*self.vi[n][i]/np.linalg.norm(self.vi[n][i])
#                     self.ui[n].append(newui)
#         self.v=[x for x in self.vi[-1][:]]
#     def update(self,datapoint):
#         n = len(self.vi)
#         self.u.append(datapoint)
#         self.vi.append([])
#         self.ui.append([])
#         self.ui[n].append(self.u[n])  # u_1(n)=u(n)
#         for i in range(self.max):
#             vileftterm = (n + 1 - 1. - self.l + 0.5) / (n + 1.) * self.vi[n - 1][i]  # cheated in a +0.5 term
#             virightterm = (0.5 + self.l) / (n + 1.) * self.ui[n][i] * (
#             self.ui[n][i].transpose() * self.vi[n - 1][i] / np.linalg.norm(self.vi[n - 1][i]))
#             newvi = vileftterm + virightterm
#             self.vi[n].append(newvi)
#             newui = self.ui[n][i] - (self.ui[n][i].transpose() * self.vi[n][i] / np.linalg.norm(self.vi[n][i])).flat[
#                                         0] * self.vi[n][i] / np.linalg.norm(self.vi[n][i])
#             self.ui[n].append(newui)
#
#
# class MCA:
#     def make_Ci(self,t,i):
#         Cileft = self.Ci[t][i-1]
#         testerino = (self.wi[t][i-1].transpose() * self.wi[t][i-1]).flat[0]
#         print(testerino)
#         Ciright = self.gamma * (self.wi[t][i-1] * self.wi[t][i-1].transpose()) / (self.wi[t][i-1].transpose() * self.wi[t][i-1]).flat[0]
#         self.Ci[t].append(Cileft + Ciright)
#
#     def __init__(self,data,nu,lambda1):
#         epsilon=0.05 #That's what we're going with for now.
#         self.wi=[] #wi[t][i] =w_i(t)
#         self.Ci=[] #Ci[t][i] =C_i(t)
#         self.lambda1=lambda1
#         self.z=data
#         length=10
#         self.max=1
#         self.gamma=self.lambda1+epsilon
#         for t in range(length):
#             print("t="+str(t))
#             self.Ci.append([])
#             self.wi.append([])
#             for i in range(min(t+1,self.max)):
#                 print("i="+str(i))
#                 if i==t:
#                     self.wi[t].append(self.z[t])
#                     if i==0:
#                         self.Ci[t].append(self.z[t]*self.z[t].transpose())
#                     else:
#                         self.make_Ci(t,i)
#                 else:
#                     #Update C_i(t)
#                     if i==0:
#                         self.Ci[t].append(self.z[t]*self.z[t].transpose())
#                     else:
#                         self.make_Ci(t,i)
#                     #Updating w_i(t)
#                     termleft=1.5*self.wi[t-1][i]
#                     termmiddle=nu*self.Ci[t][i]*self.wi[t-1][i]
#                     termright=nu*(self.wi[t-1][i].transpose()*self.wi[t-1][i]).flat[0]*self.wi[t-1][i]
#                     termfinal=termleft-termmiddle-termright
#                     self.wi[t].append(termfinal)
#             if t+1<self.max:
#                 self.Ci[t].append(np.eye(self.max))
#                 help=np.matrix([[float(x==(t+1))*0.01] for x in range(self.max)])
#


