import os.path
import matplotlib.pyplot as plt
import numpy as np
f,axa = plt.subplots(1, sharex=True)
precision=[]
recall=[]
tim_=[]
reg=[]
Lin=[]
filename=open('0.02_cbj+cbij_09_25_22_48_nopca'+'.cluster','r')
i=0
# i%30 userID
print(filename)
#filename.readline()
for line in filename:
	TP=0
	cluster=line.strip().split('\t')
	for k in range(len(cluster)-1):
		if k>0:
			if int(cluster[k])/5 == (i%30)/5:
				TP=TP+1
	precision.append(float(TP)/float(len(cluster)-2))		
	recall.append(float(TP)/5)
	print(precision[-1],recall[-1])	
	tim_.append(i)	
	i=i+1
	
sum_p={}
sum_r={}
up={}
ur={}
for i in range(200):
	sum_p[i]=[]
	sum_r[i]=[]
for i in range(30):
	up[i]=[]
	ur[i]=[]
for k in range(len(tim_)):
	sum_p[tim_[k]/30].append(precision[k])
	sum_r[tim_[k]/30].append(recall[k])
	up[tim_[k]%30].append(precision[k])
	ur[tim_[k]%30].append(recall[k])
	
p=[]
r=[]
for k in sum_p:
	p.append(np.array(sum_p[k]).mean())
	r.append(np.array(sum_r[k]).mean())
	
#print(up[0][199],up[4][199])
#for k in range(5):
	#axa.plot(range(200),up[k],label = str(k)+'p')
	
	#axa.plot(range(200),ur[k],label = str(k)+'r')
axa.plot(range(200),p,label = "precision")
axa.plot(range(200),r,label = 'recall')
axa.legend(loc='upper left',prop={'size':9})
axa.set_xlabel("Iteration")
axa.set_ylabel("Regret")
axa.set_title("CAB")
#plt.axis([0,10000,0,500])
plt.show()	