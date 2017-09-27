import numpy as np
from LinUCB import *
import math
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import datetime
import os.path
from conf import sim_files_folder, save_address
from sklearn import linear_model
import matplotlib.pyplot as plt  
#import statsmodels.api as sm
#from statsmodels.formula.api import ols
#from statsmodels.sandbox.regression.predstd import wls_prediction_std

class CABUserStruct(LinUCBUserStruct):
	def __init__(self,featureDimension,  lambda_, userID):
		LinUCBUserStruct.__init__(self,featureDimension = featureDimension, lambda_= lambda_)
		self.reward = 0
		self.I = lambda_*np.identity(n = featureDimension)	
		#self.betaA=lambda_*np.identity(n = featureDimension)
		self.counter = 0
		self.CBPrime = 0
		self.CoTheta= np.zeros(featureDimension)
		self.d = featureDimension
		self.ID = userID
		self.betaA={}
		self.betab={}
		self.hisReward=[]
		self.hisArticle=[]
		self.hisjReward={}
		self.betaj={}
		self.cbbetaj={}
		self.linearmodel={}
		for i in range(30):
			self.hisjReward[i]=[]
			self.betaj[i]=np.zeros(featureDimension)
			self.cbbetaj[i]=0
			self.betaA[i]=lambda_*np.identity(n = featureDimension)
			self.betab[i]=np.zeros(self.d)
			self.linearmodel[i]=linearmodelj(featureDimension,lambda_,i)
		
		
	def updateParameters(self, articlePicked_FeatureVector, click):
		self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
		self.b +=  articlePicked_FeatureVector*click
		self.AInv = np.linalg.inv(self.A)
		self.CoTheta = np.dot(self.AInv, self.b)
		self.counter+=1
		#print(self.CoTheta)
	def getCBP(self, alpha, article_FeatureVector,time):
		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
		pta = alpha * var*np.sqrt(math.log10(time+1))
		return pta
	def getBeta(self,alpha,time):
		
		
		for j in self.hisjReward:
			if j!=self.ID:
				diff=[]
				#print(self.hisjReward[j])
				for k in range(len(self.hisjReward[j])):
					diff.append(self.hisjReward[j][k]-self.hisReward[k])
				
				self.linearmodel[j].updateParameters(self.hisArticle[-1],diff[-1])
				self.betaj[j]=self.linearmodel[j].getTheta()
		

		

class linearmodelj():
	def __init__(self, featureDimension, lambda_, userID):
		self.d = featureDimension
		self.A = lambda_*np.identity(n = self.d)
		self.b = np.zeros(self.d)
		self.AInv = np.linalg.inv(self.A)
		
		self.UserTheta = np.zeros(self.d)
		

	def updateParameters(self, articlePicked_FeatureVector, click):
		self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
		self.b += articlePicked_FeatureVector*click
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = np.dot(self.AInv, self.b)
	
	
	def getA(self):
		return self.A
	def getTheta(self):
		#print(self.UserTheta)
		return self.UserTheta

	def getCB(self, alpha, article_FeatureVector,time):
		
		
		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
		#pta = mean + alpha * var
		pta=alpha * var*np.sqrt(math.log10(time+1))
		return pta


class CABAlgorithm():
	def __init__(self,dimension,alpha,lambda_,n,alpha_2,option):
		self.time = 0
		#N_LinUCBAlgorithm.__init__(dimension = dimension, alpha=alpha,lambda_ = lambda_,n=n)
		self.users = []
		#self.gamma=gamma
		#algorithm have n users, each user has a user structure
		self.userNum=n
		for i in range(n):
			self.users.append(CABUserStruct(dimension,lambda_, i)) 
			
		self.dimension = dimension
		self.alpha = alpha
		self.CanEstimateCoUserPreference = True
		self.CanEstimateUserPreference = False
		self.CanEstimateW = False
		self.CanEstimateV = False
		self.cluster=[]
		self.UCB1=[]
		self.UCB2=[]
		self.LCB1=[]
		self.LCB2=[]
		self.a=0
		self.startTime = datetime.datetime.now()
		self.option=option
		timeRun = self.startTime.strftime('_%m_%d_%H_%M') 
		self.filenameWritePara1=os.path.join(save_address, str(self.alpha)+'_'+'cbj+cbij'+timeRun+'.cluster')
		self.filenameWritePara2=os.path.join(save_address, str(self.alpha)+'_'+'CBJ'+timeRun+'.cluster')
		self.filenameWritePara3=os.path.join(save_address, str(self.alpha)+'_'+'Cbij'+timeRun+'.cluster')
		self.filenameWritePara4=os.path.join(save_address, str(self.alpha)+'_'+'reject'+timeRun+'.cluster')
		
		
	def decide(self,pool_articles,userID,thetaStar):
		#print(self.users[userID].A)
		maxPTA = float('-inf')
		articlePicked = None
		WI=self.users[userID].CoTheta
		self.LCB1=[]
		self.LCB2=[]
		self.UCB1=[]
		self.UCB2=[]
		for k in pool_articles:
			clusterItem=[]
			featureVector = k.contextFeatureVector[:self.dimension]
			#rwd1=np.dot(ThetaStar,featureVector)
			CBI=self.users[userID].getCBP(self.alpha,featureVector,self.time)
			temp=np.zeros((self.dimension,))
			WJTotal=np.zeros((self.dimension,))
			CBJTotal=0.0
			ri=np.dot(WI,featureVector)
			
			for j in range(len(self.users)):
				WJ=self.users[j].CoTheta
				CBJ=self.users[j].getCBP(self.alpha,featureVector,self.time)
				rj=np.dot(WJ,featureVector)
				#print(self.users[userID].hisReward)
				
				
				if(j!=userID):
					if(self.users[userID].counter!=0):
					
						diffR=np.dot(self.users[userID].betaj[j],k.contextFeatureVector[:self.dimension])
						diffR_true=np.dot(thetaStar,k.contextFeatureVector[:self.dimension])-np.dot(self.users[j].CoTheta,k.contextFeatureVector[:self.dimension])
						compare= np.dot(WI,featureVector)-np.dot(WJ,featureVector)
						cbij=self.users[userID].linearmodel[j].getCB(self.alpha,featureVector,self.time)
						#print(abs(diffR),cbij,CBJ,CBI)
						self.UCB1.append(abs(diffR)+cbij-abs(diffR_true))
						self.LCB1.append(abs(diffR)-cbij-abs(diffR_true))
						self.UCB2.append(abs(compare)+CBI-abs(diffR_true))
						self.LCB2.append(abs(compare)-CBI-abs(diffR_true))
						if(self.option==1):
							if abs(diffR)<=CBJ+cbij:
						
								clusterItem.append(self.users[j])
								WJTotal+=WJ
								CBJTotal+=CBJ
						if(self.option==2):
							if abs(diffR)<=CBJ:
						
								clusterItem.append(self.users[j])
								WJTotal+=WJ
								CBJTotal+=CBJ
						if(self.option==3):
							if abs(diffR)<=cbij:
						
								clusterItem.append(self.users[j])
								WJTotal+=WJ
								CBJTotal+=CBJ
						if(self.option==4):
							if ((rj-(abs(diffR)-cbij))<=ri+CBI) & ((rj+(abs(diffR)+cbij))>=ri-CBI):
								clusterItem.append(self.users[j])
								WJTotal+=WJ
								CBJTotal+=CBJ
					
				else: 
				
					clusterItem.append(self.users[userID])
					#print('trues')
					WJTotal+=WI
					CBJTotal+=CBI
			
			CW= WJTotal/len(clusterItem)
			CB= CBJTotal/len(clusterItem)
			
			#CW=WI
			#CB=CBI
			x_pta = np.dot(CW,featureVector)+CB
			# pick article with highest Prob
			
			if maxPTA < x_pta:
				articlePicked = k.id
				featureVectorPicked = k.contextFeatureVector[:self.dimension]
				picked = k
				maxPTA = x_pta
				self.cluster=clusterItem
		#print(len(self.cluster))
		return picked
		
	

		
		
	
	def updateParameters(self, articlePicked, click,userID,gamma,rwdu):
		
		#self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)
		self.users[userID].hisReward.append(click)
		self.users[userID].hisArticle.append(articlePicked.contextFeatureVector[:self.dimension])
		for j in range(self.userNum):
			if j!=userID:
				
				rj=np.dot(self.users[j].CoTheta,articlePicked.contextFeatureVector[:self.dimension])
				#rj=rwdu[j] % this is for 
				self.users[userID].hisjReward[j].append(rj)
				
				
		self.users[userID].getBeta(self.alpha,self.time)
		clusterNow=[]
		if(self.users[userID].getCBP(self.alpha,articlePicked.contextFeatureVector[:self.dimension],self.time)>=-1):
			self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)
		if(self.option==1):
			self.filenameWriteParatemp=self.filenameWritePara1
		if(self.option==2):
			self.filenameWriteParatemp=self.filenameWritePara2
		if(self.option==3):
			self.filenameWriteParatemp=self.filenameWritePara3
		if(self.option==4):
			self.filenameWriteParatemp=self.filenameWritePara4
       
		with open(self.filenameWriteParatemp, 'a+') as f:
			if(self.cluster!=[]):
				f.write(str(self.time/self.userNum))
				for i in range(len(self.cluster)):
					f.write('\t'+str(self.cluster[i].ID))
					#print(self.filenameWritePara)
				f.write('\t'+str(articlePicked.id))
				f.write('\n')
		
		self.a=0
		self.time +=1
		
					
	def getbeta(self,userID):
		betaList={}
		thetajList={}
		for i in range(self.userNum):
			if (i!=userID):
				betaList[i]=self.users[userID].betaj[i]
				thetajList[i]=self.users[i].CoTheta
		return [betaList,thetajList]	
					
	def getCoTheta(self, userID):
		return self.users[userID].CoTheta
	def getUCBLCB(self):
		ucb1=np.array(self.UCB1).mean()
		ucb2=np.array(self.UCB2).mean()
		lcb1=np.array(self.LCB1).mean()
		lcb2=np.array(self.LCB2).mean()
		cbinfo= [ucb1,ucb2,lcb1,lcb2]
		return cbinfo

	



