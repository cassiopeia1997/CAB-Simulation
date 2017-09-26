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
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std

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
					diff.append(abs(self.hisjReward[j][k]-self.hisReward[k]))
				'''
				CB=self.linearmodel[j].getCB(alpha,self.hisArticle[-1],time)
				
				self.linearmodel[j].updateParameters(self.hisArticle[-1],diff[-1])
				self.betaj[j]=self.linearmodel[j].getTheta()
				'''
				#print(self.betaj[j])
				
				'''
				self.betaA[j] += np.outer(self.hisArticle[-1],self.hisArticle[-1])
				betaAinv=np.linalg.inv(self.betaA[j])
				betavar=np.sqrt(np.dot(np.dot(self.hisArticle[-1], betaAinv),  self.hisArticle[-1]))
				CB=alpha * betavar * np.sqrt(math.log10(time+1))
				'''
				#regr=linear_model.LinearRegression(fit_intercept=False)
				#regr.fit(np.array(self.hisArticle),np.array(diff))
				
				#print('1',regr.coef_)
				
				linear_model1=sm.OLS(np.array(diff),np.array(self.hisArticle))
				results=linear_model1.fit()
				self.betaj[j]=results.params
				'''
				#wls_prediction_std(results)
				#_,condifence_interval_lower,confidence_interval_upper=wls_prediction_std(results)
				#print(CB,confidence_interval_upper)
				#for i in range(len(self.hisReward)):
					
				#print('rwd',diff)
				#xx=[]
				#for i in range(len(diff)):
				#	xx.append(confidence_interval_upper[i]-diff[i])
				
				self.cbbetaj[j]=CB
				#print(math.isnan(self.cbbetaj[j]))
				
				'''

		#a, b = regr.coef_, regr.intercept_
		#plt.show()

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
		
		mean = np.dot(self.UserTheta,  article_FeatureVector)
		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv),  article_FeatureVector))
		#pta = mean + alpha * var
		pta=mean+alpha * var*np.sqrt(math.log10(time+1))
		return pta


class CABidealAlgorithmincrease():
	def __init__(self,dimension,alpha,lambda_,n,alpha_2):
		self.time = 0
		#N_LinUCBAlgorithm.__init__(dimension = dimension, alpha=alpha,lambda_ = lambda_,n=n)
		self.users = []
		#self.gamma=gamma
		#algorithm have n users, each user has a user structure
		self.userNum=n
		self.selectedGroup={}
		for i in range(n):
			self.users.append(CABUserStruct(dimension,lambda_, i)) 
			self.selectedGroup[i]=[]
		self.dimension = dimension
		self.alpha = alpha
		self.CanEstimateCoUserPreference = True
		self.CanEstimateUserPreference = False
		self.CanEstimateW = False
		self.CanEstimateV = False
		self.cluster=[]
		self.a=0
		self.startTime = datetime.datetime.now()
		
		timeRun = self.startTime.strftime('_%m_%d_%H_%M') 
		self.filenameWritePara=os.path.join(save_address, str(self.alpha)+'_'+'CAB_new'+timeRun+'.cluster')
	def decide(self,pool_articles,userID):
		#print(self.users[userID].A)
		maxPTA = float('-inf')
		articlePicked = None
		WI=self.users[userID].CoTheta
		for k in pool_articles:
			clusterItem=[]
			featureVector = k.contextFeatureVector[:self.dimension]
			#rwd1=np.dot(ThetaStar,featureVector)
			CBI=self.users[userID].getCBP(self.alpha,featureVector,self.time)
			temp=np.zeros((self.dimension,))
			WJTotal=np.zeros((self.dimension,))
			CBJTotal=0.0
			t=1
			t2=1
			t3=1
			for j in range(len(self.users)):
			
				WJ=self.users[j].CoTheta
				CBJ=self.users[j].getCBP(self.alpha,featureVector,self.time)
				if(j!=userID):
					if self.users[userID].counter!=0:
						if(self.time/30>40):
							if(j/5==userID/5):
							
								clusterItem.append(self.users[j])
								WJTotal+=WJ
								CBJTotal+=CBJ
									
						if(30<self.time/30<=40):
							if(j/5==userID/5):
								if(t<=3):
									clusterItem.append(self.users[j])
									WJTotal+=WJ
									CBJTotal+=CBJ
									t=t+1
								
						if(20<self.time/30<=30):
							if(t2<=2):
								if(j/5==userID/5):
									clusterItem.append(self.users[j])
									WJTotal+=WJ
									CBJTotal+=CBJ
									t2=t2+1		
						
						if(10<self.time/30<=20):
							if(t3<=1):
								if(j/5==userID/5):
									clusterItem.append(self.users[j])
									WJTotal+=WJ
									CBJTotal+=CBJ
									t3=t3+1
						'''	
						if(self.time/30>150):
							
									
								clusterItem.append(self.users[j])
								WJTotal+=WJ
								CBJTotal+=CBJ
						'''
					
				else:
					WJTotal+=WI
					CBJTotal+=CBJ
					clusterItem.append(self.users[userID])
				
			
			
			
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
		
	

		
		
	
	def updateParameters(self, articlePicked, click,userID,gamma):
		self.selectedGroup[userID]=[]
		#self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)
		self.users[userID].hisReward.append(click)
		self.users[userID].hisArticle.append(articlePicked.contextFeatureVector[:self.dimension])
		for j in range(self.userNum):
			if j!=userID:
				#print(j,self.users[j].CoTheta)
				rj=np.dot(self.users[j].CoTheta,articlePicked.contextFeatureVector[:self.dimension])
				self.users[userID].hisjReward[j].append(rj)
				#print(self.users[userID].hisjReward[j])
				if abs(rj-click)<=self.users[j].getCBP(self.alpha,articlePicked.contextFeatureVector[:self.dimension],self.time):
					self.selectedGroup[userID].append(j)
		#print(userID,self.selectedGroup[userID])
		#print(self.selectedGroup)
		self.users[userID].getBeta(self.alpha,self.time)
		clusterNow=[]
		if(self.users[userID].getCBP(self.alpha,articlePicked.contextFeatureVector[:self.dimension],self.time)>=-1):
			self.users[userID].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)
			
       
		
		
		else:
			
		
			for i in range(len(self.cluster)):
				
				if(self.cluster[i].getCBP(self.alpha,articlePicked.contextFeatureVector[:self.dimension],self.time)<gamma/4):
					self.cluster[i].updateParameters(articlePicked.contextFeatureVector[:self.dimension], click)
					self.a +=1
					clusterNow.append(self.cluster[i].ID)
			#clusterNow.append(userID)
			#clusterNow.append(articlePicked.id)
			if (clusterNow!=[]):
				clusterNow.append(self.a)
				clusterNow.append(userID)
				#print(clusterNow)
		with open(self.filenameWritePara, 'a+') as f:
			if(self.cluster!=[]):
				f.write(str(self.time/self.userNum))
				for i in range(len(self.cluster)):
					f.write('\t'+str(self.cluster[i].ID))
					#print(self.filenameWritePara)
				f.write('\n')
			

		self.a=0
		self.time +=1
		
					
				
					
	def getCoTheta(self, userID):
		return self.users[userID].CoTheta

	



