import copy
import numpy as np
from random import sample, shuffle
from scipy.sparse import csgraph
import datetime
import os.path
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import TruncatedSVD
from sklearn import cluster
from sklearn.decomposition import PCA
# local address to save simulated users, simulated articles, and results
from conf import sim_files_folder, save_address
from util_functions import featureUniform, gaussianFeature
from Articles import ArticleManager
from Users import UserManager

from lib.LinUCB import N_LinUCBAlgorithm, Uniform_LinUCBAlgorithm,Hybrid_LinUCBAlgorithm
from lib.hLinUCB import HLinUCBAlgorithm
from lib.factorUCB import FactorUCBAlgorithm
from lib.CoLin import AsyCoLinUCBAlgorithm
from lib.CLUB import *
from lib.PTS import PTSAlgorithm
from lib.UCBPMF import UCBPMFAlgorithm
from lib.CAB import *
from lib.CAB_original import *
from lib.CABdiff import *
from lib.CAB_temp import *
from lib.Lin_new import *
from lib.CAB_ideal import *
from lib.CAB_optimal import *
from lib.CAB_ideal_increase import *


class simulateOnlineData(object):
	def __init__(self, context_dimension, latent_dimension, training_iterations, testing_iterations, testing_method, plot, articles, users, 
					batchSize = 1000,
					noise = lambda : 0,
					matrixNoise = lambda:0,
					type_ = 'UniformTheta', 
					signature = '', 
					poolArticleSize = 10, 
					NoiseScale = 0,
					sparseLevel = 0,  
					epsilon = 1, Gepsilon = 1):

		self.simulation_signature = signature
		self.type = type_

		self.context_dimension = context_dimension
		self.latent_dimension = latent_dimension
		self.training_iterations = training_iterations
		self.testing_iterations = testing_iterations
		self.testing_method = testing_method
		self.plot = plot

		self.noise = noise
		self.matrixNoise = matrixNoise # noise to be added to W
		self.NoiseScale = NoiseScale
		
		self.articles = articles 
		self.users = users
		self.sparseLevel = sparseLevel

		self.poolArticleSize = poolArticleSize
		self.batchSize = batchSize
		
		#self.W = self.initializeW(epsilon)
		#self.GW = self.initializeGW(Gepsilon)
		self.W, self.W0 = self.constructAdjMatrix(sparseLevel)
		W = self.W.copy()
		#self.GW = self.constructLaplacianMatrix(W, Gepsilon)
		
	def constructGraph(self):
		n = len(self.users)	

		G = np.zeros(shape = (n, n))
		for ui in self.users:
			for uj in self.users:
				G[ui.id][uj.id] = np.dot(ui.theta, uj.theta) # is dot product sufficient
		return G
		
	def constructAdjMatrix(self, m):
		n = len(self.users)	

		G = self.constructGraph()
		W = np.identity(n)
		W0 = np.identity(n)
		'''
		W0 = np.zeros(shape = (n, n)) # corrupt version of W
		for ui in self.users:
			for uj in self.users:
				W[ui.id][uj.id] = G[ui.id][uj.id]
				sim = W[ui.id][uj.id] + self.matrixNoise() # corrupt W with noise
				if sim < 0:
					sim = 0
				W0[ui.id][uj.id] = sim
				
			# find out the top M similar users in G
			if m>0 and m<n:
				similarity = sorted(G[ui.id], reverse=True)
				threshold = similarity[m]				
				
				# trim the graph
				for i in range(n):
					if G[ui.id][i] <= threshold:
						W[ui.id][i] = 0;
						W0[ui.id][i] = 0;
					
			W[ui.id] /= sum(W[ui.id])
			W0[ui.id] /= sum(W0[ui.id])
		'''
		return [W, W0]

	def constructLaplacianMatrix(self, W, Gepsilon):
		G = W.copy()
		#Convert adjacency matrix of weighted graph to adjacency matrix of unweighted graph
		for i in self.users:
			for j in self.users:
				if G[i.id][j.id] > 0:
					G[i.id][j.id] = 1	

		L = csgraph.laplacian(G, normed = False)
		#print L
		I = np.identity(n = G.shape[0])
		GW = I + Gepsilon*L  # W is a double stochastic matrix
		print 'GW', GW
		return GW.T

	def getW(self):
		return self.W
	def getW0(self):
		return self.W0
	def getFullW(self):
		return self.FullW
	
	def getGW(self):
		return self.GW

	def getTheta(self):
		Theta = np.zeros(shape = (self.dimension, len(self.users)))
		for i in range(len(self.users)):
			Theta.T[i] = self.users[i].theta
		return Theta
	def generateUserFeature(self,W):
		svd = TruncatedSVD(n_components=20)
		result = svd.fit(W).transform(W)
		return result

	def CoTheta(self):
		for ui in self.users:
			ui.CoTheta = np.zeros(self.context_dimension+self.latent_dimension)
			for uj in self.users:
				ui.CoTheta += self.W[uj.id][ui.id] * np.asarray(uj.theta)
			print 'Users', ui.id, 'CoTheta', ui.CoTheta	
	
	def batchRecord(self, iter_):
		print "Iteration %d"%iter_, "Pool", len(self.articlePool)," Elapsed time", datetime.datetime.now() - self.startTime

	def regulateArticlePool(self):
		# Randomly generate articles
		self.articlePool = sample(self.articles, self.poolArticleSize)  
		
		

	def getReward(self, user, pickedArticle):
		return np.dot(user.CoTheta, pickedArticle.featureVector)

	def GetOptimalReward(self, user, articlePool):		
		maxReward = float('-inf')
		maxx = None
		for x in articlePool:	 
			reward = self.getReward(user, x)
			if reward > maxReward:
				maxReward = reward
				maxx = x
		return maxReward, x
	
	def getL2Diff(self, x, y):
		return np.linalg.norm(x-y) # L2 norm
	def getGamma(self,articlePool,userID,userNum,clusterNum):
		gamma=[]
		diffMin=float('inf')
		diffMax=float('-inf')
		for i in range(1):
			rewardUser=self.getReward(self.users[userID],articlePool[i])
			for j in range(len(self.users)):
				if (j/clusterNum)!=(userID/clusterNum):
					
					rewardOther=self.getReward(self.users[j],articlePool[i])
					
					diff=abs(rewardUser-rewardOther)
					#print(diff)
					if diff<diffMin:
						diffMin=diff
						#print(diff)
				else:
					rewardOther=self.getReward(self.users[j],articlePool[i])
					diff=abs(rewardUser-rewardOther)
					if diff>diffMax:
						diffMax=diff
		gamma=[diffMin,diffMax]
		#print gamma
		return gamma
	
	# This part is for selecting gamma that is only related to user
	def getGammaUser(self,pickedArticle,userID,userNum,clusterNum):
		gammaPicked=[]
		diffMin=float('inf')
		diffMax=float('-inf')		
		for j in range(len(self.users)):
			rewardUser=self.getReward(self.users[j],pickedArticle)
			for k in range(len(self.users)):
				if (k/clusterNum)!=(j/clusterNum):
					rewardOther=self.getReward(self.users[k],pickedArticle)
					diff=abs(rewardUser-rewardOther)
					
					if diff<diffMin:
						diffMin=diff
		gammaPicked.append(diffMin)
		print gammaPicked
		return gammaPicked
	# This part is for selecting gamma that is only related to article
	def getGammaArticle(self,pickedArticle,userID,userNum,clusterNum):
		gammaPicked=[]
		diffMin=float('inf')
		for j in range(len(self.articles)):
			rewardUser=self.getReward(self.users[userID],self.articles[j])
			for k in range(len(self.articles)):
				if k!=j:
					rewardOther=self.getReward(self.users[userID],self.articles[k])
					diff=abs(rewardOther-rewardUser)
					if diff<diffMin:
						diffMin=diff
		gammaPicked.append(diffMin)
		print('article',gammaPicked[0])
		return gammaPicked
	
	# This part is for selecting gamma that is related to user and article
	def getGammaPicked(self,pickedArticle,userID,userNum,clusterNum):
		gammaPicked=[]
		diffMin=float('inf')
		diffMax=float('-inf')
		rewardUser=self.getReward(self.users[userID],pickedArticle)
		for j in range(len(self.users)):
			if (j/clusterNum)!=(userID/clusterNum):
					
				rewardOther=self.getReward(self.users[j],pickedArticle)
					
				diff=abs(rewardUser-rewardOther)
					#print(diff)
				if diff<diffMin:
					diffMin=diff
					#print(diff)
			else:
				
				rewardOther=self.getReward(self.users[j],pickedArticle)
				
				diff=abs(rewardUser-rewardOther)
				
				if diff>diffMax:
					diffMax=diff
				
		gammaPicked=[diffMin,diffMax]
		
		
		#print gammaPicked
		return gammaPicked					
					
					
		  
	def runAlgorithms(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = self.startTime.strftime('_%m_%d_%H_%M') 
		filenameWriteRegret = os.path.join(save_address, 'AccRegret' + timeRun + '.csv')
		filenameWritePara = os.path.join(save_address, 'ParameterEstimation' + timeRun + '.csv')

		# compute co-theta for every user
		self.CoTheta()

		tim_ = []
		BatchCumlateRegret = {}
		AlgRegret = {}
		ThetaDiffList = {}
		CoThetaDiffList = {}
		WDiffList = {}
		VDiffList = {}
		CoThetaVDiffList = {}
		RDiffList ={}
		RVDiffList = {}
		BetaDiffList={}
		UCBList1={}
		UCBList2={}
		LCBList1={}
		LCBList2={}
		
		UCB1={}
		UCB2={}
		LCB1={}
		LCB2={}
		ThetaDiff = {}
		CoThetaDiff = {}
		BetaDiff={}
		WDiff = {}
		VDiff = {}
		CoThetaVDiff = {}
		RDiff ={}
		RVDiff = {}

		Var = {}
		
		# Initialization
		userSize = len(self.users)
		for alg_name, alg in algorithms.items():
			AlgRegret[alg_name] = []
			BatchCumlateRegret[alg_name] = []
			if alg.CanEstimateUserPreference:
				ThetaDiffList[alg_name] = []
			if alg.CanEstimateCoUserPreference:
				CoThetaDiffList[alg_name] = []
			if alg.CanEstimateW:
				WDiffList[alg_name] = []
			if alg.CanEstimateV:
				VDiffList[alg_name] = []
				CoThetaVDiffList[alg_name] = []
				RVDiffList[alg_name] = []
				RDiffList[alg_name] = []
				Var[alg_name] = []
			if alg_name=='CAB_reject'or alg_name=='CABDiff'or alg_name=='CAB_cbij'or alg_name=='CAB_cbj'or alg_name=='CAB_cbj+cbij':
				BetaDiffList[alg_name]=[]
				UCBList1[alg_name]=[]
				UCBList2[alg_name]=[]
				LCBList1[alg_name]=[]
				LCBList2[alg_name]=[]

		
		with open(filenameWriteRegret, 'w') as f:
			f.write('Time(Iteration)')
			f.write(',' + ','.join( [str(alg_name) for alg_name in algorithms.iterkeys()]))
			f.write('\n')
		
		with open(filenameWritePara, 'w') as f:
			f.write('Time(Iteration)')
			f.write(',' + ','.join([str(alg_name)+'CoTheta' for alg_name in CoThetaDiffList.iterkeys()]))
			f.write(','+ ','.join([str(alg_name)+'Theta' for alg_name in ThetaDiffList.iterkeys()]))
			f.write(','+ ','.join([str(alg_name)+'W' for alg_name in WDiffList.iterkeys()]))
			f.write(','+ ','.join([str(alg_name)+'V' for alg_name in VDiffList.iterkeys()]))
			f.write(',' + ','.join([str(alg_name)+'CoThetaV' for alg_name in CoThetaVDiffList.iterkeys()]))
			f.write(','+ ','.join([str(alg_name)+'R' for alg_name in RDiffList.iterkeys()]))
			f.write(','+ ','.join([str(alg_name)+'RV' for alg_name in RVDiffList.iterkeys()]))
			f.write('\n')
		
		
		#Testing
		
		
		for iter_ in range(self.testing_iterations):
			# prepare to record theta estimation error
			for alg_name, alg in algorithms.items():
				if alg.CanEstimateUserPreference:
					ThetaDiff[alg_name] = 0
				if alg.CanEstimateCoUserPreference:
					CoThetaDiff[alg_name] = 0
				if alg.CanEstimateW:
					WDiff[alg_name] = 0
				if alg.CanEstimateV:
					VDiff[alg_name]	= 0	
					CoThetaVDiff[alg_name] = 0	
					RVDiff[alg_name]	= 0	
				RDiff[alg_name]	= 0	
				if alg_name=='CAB_reject'or alg_name=='CABDiff'or alg_name=='CAB_cbij'or alg_name=='CAB_cbj'or alg_name=='CAB_cbj+cbij':
					BetaDiff[alg_name]=0
					UCB1[alg_name]=0
					UCB2[alg_name]=0
					LCB1[alg_name]=0
					LCB2[alg_name]=0
					
			for u in self.users:

				self.regulateArticlePool() # select random articles
				rwd={}
				noise = self.noise()
				#print(noise)
				#get optimal reward for user x at time t
				OptimalReward, OptimalArticle = self.GetOptimalReward(u, self.articlePool) 
				OptimalReward += noise
				
				'''
				if alg_name=='CLUB':
					print('alggraph',alg.Graph)
				'''
				
					
				
				for alg_name, alg in algorithms.items():
					#if alg_name=='CAB_original':
					#	gamma=self.getGamma(self.articles,u.id,userSize,5)
					rwd={}
					if alg_name=='CAB_reject'or alg_name=='CABDiff'or alg_name=='CAB_cbij'or alg_name=='CAB_cbj'or alg_name=='CAB_cbj+cbij':
						pickedArticle = alg.decide(self.articlePool, u.id,u.CoTheta)
					else:
						pickedArticle = alg.decide(self.articlePool, u.id)
					#print(pickedArticle.id)
					reward = self.getReward(u, pickedArticle) + noise #yt
					for uj in self.users:
						rwd[uj.id]=self.getReward(uj,pickedArticle)+noise
					if (self.testing_method=="online"): # for batch test, do not update while testing
						if alg_name=='CAB_reject'or alg_name=='CABDiff'or alg_name=='CAB_cbij'or alg_name=='CAB_cbj'or alg_name=='CAB_cbj+cbij':
							#gammaPicked=self.getGammaPicked(pickedArticle,u.id,userSize,5)
							#gammaPicked=self.getGammaArticle(pickedArticle,u.id,userSize,5)
							alg.updateParameters(pickedArticle, reward, u.id,100,rwd)
							[betau_i,thetaj]=alg.getbeta(u.id)
							
						elif alg_name=='CAB_User':
							#gammaPicked=self.getGammaPicked(pickedArticle,u.id,userSize,5)
							#gammaPicked=self.getGammaUser(pickedArticle,u.id,userSize,5)
							alg.updateParameters(pickedArticle, reward, u.id,gammaPicked[0])
						elif alg_name=='CAB_original':
							alg.updateParameters(pickedArticle, reward, u.id,0.01)
						elif alg_name=='CAB_true_decrease'or alg_name=='CAB_5_6' or alg_name=='CAB_3_1'or alg_name=='CAB_3_2'or  alg_name=='CAB_optimal':
							#gammaPicked=self.getGammaPicked(pickedArticle,u.id,userSize,5)
							#gammaPicked=self.getGammaArticle(pickedArticle,u.id,userSize,5)
							alg.updateParameters(pickedArticle, reward, u.id,100)
						else:
							alg.updateParameters(pickedArticle, reward, u.id)
						if alg_name =='CLUB':
							n_components= alg.updateGraphClusters(u.id,'False')
							

					regret = OptimalReward - reward	
					
					AlgRegret[alg_name].append(regret)

					if u.id == 0:
						if alg_name in ['LBFGS_random','LBFGS_random_around','LinUCB', 'LBFGS_gradient_inc']:
							means, vars = alg.getProb(self.articlePool, u.id)
							#Var[alg_name].append(vars[0])

					#update parameter estimation record
					if alg.CanEstimateUserPreference:
						ThetaDiff[alg_name] += self.getL2Diff(u.theta, alg.getTheta(u.id))
					if alg.CanEstimateCoUserPreference:
						CoThetaDiff[alg_name] += self.getL2Diff(u.CoTheta[:self.context_dimension], alg.getCoTheta(u.id)[:self.context_dimension])
						
					if alg.CanEstimateW:
						WDiff[alg_name] += self.getL2Diff(self.W.T[u.id], alg.getW(u.id))	
					if alg.CanEstimateV:
						VDiff[alg_name]	+= self.getL2Diff(self.articles[pickedArticle.id].featureVector, alg.getV(pickedArticle.id))
						CoThetaVDiff[alg_name]	+= self.getL2Diff(u.CoTheta[self.context_dimension:], alg.getCoTheta(u.id)[self.context_dimension:])
						RVDiff[alg_name] += abs(u.CoTheta[self.context_dimension:].dot(self.articles[pickedArticle.id].featureVector[self.context_dimension:]) - alg.getCoTheta(u.id)[self.context_dimension:].dot(alg.getV(pickedArticle.id)[self.context_dimension:]))
						RDiff[alg_name] += reward-noise -  alg.getCoTheta(u.id).dot(alg.getV(pickedArticle.id))
					if alg_name=='CAB_reject'or alg_name=='CABDiff'or alg_name=='CAB_cbij'or alg_name=='CAB_cbj'or alg_name=='CAB_cbj+cbij':
						beta_theta_l2=0
						for k in betau_i:
							beta_theta_l2 += self.getL2Diff(betau_i[k],-u.CoTheta+self.users[k].CoTheta)
						BetaDiff[alg_name]+=beta_theta_l2/float(userSize-1)
						UCB1[alg_name]+=alg.getUCBLCB()[0]
						UCB2[alg_name]+=alg.getUCBLCB()[1]
						LCB1[alg_name]+=alg.getUCBLCB()[2]
						LCB2[alg_name]+=alg.getUCBLCB()[3]
						#print(alg.getUCBLCB())
						
			if 'syncCoLinUCB' in algorithms:
				algorithms['syncCoLinUCB'].LateUpdate()	
			
			for alg_name, alg in algorithms.items():
				if alg.CanEstimateUserPreference:
					ThetaDiffList[alg_name] += [ThetaDiff[alg_name]/userSize]
				if alg.CanEstimateCoUserPreference:
					CoThetaDiffList[alg_name] += [CoThetaDiff[alg_name]/userSize]
					
				if alg.CanEstimateW:
					WDiffList[alg_name] += [WDiff[alg_name]/userSize]	
				if alg.CanEstimateV:
					VDiffList[alg_name] += [VDiff[alg_name]/userSize]	
					CoThetaVDiffList[alg_name] += [CoThetaVDiff[alg_name]/userSize]
					RVDiffList[alg_name] += [RVDiff[alg_name]/userSize]
					RDiffList[alg_name] += [RDiff[alg_name]/userSize]	
				if alg_name=='CAB_reject'or alg_name=='CABDiff'or alg_name=='CAB_cbij'or alg_name=='CAB_cbj'or alg_name=='CAB_cbj+cbij':
					BetaDiffList[alg_name].append(BetaDiff[alg_name]/userSize)
					UCBList1[alg_name].append(UCB1[alg_name]/userSize)
					UCBList2[alg_name].append(UCB2[alg_name]/userSize)
					LCBList1[alg_name].append(LCB1[alg_name]/userSize)
					LCBList2[alg_name].append(LCB2[alg_name]/userSize)
			if iter_%self.batchSize == 0:
				self.batchRecord(iter_)
				tim_.append(iter_)
				for alg_name in algorithms.iterkeys():
					BatchCumlateRegret[alg_name].append(sum(AlgRegret[alg_name]))

				with open(filenameWriteRegret, 'a+') as f:
					f.write(str(iter_))
					f.write(',' + ','.join([str(BatchCumlateRegret[alg_name][-1]) for alg_name in algorithms.iterkeys()]))
					f.write('\n')
				with open(filenameWritePara, 'a+') as f:
					f.write(str(iter_))
					f.write(',' + ','.join([str(CoThetaDiffList[alg_name][-1]) for alg_name in CoThetaDiffList.iterkeys()]))
					f.write(','+ ','.join([str(ThetaDiffList[alg_name][-1]) for alg_name in ThetaDiffList.iterkeys()]))
					f.write(','+ ','.join([str(WDiffList[alg_name][-1]) for alg_name in WDiffList.iterkeys()]))
					f.write(',' + ','.join([str(VDiffList[alg_name][-1]) for alg_name in VDiffList.iterkeys()]))
					f.write(',' + ','.join([str(CoThetaVDiffList[alg_name][-1]) for alg_name in CoThetaVDiffList.iterkeys()]))
					f.write(',' + ','.join([str(RVDiffList[alg_name][-1]) for alg_name in RVDiffList.iterkeys()]))
					f.write(',' + ','.join([str(RDiffList[alg_name][-1]) for alg_name in RDiffList.iterkeys()]))
					f.write('\n')
			
		if (self.plot==True): # only plot
			# plot the results	
			f, axa = plt.subplots(1, sharex=True)
			for alg_name in algorithms.iterkeys():	
				axa.plot(tim_, BatchCumlateRegret[alg_name],label = alg_name)
				print '%s: %.2f' % (alg_name, BatchCumlateRegret[alg_name][-1])
			axa.legend(loc='upper left',prop={'size':9})
			axa.set_xlabel("Iteration")
			axa.set_ylabel("Regret")
			axa.set_title("Accumulated Regret")
			plt.show()

			# plot the estimation error of co-theta
			f, axa = plt.subplots(1, sharex=True)
			time = range(self.testing_iterations)
			for alg_name, alg in algorithms.items():
				if alg.CanEstimateUserPreference:
					axa.plot(time, ThetaDiffList[alg_name], label = alg_name + '_Theta')
				if alg.CanEstimateCoUserPreference:
					axa.plot(time, CoThetaDiffList[alg_name], label = alg_name + '_CoTheta')
				# if alg.CanEstimateV:
				# 	axa.plot(time, VDiffList[alg_name], label = alg_name + '_V')			
				# 	axa.plot(time, CoThetaVDiffList[alg_name], label = alg_name + '_CoThetaV')	
				# 	axa.plot(time, RVDiffList[alg_name], label = alg_name + '_RV')	
				# 	axa.plot(time, RDiffList[alg_name], label = alg_name + '_R')		
			axa.legend(loc='upper right',prop={'size':6})
			axa.set_xlabel("Iteration")
			axa.set_ylabel("L2 Diff")
			axa.set_yscale('log')
			axa.set_title("Parameter estimation error")
			plt.show()
			
			f, axa = plt.subplots(1, sharex=True)
			time = range(self.testing_iterations)
			for alg_name, alg in algorithms.items():
				if alg_name=='CAB_reject'or alg_name=='CABDiff'or alg_name=='CAB_cbij'or alg_name=='CAB_cbj'or alg_name=='CAB_cbj+cbij':
					axa.plot(time, BetaDiffList[alg_name], label = alg_name + '_beta')
				
				# if alg.CanEstimateV:
				# 	axa.plot(time, VDiffList[alg_name], label = alg_name + '_V')			
				# 	axa.plot(time, CoThetaVDiffList[alg_name], label = alg_name + '_CoThetaV')	
				# 	axa.plot(time, RVDiffList[alg_name], label = alg_name + '_RV')	
				# 	axa.plot(time, RDiffList[alg_name], label = alg_name + '_R')		
			axa.legend(loc='upper right',prop={'size':6})
			axa.set_xlabel("Iteration")
			axa.set_ylabel("L2 Diff")
			axa.set_yscale('log')
			axa.set_title("Beta estimation error")
			plt.show()
			
			f, axa = plt.subplots(1, sharex=True)
			time = range(self.testing_iterations)
			for alg_name, alg in algorithms.items():
				if alg_name=='CAB_reject'or alg_name=='CABDiff'or alg_name=='CAB_cbij'or alg_name=='CAB_cbj'or alg_name=='CAB_cbj+cbij':
					axa.plot(time, UCBList1[alg_name], label = alg_name + '_UCB1')
					axa.plot(time, UCBList2[alg_name], label = alg_name + '_UCB2')
					axa.plot(time, LCBList1[alg_name], label = alg_name + '_LCB1')
					axa.plot(time, LCBList2[alg_name], label = alg_name + '_LCB2')
				# if alg.CanEstimateV:
				# 	axa.plot(time, VDiffList[alg_name], label = alg_name + '_V')			
				# 	axa.plot(time, CoThetaVDiffList[alg_name], label = alg_name + '_CoThetaV')	
				# 	axa.plot(time, RVDiffList[alg_name], label = alg_name + '_RV')	
				# 	axa.plot(time, RDiffList[alg_name], label = alg_name + '_R')	
			axa.legend(loc='upper right',prop={'size':6})
			axa.set_xlabel("Iteration")
			axa.set_ylabel("Bound")
			#axa.set_yscale('log')
			axa.set_title("Parameter estimation error")
			plt.show()
		

		finalRegret = {}
		for alg_name in algorithms.iterkeys():
			finalRegret[alg_name] = BatchCumlateRegret[alg_name][:-1]
		return finalRegret

def pca_articles(articles, order):
	X = []
	for i, article in enumerate(articles):
		X.append(article.featureVector)
	pca = PCA()
	X_new = pca.fit_transform(X)
	# X_new = np.asarray(X)
	print('pca variance in each dim:', pca.explained_variance_ratio_) 

	print X_new
	#default is descending order, where the latend features use least informative dimensions.
	if order == 'random':
		np.random.shuffle(X_new.T)
	elif order == 'ascend':
		X_new = np.fliplr(X_new)
	elif order == 'origin':
		X_new = X
	for i, article in enumerate(articles):
		articles[i].featureVector = X_new[i]
	return


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--alg', dest='alg', help='Select a specific algorithm, could be CoLin, hLinUCB, factorUCB, LinUCB, etc.')
	parser.add_argument('--contextdim', type=int, help='Set dimension of context features.')
	parser.add_argument('--hiddendim', type=int, help='Set dimension of hidden features.')

	args = parser.parse_args()
	algName = str(args.alg)

	if args.contextdim:
		context_dimension = args.contextdim
	else:
		context_dimension = 20
	if args.hiddendim:
		latent_dimension = args.hiddendim
	else:
		latent_dimension = 0

	training_iterations = 0
	testing_iterations = 200
	
	NoiseScale = .02

	

	alpha  = 0.02
	lambda_ = 0.1   # Initialize A
	epsilon = 0 # initialize W
	eta_ = 0.5

	n_articles = 1000
	ArticleGroups = 5
	
	n_users = 30
	UserGroups = 0
	
	poolSize = 10
	batchSize = 1

	# Matrix parameters
	matrixNoise = 0.01
	sparseLevel = n_users  # if smaller or equal to 0 or larger or enqual to usernum, matrix is fully connected


	# Parameters for GOBLin
	G_alpha = alpha
	G_lambda_ = lambda_
	Gepsilon = 1
	
	userFilename = os.path.join(sim_files_folder, "users_"+str(n_users)+"context_"+str(context_dimension)+"latent_"+str(latent_dimension)+ "Ugroups" + str(UserGroups)+".json")
	
	#"Run if there is no such file with these settings; if file already exist then comment out the below funciton"
	# we can choose to simulate users every time we run the program or simulate users once, save it to 'sim_files_folder', and keep using it.
	

	articlesFilename = os.path.join(sim_files_folder, "articles_"+str(n_articles)+"context_"+str(context_dimension)+"latent_"+str(latent_dimension)+ "Agroups" + str(ArticleGroups)+".json")
	# Similarly, we can choose to simulate articles every time we run the program or simulate articles once, save it to 'sim_files_folder', and keep using it.
	AM = ArticleManager(context_dimension+latent_dimension, n_articles=n_articles, ArticleGroups = ArticleGroups,
			FeatureFunc=featureUniform,  argv={'l2_limit':1})
	#articles = AM.simulateArticlePool()
	#AM.saveArticles(articles, articlesFilename, force=False)
	articles = AM.loadArticles(articlesFilename)
	#pca_articles(articles, 'random')
	UM = UserManager(context_dimension+latent_dimension, n_users, UserGroups = UserGroups, thetaFunc=featureUniform, argv={'l2_limit':1})
	#users = UM.simulateThetafromUsers(articles)
	#UM.saveUsers(users, userFilename, force = False)
	users = UM.loadUsers(userFilename)
	
	

	
	for i in range(len(articles)):
		articles[i].contextFeatureVector = articles[i].featureVector[:context_dimension]

	simExperiment = simulateOnlineData(context_dimension = context_dimension,
						latent_dimension = latent_dimension,
						training_iterations = training_iterations,
						testing_iterations = testing_iterations,
						testing_method = "online", # batch or online
						plot = True,
						articles=articles,
						users = users,		
						noise = lambda : np.random.normal(scale = NoiseScale),
						matrixNoise = lambda : np.random.normal(scale = matrixNoise),
						batchSize = batchSize,
						type_ = "UniformTheta", 
						signature = AM.signature,
						sparseLevel = sparseLevel,
						poolArticleSize = poolSize, NoiseScale = NoiseScale, epsilon = epsilon, Gepsilon =Gepsilon)

	print "Starting for ", simExperiment.simulation_signature
	
	algorithms = {}
	
	if algName == 'LinUCB':
		algorithms['LinUCB'] = N_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
		algorithms['LinUCB_new'] = N_LinUCBAlgorithm1(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
	if algName == 'hLinUCB':
		algorithms['hLinUCB'] = HLinUCBAlgorithm(context_dimension = context_dimension, latent_dimension = latent_dimension, alpha = 0.1, alpha2 = 0.1, lambda_ = lambda_, n = n_users, itemNum=n_articles, init='zero', window_size = -1)	
		algorithms['LinUCB'] = N_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
	if algName == 'PTS':
		algorithms['PTS'] = PTSAlgorithm(particle_num = 10, dimension = 10, n = n_users, itemNum=n_articles, sigma = np.sqrt(.5), sigmaU = 1, sigmaV = 1)
	if algName == 'HybridLinUCB':
		algorithms['HybridLinUCB'] = Hybrid_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, userFeatureList=simExperiment.generateUserFeature(simExperiment.getW()))
	if args.alg == 'UCBPMF':
		algorithms['UCBPMF'] = UCBPMFAlgorithm(dimension = 10, n = n_users, itemNum=n_articles, sigma = np.sqrt(.5), sigmaU = 1, sigmaV = 1, alpha = 0.1) 
	if args.alg == 'factorUCB':
		algorithms['FactorUCB'] = FactorUCBAlgorithm(context_dimension = context_dimension, latent_dimension = 5, alpha = 0.05, alpha2 = 0.025, lambda_ = lambda_, n = n_users, itemNum=n_articles, W = simExperiment.getW(), init='random', window_size = -1)	
		algorithms['LinUCB'] = N_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
	if args.alg == 'CoLin':
		algorithms['CoLin'] = AsyCoLinUCBAlgorithm(dimension=context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, W = simExperiment.getW())
		algorithms['LinUCB'] = N_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
	if algName == 'CLUB':
		algorithms['CLUB'] = CLUBAlgorithm(dimension =context_dimension,alpha = alpha, lambda_ = lambda_, n = n_users, alpha_2 = 0.48, cluster_init = 'Erdos-Renyi')	
	if algName=='CAB':
		#algorithms['CAB_Article']=CABAlgorithm(dimension=context_dimension,alpha=alpha,lambda_=lambda_, n=n_users,alpha_2 = 0.18)
		algorithms['LinUCB'] = N_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
		algorithms['CAB_original']=CABOriginalAlgorithm(dimension=context_dimension,alpha=alpha,lambda_=lambda_, n=n_users,alpha_2 = 0.18)
		#algorithms['CAB_nodecide']=CABAlgorithm(dimension=context_dimension,alpha=alpha,lambda_=lambda_, n=n_users,alpha_2 = 0.18)
		algorithms['CAB_cbj+cbij']=CABAlgorithm(dimension=context_dimension,alpha=alpha,lambda_=lambda_, n=n_users,alpha_2 = 0.18,option=1)
		#algorithms['CAB_cbj']=CABAlgorithm(dimension=context_dimension,alpha=alpha,lambda_=lambda_, n=n_users,alpha_2 = 0.18,option=2)
		#algorithms['CAB_cbij']=CABAlgorithm(dimension=context_dimension,alpha=alpha,lambda_=lambda_, n=n_users,alpha_2 = 0.18,option=3)
		#algorithms['CAB_reject']=CABAlgorithm(dimension=context_dimension,alpha=alpha,lambda_=lambda_, n=n_users,alpha_2 = 0.18,option=4)
		#algorithms['CAB_true_increase']=CABidealAlgorithmincrease(dimension=context_dimension,alpha=alpha,lambda_=lambda_, n=n_users,alpha_2 = 0.18)
		#algorithms['CAB_5_6']=CABidealAlgorithm(dimension=context_dimension,alpha=alpha,lambda_=lambda_, n=n_users,alpha_2 = 0.18,true_num=5, false_num=2)
		#algorithms['CAB_2_3']=CABidealAlgorithm(dimension=context_dimension,alpha=alpha,lambda_=lambda_, n=n_users,alpha_2 = 0.18,true_num=2, false_num=3)
		#algorithms['CAB_3_1']=CABidealAlgorithm(dimension=context_dimension,alpha=alpha,lambda_=lambda_, n=n_users,alpha_2 = 0.18,true_num=3, false_num=1)
		
		algorithms['CAB_optimal']=CABoptimalAlgorithm(dimension=context_dimension,alpha=alpha,lambda_=lambda_, n=n_users,alpha_2 = 0.18)
		
		#algorithms['CABDiff']=CABDiffAlgorithm(dimension=context_dimension,alpha=alpha,lambda_=lambda_, n=n_users,alpha_2 = 0.18)
		#algorithms['CLUB'] = CLUBAlgorithm(dimension =context_dimension,alpha = alpha, lambda_ = lambda_, n = n_users, alpha_2 = 0.5, cluster_init = 'Erdos-Renyi')	
	if algName == 'All':
		algorithms['LinUCB'] = N_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users)
		algorithms['hLinUCB'] = HLinUCBAlgorithm(context_dimension = context_dimension, latent_dimension = 5, alpha = 0.1, alpha2 = 0.1, lambda_ = lambda_, n = n_users, itemNum=n_articles, init='random', window_size = -1)	
		algorithms['PTS'] = PTSAlgorithm(particle_num = 10, dimension = 10, n = n_users, itemNum=n_articles, sigma = np.sqrt(.5), sigmaU = 1, sigmaV = 1)
		algorithms['HybridLinUCB'] = Hybrid_LinUCBAlgorithm(dimension = context_dimension, alpha = alpha, lambda_ = lambda_, userFeatureList=simExperiment.generateUserFeature(simExperiment.getW()))
		algorithms['UCBPMF'] = UCBPMFAlgorithm(dimension = 10, n = n_users, itemNum=n_articles, sigma = np.sqrt(.5), sigmaU = 1, sigmaV = 1, alpha = 0.1) 
		algorithms['CoLin'] = AsyCoLinUCBAlgorithm(dimension=context_dimension, alpha = alpha, lambda_ = lambda_, n = n_users, W = simExperiment.getW())
		algorithms['factorUCB'] = FactorUCBAlgorithm(context_dimension = context_dimension, latent_dimension = 5, alpha = 0.05, alpha2 = 0.025, lambda_ = lambda_, n = n_users, itemNum=n_articles, W = simExperiment.getW(), init='zero', window_size = -1)	

	simExperiment.runAlgorithms(algorithms)
