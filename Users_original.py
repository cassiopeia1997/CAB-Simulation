import numpy as np 
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
import json
from random import choice, randint

class User():
	def __init__(self, id, theta = None, CoTheta = None):
		self.id = id
		self.theta = theta
		self.CoTheta = CoTheta


class UserManager():
	def __init__(self, dimension, userNum,  UserGroups, thetaFunc, argv = None):
		self.dimension = dimension
		self.thetaFunc = thetaFunc
		self.userNum = userNum
		self.UserGroups = UserGroups
		self.argv = argv
		self.signature = "A-"+"+PA"+"+TF-"+self.thetaFunc.__name__

	def saveUsers(self, users, filename, force = False):
		fileOverWriteWarning(filename, force)
		with open(filename, 'w') as f:
			for i in range(len(users)):
				#print users[i].theta
				f.write(json.dumps((users[i].id, users[i].theta.tolist())) + '\n')
				
	def loadUsers(self, filename):
		users = []
		with open(filename, 'r') as f:
			for line in f:
				id, theta = json.loads(line)
				users.append(User(id, np.array(theta)))
		return users

	def generateMasks(self):
		mask = {}
		for i in range(self.UserGroups):
			mask[i] = np.random.randint(2, size = self.dimension)
		return mask
	
	def simulateThetafromUsers(self,articles):
		usersids = {}
		users = []
		mask = self.generateMasks()
		rewardCertain=[]
		thetaSave=[]
		if (self.UserGroups==0):
			j=0
			key=0
			thetaVector0=self.thetaFunc(self.dimension,argv=self.argv)
			l2_norm0=np.linalg.norm(thetaVector0,ord=2)
			thetaVector0Final=thetaVector0/(2*np.sqrt(2)*l2_norm0)
			thetaSave.append(thetaVector0Final)
			print(thetaSave[0])
			#for i in range(5):
			#	users.append(User(key,thetaVector0Final))
			while j<5:
				thetaVector1=self.thetaFunc(self.dimension,argv=self.argv)
				l2_norm1=np.linalg.norm(thetaVector1,ord=2)
				thetaVector1Final=thetaVector1/(2*np.sqrt(2)*l2_norm1)
				thetaSave.append(thetaVector1Final)
				
				#compute distance
				for k in range(len(thetaSave)-1):
					for l in range(len(articles)):
						reward0=np.dot(thetaSave[k], articles[l].featureVector)
						reward1=np.dot(thetaVector1Final,articles[l].featureVector)
						if abs(reward0-reward1)<0.1:
							thetaSave.pop(-1)
							break;
					'''	
					if diff<0.2 or diff==0.00:
						thetaSave.pop(-1)
						#print("stop")
						break;
					else:
						print(thetaSave[-1])
					'''
				j=len(thetaSave)-1
			for i in range(6):
				for k in range(5):
					users.append(User(key,thetaSave[i]))
					key =key+1
		'''
		if (self.UserGroups == 0):
			key=0
			for x in range(6):
				thetaVector = self.thetaFunc(self.dimension, argv = self.argv)
				l2_norm = np.linalg.norm(thetaVector, ord =2)
				print('l2_norm',l2_norm)
				for i in range(5):
					users.append(User(key, thetaVector/(2*np.sqrt(2)*l2_norm)))
					key=key+1
		else:
			for i in range(self.UserGroups):
				usersids[i] = range(self.userNum*i/self.UserGroups, (self.userNum*(i+1))/self.UserGroups)

				for key in usersids[i]:
					thetaVector = np.multiply(self.thetaFunc(self.dimension, argv = self.argv), mask[i])
					l2_norm = np.linalg.norm(thetaVector, ord =2)
					users.append(User(key, thetaVector/(2*np.sqrt(2)*l2_norm)))
		'''
		return users
	
	#def simulateThetafromUsers(self):
		
