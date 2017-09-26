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
	def proj(self,v1,v2):
		if v2.ndim!=1:
			result=v1-np.sum(np.dot(v1,v2[i])*v2[i] for i in range(v2.shape[0]))
		else:
			result=v1-np.dot(v1,v2)*v2 
		#print('proj',result)
		return result
	def Gram_schimidt(self,basis,v1):
		#print (basis,v1)
		#print(np.sum( np.dot(v1,b)*b  for b in basis ))
		
		if basis.ndim!=1:
			w=v1-np.sum(np.dot(v1,basis[i])*basis[i] for i in range(basis.shape[0]))
		else:
			w=v1-np.dot(v1,basis)*basis
		#w = v1 - np.sum( np.dot(v1,basis[i])*basis[i]  for i in range(basis.shape[1] ))
		#print('w',w)
		return w
	
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
			thetaVector0Final=thetaVector0/(0.5*l2_norm0)
			thetaSave.append(thetaVector0Final) #1 base vector
			zeroCheck=np.zeros((self.dimension,))
			print(thetaSave[0])
			#check_flag=True
			#for i in range(5):
			#	users.append(User(key,thetaVector0Final))
			while j<5:
				check_flag=True
				basisVector=thetaSave[0]
				for i in range(len(thetaSave)-1):
					basis=thetaSave[i+1]
					basisVector=np.vstack((basisVector,basis))
				print('basisVector',basisVector)
					
					
				while check_flag:
					thetaVector1=self.thetaFunc(self.dimension,argv=self.argv)
					#print("thetav1",thetaVector1)
					if (self.proj(thetaVector1,basisVector)!=zeroCheck).all:
						check_flag=False
							
				
				
				
					
				thetaVector1=self.Gram_schimidt(basisVector,thetaVector1)
				
				thetaVector1Final=thetaVector1/(0.5*np.linalg.norm(thetaVector1,ord=2))
				#print('mul',np.dot(thetaVector1Final,thetaSave[0]))
				#print(thetaVector1Final)
				thetaSave.append(thetaVector1Final)
				j=len(thetaSave)-1
				
		'''		
			for i in range(6):
				users.append(User(key,thetaSave[i]))
				key=key+1
		'''	
		
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
				#print('l2_norm',l2_norm)
				for i in range(5):
					users.append(User(key, thetaVector/(0.5*l2_norm)))
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
		
