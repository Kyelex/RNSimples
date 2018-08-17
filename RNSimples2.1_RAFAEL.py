# -*- coding: ISO-8859-1 -*-
#ConversÃ£o MatLab ---> Python
#JoÃ£o Pedro Dellatorre Barbosa Teles
#Rafael Tadeu Carodoso dos Santos
#Caso use na Febe, retire os comentÃ¡rios com *, e depois importe os arquivos para estinge2 
#usando scp


#import pylab
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

#[A] Data
np.random.seed(0)
C = np.array([[0,0,1,1],[0,1,0,1]]) #shape: 2x4
X = np.empty([2,0]) #matriz vazia de duas linhas
P = 30
for k in range(C.shape[1]): #k varia de 0 até 4 / C.shape[1] é o formato da dimensão colunas da matriz
		H = 0.3*np.random.rand(2,P)+np.kron(np.ones((1,P)), C[:,[k]]) 
		#kron é a multiplicação de kronecker, entre uma matriz de 1s 1x30 e em cada instância, com a matriz 2x1 na coluna k de C
		X = np.append(X, H, axis = 1) 
		#adiciona o cada valor de H no for a X, no eixo de colunas (axis=1)

X = X - np.kron(np.ones((1, X.shape[1])), np.reshape(X.mean(axis=1), (1,2)).T)
#Faz a diferença entre X e np.kron de uma matriz de 1s de 1xColunas de X, e a transposta da matriz 1x2 cujos valores são as médias das colunas de X
f1 = plt.figure() #f1 se torna a figura atual
plt.plot(X[0,:], X[1,:], 'k.') #Xv1
f1.show()
#plt.savefig('PlotXv1.png')


#t = np.empty([1,0])
#T = np.array([1,-1,-1,1])
#for k in range(C.shape[1]):
#	t = np.append(t, np.kron(np.ones((1, P)), T[k]),axis = 1)
#plt.plot(X[0,(t==1)], X[1,(t==1)], 'b.') #Xv2
#plt.savefig('PlotXv2.png')
np.savetxt('Dados.mat', X)

np.loadtxt('Dados.mat')
#[A] Randomize Data Order
np.random.seed(0)
#X = np.append(X,t,axis = 0) 
#adiciona a X os valores de t no exio vertical das linhas
X = np.append(X,np.random.randn(1, X.shape[1]), axis = 0) 
#adiciona ao exiso vertical de X uma linha de números aleatórios
X = X.T #Transpõe a matriz X
X = X[X[:,2].argsort()].T
t = X[0:2,:]
X = X[0:2,:]
K = 2 #Numbers of Layers
Delta = 10**(-6)
N = X.shape[1]
E = N
mu = 0.9
eta = 0.01
alpha = 1
Layers = [{"W":np.random.rand(2,2)-0.5,"b":np.random.rand(2,1)-0.5}, #Layer 1 (0)
		  {"W":np.random.rand(2,2)-0.5,"b":np.random.rand(2,1)-0.5}]	 #Layer 2 (1)
for k in range(K):
	Layers[k]["vb"] = np.zeros(Layers[k]["b"].shape)
	Layers[k]["vW"] = np.zeros(Layers[k]["W"].shape)

n = 1
i = 0
fim = 0


while not fim:
	for k in range(K):
		Layers[k]["db"] = np.zeros(Layers[k]["b"].shape)
		Layers[k]["dW"] = np.zeros(Layers[k]["W"].shape)
	
	if i>0:
		J = np.append(J,[0],axis=0)
	else:
		J = np.array([0],float)


	for ep in range(E):
		#[C1] Feed-Forward
		Layers[0]["x"] = X[:,[n-1]]
		for k in range(K):
			Layers[k]["u"] = Layers[k]["W"].dot(Layers[k]["x"])+Layers[k]["b"]
			Layers[k]["o"] = np.tanh(Layers[k]["u"])
			if k == K-1:
				Layers.append({"x":Layers[k]["o"]})
			else:
				Layers[k+1]["x"] = Layers[k]["o"]
		e = t[:,[n-1]] - Layers[K-1]["o"]
		

		J[i] = (J[i] + (e.T.dot(e))/2)
		#[C2] Error Backprpagation
		Layers[K]["alpha"] = e
		Layers[K]["W"] = np.eye(e.size)
		
# analisar a partir daqui!!!

		for k in range(K-1,-1,-1):
			Layers[k]["M"] = np.eye(len(Layers[k]["o"])) - (np.diagflat(Layers[k]["o"]))**2
			Layers[k]["alpha"] = Layers[k]["M"].dot(np.transpose(Layers[k+1]["W"]).dot(Layers[k+1]["alpha"]))
			Layers[k]["db"] = Layers[k]["db"] + Layers[k]["alpha"]
			Layers[k]["dW"] = Layers[k]["dW"] + np.kron(np.transpose(Layers[k]["x"]), Layers[k]["alpha"])
		n = n+1
		if n>N:
			n=1
	#[C3] Updates
	for k in range(K):
		Layers[k]["vb"] = eta*Layers[k]["db"] + mu*Layers[k]["vb"]
		Layers[k]["b"] = Layers[k]["b"] + Layers[k]["vb"]
		Layers[k]["vW"] = eta*Layers[k]["dW"] + mu*Layers[k]["vW"]
		Layers[k]["W"] = Layers[k]["W"] + Layers[k]["vW"]
	J[i] = J[i]/E
	#[C4] Stop criterion 
	if ((i>1) and ((np.absolute(J[i] - J[i-1])/J[i] < Delta) | (i>100))):
		fim = 1

	if not(fim):
		i = i+1
		if n>N:
			n = 1
		eta = eta*alpha 
	[i, J[i-1]]


#[D] Test
for n in range(X.shape[1]):
	Layers[0]["x"] = X[:,[n]]
	for k in range(K):
		Layers[k]["u"] = Layers[k]["W"].dot(Layers[k]["x"]) + Layers[k]["b"]
		Layers[k]["o"] = np.tanh(Layers[k]["u"])
		Layers[k+1]["x"] = Layers[k]["o"]
	if (np.array_equal(Layers[K-1]["o"][:,[n]],X[:,[n]])):
                Z = plt.plot(X[0,n], X[1,n], 'ko')
f2 = plt.figure()
plt.plot(J)
#plt.savefig('Resultado.png')
f2.show()