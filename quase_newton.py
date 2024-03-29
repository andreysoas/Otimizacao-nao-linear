# -*- coding: utf-8 -*-
"""Quase-Newton.ipynb
xk+1 = xk + tkdk
dk = -H . gradient(x_k) , H simétrica e def positiva.
"""

import numpy as np
from sympy import *
from sympy import Lambda
import math
x,y,z,f,a = symbols('x y z f a')

def qNewton(funcaoStr:str,x_0:list,EPSILON:float):

  variablesList = [x,y]
  NULLVECTOR = [0,0]
  numVar = len(x_0)

  H_k = returnMatrix(numVar) 
  gradientAnt = np.zeros(numVar)
  x_ant = np.zeros(numVar)
  x_0 = np.array(x_0)
  gradientAnt = np.zeros(numVar)

  K = 0

  if numVar == 3:
    variablesList.append(z)
    NULLVECTOR.append(0)

  variablesList = tuple(variablesList)
  funcao = Lambda(variablesList,funcaoStr)
  listaDerivadas = []
  
  for i in variablesList:
    f = Lambda(variablesList , diff(funcaoStr,i ))
    listaDerivadas.append(f)

  while true:
    #print('pontos: ',x_0)
    error=0

    gradient = []
    
    for i in listaDerivadas:
      if numVar == 2:
        gradient.append(i(x_0[0],x_0[1]))
      else:
        gradient.append(i(x_0[0],x_0[1],x_0[2]))

    gradient = np.array(gradient)
    error = np.dot(gradient,gradient)
    

    if not np.array_equal(gradient,NULLVECTOR) and error > EPSILON:

      if K>0:
        p_k = x_0 - x_ant
        q_k = gradient - gradientAnt
        H_k = returnNewMatrix(H_k,p_k,q_k)
        
      gradientAnt = np.copy(gradient)
      x_ant = np.copy(x_0)

      
      d_k = np.dot(H_k,gradient) * -1
      
      x1 = '('+str(x_0[0]+(a*(d_k[0])))+')' #pegando os novos x e y como string
      y1 = '('+str(x_0[1]+(a*(d_k[1])))+')'
      if numVar == 3:
        z1 = '('+str(x_0[2]+(a*(d_k[2])))+')'

      gFunction = funcaoStr
      gFunction = gFunction.replace('x',x1)
      gFunction = gFunction.replace('y',y1)
      if numVar == 3:
        gFunction = gFunction.replace('z',z1)
  
      alfa = derivates(gFunction,0)
      #print("alfa: ",alfa)
      x_0 = x_0 + alfa*d_k
      
      K+=1
      #print('ponto: ',x_0)
      
    else:
      funcVal = 0
      if numVar == 2:
        funcVal = funcao(x_0[0],x_0[1])
      else:
        funcVal = funcao(x_0[0],x_0[1],x_0[2])

      print('Gradiente: ',gradient)
      print('Ponto encontrado: ',x_0,' | Valor de função: ',funcVal)
      break

def returnMatrix(numVar):
  #Função que entrega matriz definida positiva e simétrica
  if numVar == 2:
    return np.diag([2,3])
  else:
    return np.diag([3,2,3])

def matrixProdut(A,B): # considerando vetor coluna e vetor linha
  M_retorno = []
  for i in A:
    line = []
    for j in B:
      line.append(i*j)
    M_retorno.append(line)

  return np.array(M_retorno)

def returnNewMatrix(H_k,p_k,q_k):
  term1 = np.dot(H_k,q_k)
  term1 = p_k - term1
  term2 = np.dot(H_k,p_k)
  term2 = p_k - term2
  numerator = matrixProdut(term1,term2)
  denominator = float(np.dot(term1,q_k))

  H_k = H_k + (numerator/denominator)
  return H_k

def derivates(funcao,alfa_0):
  funcao_dif1 = diff(funcao,a) # Diferencia a função uma vez
  f_1 = Lambda(a,funcao_dif1)     # Cria uma função lambda com o objeto
  
  funcao_dif2 = diff(funcao,a,2)  #Diferencia a função duas vezes 
  f_2 = Lambda(a,funcao_dif2)        #cria uma função lambda com o objeto

  return iteration_method(funcao,f_1,f_2,alfa_0)

def iteration_method(funcao,f_1,f_2,alfa_0):
  EPSILON = 0.0000001
  alfa_ant = alfa_0 + 1
  while true:
    
    alfa_ant = alfa_0
    alfa_0 = float(alfa_0 - (f_1(alfa_0)/f_2(alfa_0)))
    
    if f_1(alfa_0)==0 and f_2(alfa_0)>0:
      return alfa_0
    
    elif (math.fabs(f_1(alfa_0)) < EPSILON) and (f_2(alfa_0) > 0):
      return alfa_0

def programInterface():
  
  pInicial=[]
  numVar = 2

  funcao = str(input('Defina a função desejada("end" para terminar): '))
  if funcao == 'end':
    return true

  if 'z' in funcao:
    numVar=3

  for i in range(numVar):
    x = float(input('Coordenada do ponto(uma coordenada por vez): '))
    pInicial.append(x)

  EPSILON = float(input('Erro(ex.: 0.001 , 0.0001 , ...): '))
  
  qNewton(funcao,pInicial,EPSILON)
  print('\n')

while true:
  if programInterface():
    break
