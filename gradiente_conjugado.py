import numpy as np
from sympy import *
from sympy import Lambda
x,y,z,f,a = symbols('x y z f a')

def hessianReturn(listaDerivadas,variablesList,x_0,numVar:int):
  Hessian = []
  for i in listaDerivadas:
    for j in variablesList:
      if numVar == 2:
        F = Lambda((x,y),diff(i(x,y),j))
        Hessian.append(F(x_0[0],x_0[1]))
      else:
        F = Lambda((x,y,z),diff(i(x,y,z),j))
        Hessian.append(F(x_0[0],x_0[1],x_0[2]))
  return np.array(Hessian).reshape(numVar,numVar)

def gradientConjugado(funcaoStr:str,x_0:list,EPSILON:float):
  variablesList = [x,y]
  NULLVECTOR = [0,0]
  numVar = len(x_0)
  K=0
  gradientAnt = []

  #A = matrixPositiveDef(numVar) #Usaremos a Hessiana aqui

  if numVar == 3:
    variablesList.append(z)
    NULLVECTOR.append(0)

  variablesList = tuple(variablesList)
  funcao = Lambda(variablesList,funcaoStr)
  listaDerivadas = []
  
  for i in variablesList:
    f = Lambda(variablesList , diff(funcaoStr,i )) ##Criação da fi
    listaDerivadas.append(f)


  while K>=0:

    #print(x_0)
    gradient = []

    for i in listaDerivadas:
      if numVar == 3:
        gradient.append(float(i(x_0[0],x_0[1],x_0[2])))
      else:
        gradient.append(float(i(x_0[0],x_0[1])))

    gradient = np.array(gradient)

    A = hessianReturn(listaDerivadas,variablesList,x_0,numVar) #Aqui a função hessianReturn retorna a Hessiana

    if K==0:
      d_k = -1 * gradient 
    else:
      #print('Gradiente: ',gradient,' | gradientAnt: ',gradientAnt)
      numerator = np.dot(gradient,gradient)
      denominator = np.dot(gradientAnt,gradientAnt)

      B_k = numerator / denominator
      d_k = -1*gradient + B_k*d_k
      #print('B_k: ',B_k)
    
    K+=1
    error = np.dot(gradient,gradient)

    if not np.array_equal(gradient,NULLVECTOR) and error > EPSILON: 

      #Aqui começa a rotina de redefinir tk e beta
      #print(A)
      numerator = np.dot(gradient,d_k) * -1
      denominator = np.dot(d_k,A)
      denominator = np.dot(denominator,d_k)

      t_k = numerator / denominator

      x_0 = x_0 + np.dot(t_k,d_k)

      #descobrir d_k
      gradientAnt = np.array(gradient)

    else:
      funcVal = 0
      if numVar == 2:
        funcVal = funcao(x_0[0],x_0[1])
      else:
        funcVal = funcao(x_0[0],x_0[1],x_0[2])

      print('O ponto encontrado é: ',x_0,' | Com valor de função: ',funcVal)
      #print('K = ',K)
      break

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
  
  gradientConjugado(funcao,pInicial,EPSILON)
  print('\n')

while true:
  if programInterface():
    break
