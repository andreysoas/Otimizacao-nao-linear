import numpy as np
from sympy import *
from sympy import Lambda
x1,x2,x3,x4,x5,x6,x7,f,t, a = symbols('x1 x2 x3 x4 x5 x6 x7 f t a')
import math
init_printing(use_unicode=True)
e = math.e

def retornoGradient(numVar:int,x_0:list,listaDerivadas:list):
  #variables = [x1,x2,x3,x4,x5,x6,x7]
  gradient = []
  if numVar == 2:
    for i in listaDerivadas:
      gradient.append(i(x_0[0],x_0[1]))
  elif numVar == 3:
    for i in listaDerivadas:
      gradient.append(i(x_0[0],x_0[1],x_0[2]))
  elif numVar == 4:
    for i in listaDerivadas:
      gradient.append(i(x_0[0],x_0[1],x_0[2],x_0[3]))
  elif numVar == 5:
    for i in listaDerivadas:
      gradient.append(i(x_0[0],x_0[1],x_0[2],x_0[3],x_0[4]))
  elif numVar == 6:
    for i in listaDerivadas:
      gradient.append(i(x_0[0],x_0[1],x_0[2],x_0[3],x_0[4],x_0[5]))
  elif numVar == 7:
    for i in listaDerivadas:
      gradient.append(i(x_0[0],x_0[1],x_0[2],x_0[3],x_0[4],x_0[5],x_0[6]))

  return gradient

def functionCalc(x_0,func):
  numVar = len(x_0)
  if numVar == 2:
    return func(x_0[0],x_0[1])
  elif numVar == 3:
    return func(x_0[0],x_0[1],x_0[2])
  elif numVar == 4:
    return func(x_0[0],x_0[1],x_0[2],x_0[3])
  elif numVar == 5:
    return func(x_0[0],x_0[1],x_0[2],x_0[3],x_0[4])
  elif numVar == 6:
    return func(x_0[0],x_0[1],x_0[2],x_0[3],x_0[4],x_0[5])
  elif numVar == 7:
    return func(x_0[0],x_0[1],x_0[2],x_0[3],x_0[4],x_0[5],x_0[6])

def metodoBarreira(funcStr,x_0,EPSILON,funcaoBarreira,funcaoOriginal,CODE):
  
  variables = [x1,x2,x3,x4,x5,x6,x7]

  numVar = len(x_0)

  variablesList = variables[0:numVar]
  variablesList = tuple(variablesList)

  fBarrier = Lambda(variablesList,funcaoBarreira)
  stopMethod = True

  tk = 0.5

  newFunc = funcStr.replace('t',str(tk))

  while stopMethod:
    if CODE == 1:
      x_0 = gradientMethod(newFunc,x_0,EPSILON)
    elif CODE == 2:
      x_0 = NewtonDown(newFunc,x_0,EPSILON)
    elif CODE == 3:
      x_0 = qNewton(newFunc,x_0,EPSILON)
    else:
      x_0 = gradientConjugado(newFunc,x_0,EPSILON)

    B = functionCalc(x_0,fBarrier)

    if abs(tk*B) < 0.01:
      fx = functionCalc(x_0,Lambda(variablesList,funcaoOriginal))
      print("Minimizador: ",x_0,", Valor de fun????o: ",fx)
      stopMethod = False
    else:
      print('ponto:',x_0)
      tk = tk*0.5
      newFunc = funcStr.replace('t',str(tk))

def gradientMethod(funcaoStr:str,x_0:list,EPSILON:float):
  #print(funcaoStr)
  variables = [x1,x2,x3,x4,x5,x6,x7]

  numVar = len(x_0)

  variablesList = variables[0:numVar]
  variablesList = tuple(variablesList)

  funcao = Lambda(variablesList,funcaoStr)
  
  #print("funcao: ",newFunc)

  listaDerivadas = []
  for i in variablesList:
    f = Lambda(variablesList, diff(funcaoStr,i))
    listaDerivadas.append(f)

  while True:
    
    error=0
    gradient = retornoGradient(numVar,x_0,listaDerivadas)

    for i in gradient:
      error += i**2

    if (error > EPSILON): #x0 n??o era ??timo ou n??o atende ao erro
   
      listFuncG = []
      for i in range(numVar):
        listFuncG.append('('+str(x_0[i]+(a*-1*(gradient[i])))+')')
 
      funcAlfa = []
      for j in listFuncG:
        funcAlfa.append(Lambda(a,j))

      j = 0
      expressaoAlfa = funcaoStr
      for i in variablesList:
        expressaoAlfa = expressaoAlfa.replace(str(i),listFuncG[j])
        j+=1

      alfa = Newton_method(expressaoAlfa,0)
      #alfa = methodArmijo(expressaoAlfa,functionCalc(x_0,funcao),gradient)

      difDecide = 0
      for i in range(numVar):
        dif = x_0[i]
        x_0[i] = funcAlfa[i](alfa) 
        difDecide += math.fabs(dif - x_0[i])
      
      if difDecide < 0.001:
        return x_0

    else:
      return x_0

def methodArmijo(funcao,valor,gradient):
  n = 0.5
  t = 1
  gradient = np.array(gradient)
  d = -1*gradient

  ft = Lambda(a,funcao)
  z = np.dot(gradient,d)

  if ft(t) > (valor + (n*t*z)):
    t = 0.8*t
  return t

def derivates(funcao,alfa_0):
 #print(funcao)
  funcao_dif1 = diff(funcao,a)
  f_1 = Lambda(a,funcao_dif1)     

  funcao_dif2 = diff(funcao,a,2)
  f_2 = Lambda(a,funcao_dif2)        

  return iteration_method(funcao,f_1,f_2,alfa_0)

def Newton_method(funcao,alfa_0):
  return derivates(funcao,alfa_0)

def iteration_method(funcao,f_1,f_2,alfa_0):
  #print(funcao)
  EPSILON = 0.00001
  NUM_ITER = 500
  itercount = 1
  while itercount <= NUM_ITER:

    denominador = f_2(alfa_0)
    numerador = f_1(alfa_0)

    if abs(denominador)<0.00000001:
      break

    alfa_0 = float(alfa_0 - float(numerador/denominador))
    
    if numerador==0 and denominador>0:
      break

    elif (math.fabs(numerador) < EPSILON) and denominador > 0:
      break
    
    itercount+=1
  return alfa_0

def aureaSec(funcao:str)->float:
  iterCount = 1
  ROH = 1
  EPSILON = 0.0001
  f = Lambda(a,funcao)

  teta_1=(3-math.sqrt(5))/2
  teta_2=(sqrt(5) - 1)/2
  #Obten????o do intervalo[a,b]
  A = 0
  s = ROH
  B = 2*ROH
  while f(B) < f(s): #Busca da regi??o do m??nimo
    A = s
    s = B
    B = 2*B
  #fase 2
  u = A + teta_1*(B-A)
  v = A + teta_2*(B-A)
  while (B-A)>EPSILON and iterCount<= 100:
    if f(u)<f(v):
      B = v
      v = u
      u = A + teta_1*(B-A)
    else:
      A = u
      u = v 
      v = A + teta_2*(B-A)
  iterCount += 1
  t_bar = float((u+v)/2)
  
  return t_bar

def returnHessian(listaDerivadas,variablesList,numVar):
  Hessian = []
  for i in listaDerivadas:
    for j in variablesList:
      if numVar == 2:
        Hessian.append(Lambda((x1,x2),diff(i(x1,x2),j)))
      elif numVar == 3:
        Hessian.append(Lambda((x1,x2,x3),diff(i(x1,x2,x3),j)))
      elif numVar == 4:
        Hessian.append(Lambda((x1,x2,x3,x4),diff(i(x1,x2,x3,x4),j)))
      elif numVar == 5:
        Hessian.append(Lambda((x1,x2,x3,x4,x5),diff(i(x1,x2,x3,x4,x5),j)))
      elif numVar == 6:
        Hessian.append(Lambda((x1,x2,x3,x4,x5,x6),diff(i(x1,x2,x3,x4,x5.x6),j)))
      elif numVar == 7:
        Hessian.append(Lambda((x1,x2,x3,x4,x5,x6,x7),diff(i(x1,x2,x3,x4,x5,x6,x7),j)))
  return Hessian

def NewtonDown(funcaoStr:str, x_0:list, EPSILON:float):
  variables = [x1,x2,x3,x4,x5,x6,x7]
  #NULLVECTOR = [0,0]
  numVar = len(x_0)
  K=0

  NULLVECTOR = np.zeros(numVar)

  variablesList = variables[0:numVar]
  variablesList = tuple(variablesList)

  funcao = Lambda(variablesList,funcaoStr)
  
  listaDerivadas = []
  for i in variablesList:
    f = Lambda(variablesList , diff(funcaoStr,i )) ##Cria????o da fi
    listaDerivadas.append(f)

  x_0 = np.array(x_0)
  Hessian = []

  Hessian = returnHessian(listaDerivadas,variablesList,numVar)

  while True:

    #print('Ponto: ',x_0)

    gradient = retornoGradient(numVar,x_0,listaDerivadas)
    
    error=0
    for i in gradient:
      error+=i*i
    #print('Gradiente; ',gradient,'\n')
    if error > EPSILON:
      
      K+=1
      Hessiana = []
      
      for i in Hessian:
        valor = functionCalc(x_0,i)
        Hessiana.append(valor)

      Hessiana = np.array(Hessiana,dtype='float').reshape(numVar,numVar)
      Hessiana=np.linalg.inv(Hessiana)
 
      distance = -1*np.dot(Hessiana,gradient)
      #print(distance)
      x_0 = x_0 + distance

    else:
      return x_0

def qNewton(funcaoStr:str,x_0:list,EPSILON:float):

  variables = [x1,x2,x3,x4,x5,x6,x7]

  numVar = len(x_0)

  variablesList = variables[0:numVar]
  variablesList = tuple(variablesList)

  NULLVECTOR = np.zeros(numVar)

  H_k = returnMatrix(numVar) 
  gradientAnt = np.zeros(numVar)
  x_ant = np.zeros(numVar)
  x_0 = np.array(x_0)

  K = 0

  funcao = Lambda(variablesList,funcaoStr)
  listaDerivadas = []
  
  for i in variablesList:
    f = Lambda(variablesList , diff(funcaoStr,i ))
    listaDerivadas.append(f)
  

  while True:
    #print('pontos: ',x_0)
    error=0

    gradient = retornoGradient(numVar,x_0,listaDerivadas)
    
    gradient = np.array(gradient)
    error = np.dot(gradient,gradient)
    
    if error > EPSILON:

      if K>0:
        p_k = x_0 - x_ant
        q_k = gradient - gradientAnt
        H_k = returnNewMatrix(H_k,p_k,q_k)
        
      gradientAnt = np.copy(gradient)
      x_ant = np.copy(x_0)
      
      d_k = np.dot(H_k,gradient) * -1

      gFunction = funcaoStr
      
      for i in range(numVar):
        gFunction = gFunction.replace(str(variablesList[i]),'('+str(x_0[i]+(a*(d_k[i])))+')')
  
      alfa = Newton_method(gFunction,0)
      #alfa = aureaSec(gFunction)
      #print("alfa: ",alfa)
      x_0 = x_0 + alfa*d_k
      
      K+=1
      
      
    else:
      return x_0

def returnMatrix(numVar):
  #Fun????o que entrega matriz definida positiva e sim??trica
  mid = np.random.randint(1,10,numVar)
  return np.diag(mid)

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

def hessianReturn(listaDerivadas,variablesList,x_0,numVar:int):
  Hessian = []
  for i in listaDerivadas:
    for j in variablesList:
      if numVar == 2:
        F = Lambda((x1,x2),diff(i(x1,x2),j))
        Hessian.append(F(x_0[0],x_0[1]))
     
      elif numVar == 3:
        F = Lambda((x1,x2,x3),diff(i(x1,x2,x3),j))
        Hessian.append(F(x_0[0],x_0[1],x_0[2]))
     
      elif numVar == 4:
        F = Lambda((x1,x2,x3,x4),diff(i(x1,x2,x3,x4),j))
        Hessian.append(F(x_0[0],x_0[1],x_0[2],x_0[3]))
      
      elif numVar == 5:
        F = Lambda((x1,x2,x3,x4,x5),diff(i(x1,x2,x3,x4,x5),j))
        Hessian.append(F(x_0[0],x_0[1],x_0[2],x_0[3],x_0[4]))
      
      elif numVar == 6:
        F = Lambda((x1,x2,x3,x4,x5,x6),diff(i(x1,x2,x3,x4,x5),j))
        Hessian.append(F(x_0[0],x_0[1],x_0[2],x_0[3],x_0[4],x_0[5]))

      elif numVar == 7:
        F = Lambda((x1,x2,x3,x4,x5,x6,x7),diff(i(x1,x2,x3),j))
        Hessian.append(F(x_0[0],x_0[1],x_0[2],x_0[3],x_0[4],x_0[5],x_0[6]))


  return np.array(Hessian).reshape(numVar,numVar)

def gradientConjugado(funcaoStr:str,x_0:list,EPSILON:float):
  variablesList = [x1,x2,x3,x4,x5,x6,x7]

  numVar = len(x_0)
  variablesList = variablesList[0:numVar]
  K=0
  gradientAnt = []

  #A = matrixPositiveDef(numVar) #Usaremos a Hessiana aqui

  variablesList = tuple(variablesList)
  funcao = Lambda(variablesList,funcaoStr)
  listaDerivadas = []
  
  for i in variablesList:
    f = Lambda(variablesList , diff(sympify(funcaoStr),i )) ##Cria????o da fi
    listaDerivadas.append(f)


  while K>=0:

    #print(x_0)
    gradient = retornoGradient(numVar,x_0,listaDerivadas)

    gradient = np.array(gradient)

    A = hessianReturn(listaDerivadas,variablesList,x_0,numVar) #Aqui a fun????o hessianReturn retorna a Hessiana

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

    if math.sqrt(error) > EPSILON: 

      #Aqui come??a a rotina de redefinir tk e beta
      #print(A)
      numerator = np.dot(gradient,d_k) * -1
      denominator = np.dot(d_k,A)
      denominator = np.dot(denominator,d_k)

      t_k = numerator / denominator

      x_0 = x_0 + np.dot(t_k,d_k)
      #descobrir d_k
      gradientAnt = np.array(gradient)

    else:
      return x_0

def programInterface():
  
  pInicial=[]
  numVar = input("Quantidade de vari??veis('end' para terminar): ")
  if numVar == 'end':
    return True
  else:
    numVar = int(numVar)

  funcaoOriginal = str(input('Defina a fun????o desejada: '))

  #t_val = float(input('Par??metro t inicial: '))
  funcao = '('+funcaoOriginal+')'
  funcaoBarreira = ''
  K = 0
  while True:
    restrict = str(input('Defina a restri????o("end" para terminar a entrada): '))
    if restrict == 'end':
      break
    K+=1
    
    barreira = '(1/(' + restrict + '))'
    if K > 1:
      funcaoBarreira = funcaoBarreira + ' + ' + barreira
    else:
      funcaoBarreira = barreira
    barreira = ' + (-t*'+barreira+')'
    funcao = funcao + barreira
  #funcao = '('+ funcao + ') +(-t*(' + funcaoBarreira + '))'

  for i in range(numVar):
    x = float(input('Coordenada do ponto(uma coordenada por vez): '))
    pInicial.append(x)

  EPSILON = float(input('Erro(ex.: 0.001 , 0.0001 , ...): '))

  CODE = int(input('M??todo de descida: \n D??gito(1) : Gradiente \n D??gito(2) : Newton \n D??gito(3) : QuasiNewton \n D??gito(4) : GradienteConjugado\n'))

  print(funcao)
  metodoBarreira(funcao,pInicial,EPSILON,funcaoBarreira,funcaoOriginal,CODE)
  print('\n')

while True:
  if programInterface():
    break
