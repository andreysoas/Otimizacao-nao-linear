import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy import optimize

from sympy import *
a = Symbol('a')
g_1,g_2 = Function('g_1'),Function('g_2')

"""#Grad Algorithm"""

class gradient_descent:

  def __init__(self,f,gradient,epsilon,initial_point):
    self.f = f
    self.gradient = gradient
    self.epsilon = epsilon
    self.initial_point = initial_point

    self.opt_value = None
    self.opt = None
    self.exe_time = None
    self.iter_count = None

  def exe_algorithm(self):

    iter_count = 0
    x_k = self.initial_point

    ini_time = time.time()
    f_gradient = np.array([i(x_k) for i in self.gradient])
    
    while np.linalg.norm(f_gradient) > self.epsilon:
      f_a = lambda a: self.f(x_k-(a*f_gradient))
      alpha = self.newton_search(f_gradient,x_k) #optimize.minimize(f_a,x0=1,method='BFGS').x
      x_k = x_k - (alpha*f_gradient)

      f_gradient = np.array([i(x_k) for i in self.gradient])
      print(self.f(x_k))
      iter_count+=1

    end_time = time.time()

    self.opt,self.opt_value,self.iter_count,self.exe_time = x_k,f(x_k),iter_count,end_time-ini_time


  def newton_search(self,f_gradient,x_k): 

    g_1 = simplify(diff(self.f(x_k-(a*f_gradient)),a))
    g_1 = Lambda((a),g_1)
    g_2 = simplify(g_1(a).diff(a))
    g_2 = Lambda((a),g_2)
    alpha_k = 10
    print(g_1,g_2)
    while abs(g_1(alpha_k)) > self.epsilon:
      try:
        alpha_k = float(alpha_k - (g_1(alpha_k)/g_2(alpha_k)))
      except ZeroDivisionError:
        print('ZeroDivision ocurred!')
        break

    return float(alpha_k)
