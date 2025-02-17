import numpy as np
import matplotlib . pyplot as plt
import random
import math
import pandas as pd
import scipy.stats as stat
import scipy.optimize as opt
import scipy.special as sp
from mpmath import hyper


def list_IT(IT, s1, s2):
    
    list_IT =[]
    
    for i in range(len(s1)):
        for j in IT:
            list_IT.append(j)
    return list_IT

def list_x(IT,s1,s2,x_1,x_2):
    list_x =[]
    for j in range(len(IT)):
        list_x.append(x_1)
    for i in range(len(IT)):
        list_x.append(x_2)
    return list_x
    



#######################GAMMA####################
#parámetro de forma
def gamma_alpha_i(theta, s1,s2):
    a0 = theta[0]
    a1 = theta[1]
  
    alphai =[]
    for i in s1:
        alphai.append( np.exp(a0 + a1*i))
    return np.array(alphai)

#parámetro de escala
def gamma_lambda_i(theta, s1,s2):
    b0 = theta[2]
    b1 = theta[3]
 
    lambdai = []
    for i in s1:
        lambdai.append(np.exp(b0 + b1*i))
    return np.array(lambdai)

#lista con los parámetros de forma
def list_gamma_alpha_i(theta, s1,s2, IT):
    a0 = theta[0]
    a1 = theta[1]
  
    alphai =[]
    for i in s1:
        for j in range(len(IT)):
            alphai.append( np.exp(a0 + a1*i))
    return np.array(alphai)

#lista con los parámetros de escala
def list_gamma_lambda_i(theta, s1,s2, IT):
    b0 = theta[2]
    b1 = theta[3]
 
    lambdai = []
    for i in s1:
        for j in range(len(IT)):
            lambdai.append(np.exp(b0 + b1*i))
    return np.array(lambdai)



########ESTIMACIÓN#######

def gamma_distribucion(t, theta, s1, s2):
  alphai = gamma_alpha_i(theta, s1, s2)
  lambdai = gamma_lambda_i(theta, s1, s2)
  return stat.gamma.cdf(t, alphai, scale = lambdai)
    
#Cálculo de probabilidad de fallo en el el momento de inspección IT_i
def probabilidad_gamma(theta, IT, s1, s2):
  probabilidades1 = []
  for l in range(len(IT)):
    probabilidades1.extend(gamma_distribucion(IT[l], theta, s1 , s2))
  return np.array(probabilidades1)

#print(probabilidad_gamma_sin_contaminar(theta_0_gamma, IT_gamma, s1_gamma, s2_gamma))

#Generación de la muestra 
def gen_muestra_binomial_gamma(theta_0, IT, s1, s2, K, seed):
  n_i =  []
  pi_theta1 = probabilidad_gamma(theta_0, IT, s1, s2)
  np.random.seed(seed)
  for i in range(len(pi_theta1)):
        n_i.append(np.random.binomial(K, pi_theta1[i]))
  return np.array(n_i)

#Cálculo del vector de probabilidades de fallo para la muestra
def probabilidad_estimada_gamma(muestra, K):
  p1 = []
  p2 = []
  for i in range(len(muestra)):
    p1.append(muestra[i]/K)
    p2.append(1 - muestra[i]/K)
  return np.array(p1)

#Divergencia de Kullback-Leibler
def divergencia_KL(pi_theta1, pi_theta2, p1, p2, theta, IT, s1, s2, muestra, K):
  pi_theta1 = probabilidad_gamma(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada_gamma(muestra, K)
  p2 = 1 - p1
  div_KL = []
  eps = 1e-8
  for i in range(len(muestra)):
      if np.any(np.isclose([pi_theta1[i], pi_theta2[i], p1[i], p2[i]], 0, atol=eps)):
          div_KL.append(K*(((p1[i]+eps)* np.log((p1[i]+eps)/(pi_theta1[i]+eps))) + ((p2[i]+eps)* np.log((p2[i]+eps)/(pi_theta2[i]+eps)))))
      else:
          div_KL.append(K*((p1[i]* np.log(p1[i]/pi_theta1[i])) + (p2[i]* np.log(p2[i]/pi_theta2[i]))))
  return div_KL

#Divergencia de densidad de potencia en función del parámetro alpha
def divergencia_gamma(theta, alpha, IT, s1, s2, K, muestra):
  pi_theta1 = probabilidad_gamma(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada_gamma(muestra, K)
  p2 = 1 - p1
  K_total = len(muestra)*K
  div_alpha = []
  
  if alpha == 0:
      for i in range(len(muestra)) :
          div = divergencia_KL(pi_theta1, pi_theta2, p1, p2, theta, IT, s1, s2, muestra, K)
          div_alpha.append(div)
 
  else:
    for i in range(len(muestra)):
        term1 =(pi_theta1[i]**(1+ alpha) + pi_theta2[i]**(1+ alpha))
        term2 = (1 + 1/alpha)*((p1[i])*(pi_theta1[i])**alpha + (p2[i])*(pi_theta2[i])**alpha)
        term3 = (1/alpha)*((p1[i])**(1+alpha)+(p2[i])**(1+alpha))
        div_alpha.append(K*(term1-term2+term3))
  
  div_alpha_pond = (np.sum(div_alpha))/K_total
  return div_alpha_pond

#Cálculo del EMDP
def emdp_gamma(theta_inicial, alpha, IT, s1, s2, K, muestra):
  args = (alpha, IT, s1,s2, K,  muestra)
  #bounds = [(4, 5),(-0.1, 1e-3), (-0.5, 1e-3),(-0.08, 1e-3)]
  bounds = [(1e-3, None),(None, 1e-3), (None, 1e-3),(1e-3, None)]
  estimador = opt.minimize(divergencia_gamma, theta_inicial, args=args, bounds = bounds, method = 'L-BFGS-B') #bounds = bounds,
  return estimador.x

######CALCULO DIC GAMMA ######

def densidad_gamma(theta, IT, s1,s2): #Función de densidad

  alphai =list_gamma_alpha_i(theta, s1,s2, IT)
  lambdai =list_gamma_lambda_i(theta, s1,s2, IT)
  list_ITi =list_IT(IT,s1,s2)

  return stat.gamma.pdf(list_ITi, alphai, scale = lambdai)

def s_i_gamma(theta, IT, s1, s2):
  list_ITi = np.array(list_IT(IT, s1, s2))  
  densidad_gammai = densidad_gamma(theta, IT, s1, s2)  

    
  si = -densidad_gammai * list_ITi

  return si 



def hyperf (IT, theta, s1, s2):

  alphai =list_gamma_alpha_i(theta, s1,s2, IT)
  lambdai =list_gamma_lambda_i(theta, s1,s2, IT)
  list_ITi =list_IT(IT,s1,s2)

  hyperf=[]

  for i in range(len(list_ITi)):
    z = -list_ITi[i]/lambdai[i]
    hyperf.append(np.float64(hyper([alphai[i],alphai[i]],[1+alphai[i],1+alphai[i]], z)))

  return hyperf

def l_i_gamma(theta, IT, s1, s2):
    
  alphai =list_gamma_alpha_i(theta, s1,s2, IT)
  lambdai =list_gamma_lambda_i(theta, s1,s2, IT)
  list_ITi =list_IT(IT,s1,s2)

  pi_theta1 = probabilidad_gamma(theta, IT, s1, s2)
  #pi_theta2 = 1 - pi_theta1
  #p1 = probabilidad_estimada(muestra)
  #p2 = 1 - p1
  l_i=[]
  
  hypergeom_term = hyperf (IT, theta, s1, s2)  # Hypergeometric function

  for i in range(len(list_ITi)):
    digamma_alpha = sp.digamma(alphai[i])
    l_i.append(alphai[i]*(-digamma_alpha * pi_theta1[i] + np.log(list_ITi[i] / lambdai[i]) *pi_theta1[i] - ((list_ITi[i] / lambdai[i]) ** alphai[i]) /( (alphai[i] ** 2) * sp.gamma(alphai[i])) * hypergeom_term[i]))

  return l_i

def Psi_i_gamma(theta, IT, s1, s2):

  l = l_i_gamma(theta, IT, s1, s2)
  s = s_i_gamma(theta, IT, s1, s2)
  x=list_x(IT,s1,s2,x_1,x_2)

  Psi_matrices =[]
  for i in range(len(l)):
    Psi_matrices.append(([[l[i]**2 * np.dot(x[i], x[i]), l[i]* s[i] * np.dot(x[i],x[i])],
                  [l[i] * s[i] * np.dot(x[i], x[i]), s[i]**2 * np.dot(x[i],x[i])]]))
  return np.array(Psi_matrices)

def J_alpha_gamma(theta, alpha, IT, s1, s2, K, muestra):

  K_tot = K*len(muestra)
  pi_theta1 = probabilidad_gamma(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada_gamma(muestra,K)
  p2 = 1 - p1
  Psi = Psi_i_gamma(theta, IT, s1, s2)
  
  J_alpha = np.zeros((2,2))  # Initialize with zeros

  eps=1e-8    
  for i in range(len(muestra)):
    if np.any(np.isclose([pi_theta1[i], pi_theta2[i]] ,0, atol=eps)):
        term = ((pi_theta1[i]+eps) ** (alpha - 1)) + ((pi_theta2[i]+eps) ** (alpha - 1))
          
    else:  
        term = ((pi_theta1[i]) ** (alpha - 1)) + ((pi_theta2[i]) ** (alpha - 1))
    J_alpha += (K/K_tot) * Psi[i] * term
  return J_alpha

def K_alpha_gamma(theta, alpha, IT, s1, s2, K, muestra):

  K_tot = K*len(muestra)
  pi_theta1 = probabilidad_gamma(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada_gamma(muestra,K)
  p2 = 1 - p1
  K_alpha = np.zeros((2,2))
  Psi = Psi_i_gamma(theta, IT, s1, s2)

  eps=1e-8
  for i in range(len(muestra)):
     if np.any(np.isclose([pi_theta1[i], pi_theta2[i]] ,0, atol=eps)):
         term1 = pi_theta1[i] * pi_theta2[i]
         term2 = ((pi_theta1[i]+eps) ** (alpha - 1)) + ((pi_theta2[i]+eps) ** (alpha - 1))
     else:
         term1 = pi_theta1[i] * pi_theta2[i]
         term2 = ((pi_theta1[i]) ** (alpha - 1)) + ((pi_theta2[i]) ** (alpha - 1))
         
     K_alpha += (K / K_tot) * Psi[i] * term1 * term2 ** 2
  return K_alpha

def compute_DIC_gamma(theta, alpha, IT, s1, s2, K, muestra):
  
  n =len(muestra)
  div_alpha = divergencia_gamma(theta, alpha, IT, s1, s2, K, muestra)
  K_a =K_alpha_gamma(theta, alpha, IT, s1, s2, K, muestra)
  J_a =J_alpha_gamma(theta, alpha, IT, s1,s2, K, muestra)

  J_a_inv = np.linalg.pinv(J_a)  # Inverse of J_alpha
  trace_term = np.trace(K_a @ J_a_inv)  # Trace of product
  DIC = div_alpha + ((alpha + 1) / n) * trace_term
  return DIC
####################################
####################################

####################################WEIBULL####################################

#parámetro de escala
def weibull_alpha_i(theta, s1, s2):
  a0 = theta[0]
  a1 = theta[1]
  alphai =[]
  for i in s1:
    alphai.append(np.exp(a0 + a1*i))
  return np.array(alphai)

#parámetro de forma
def weibull_nu_i(theta, s1, s2):
  b0 = theta[2]
  b1 = theta[3]
  nu = []
  for i in s1:
    nu.append(np.exp(b0 + b1*i))
  return np.array(nu)

#lista con los parámetros de forma
def list_weibull_alpha_i(theta, s1,s2, IT):
    a0 = theta[0]
    a1 = theta[1]
  
    alphai =[]
    for i in s1:
        for j in range(len(IT)):
            alphai.append( np.exp(a0 + a1*i))
    return np.array(alphai)

#lista con los parámetros de escala
def list_weibull_nu_i(theta, s1,s2, IT):
    b0 = theta[2]
    b1 = theta[3]
 
    nui = []
    for i in s1:
        for j in range(len(IT)):
            nui.append(np.exp(b0 + b1*i))
    return np.array(nui)


########ESTIMACIÓN#######

#Funcion de distribución weibull
def weibull_distribucion(t, theta, si1, si2): 
  alphai = weibull_alpha_i(theta, si1, si2)
  nui = weibull_nu_i(theta, si1, si2)
  return stat.weibull_min.cdf(t, nui, scale = alphai)

#Cálculo de probabilidad de fallo en el el momento de inspección IT_i
def probabilidad_weibull(theta, IT, s1, s2): #Probabilidad de fallo para cada intervalo
  probabilidades1 = []
  for l in range(len(IT)):
    probabilidades1.extend(weibull_distribucion(IT[l], theta, s1 , s2))
  return np.array(probabilidades1)

#Generación de la muestra 
def gen_muestra_binomial_weibull(theta_0, IT, s1, s2, K, seed):
  n_i =  []
  pi_theta1 = probabilidad_weibull(theta_0, IT, s1, s2)
  np.random.seed(seed)
  for i in range(len(pi_theta1)):
        n_i.append(np.random.binomial(K, pi_theta1[i]))
  return np.array(n_i)

#Cálculo del vector de probabilidades de fallo para la muestra
def probabilidad_estimada_weibull(muestra, K):
  p1 = []
  p2 = []
  for i in range(len(muestra)):
    p1.append(muestra[i]/K)
    p2.append(1 - muestra[i]/K)
  return np.array(p1)

#Divergencia de Kullback-Leibler
def divergencia_KL_weibull(theta, IT, s1, s2, muestra, K):
  pi_theta1 = probabilidad_weibull(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada_weibull(muestra, K)
  p2 = 1 - p1
  div_KL = []
  eps =1e-8
  
  for i in range(len(muestra)):
      if np.any(np.isclose([pi_theta1[i], pi_theta2[i], p1[i], p2[i]], 0, atol=eps)):
          div_KL.append(K*(((p1[i]+eps)* np.log((p1[i]+eps)/(pi_theta1[i]+eps))) + ((p2[i]+eps)* np.log((p2[i]+eps)/(pi_theta2[i]+eps)))))
      else:
          div_KL.append(K*((p1[i]* np.log(p1[i]/pi_theta1[i])) + (p2[i]* np.log(p2[i]/pi_theta2[i]))))
  return div_KL

#Divergencia de densidad de potencia en función del parámetro alpha
def divergencia_weibull(theta, alpha, IT, s1, s2, K, muestra):
  pi_theta1 = probabilidad_weibull(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada_weibull(muestra, K)
  p2 = 1 - p1
  K_total = len(muestra)*K
  div_alpha = []
  
  if alpha == 0:
      for i in range(len(muestra)) :
          div = divergencia_KL_weibull(theta, IT, s1, s2, muestra, K)
          div_alpha.append(div)
 
  else:
    for i in range(len(muestra)):
        term1 =(pi_theta1[i]**(1+ alpha) + pi_theta2[i]**(1+ alpha))
        term2 = (1 + 1/alpha)*((p1[i])*(pi_theta1[i])**alpha + (p2[i])*(pi_theta2[i])**alpha)
        term3 = (1/alpha)*((p1[i])**(1+alpha)+(p2[i])**(1+alpha))
        div_alpha.append(K*(term1-term2+term3))
  
  div_alpha_pond = (np.sum(div_alpha))/K_total
  return div_alpha_pond

#Cálculo del EMDP
def emdp_weibull(theta_inicial, alpha, IT, s1, s2, K, muestra):
  args = (alpha, IT, s1,s2, K,  muestra)
  #bounds = [(4, 5),(-0.3, 1e-3), (-1, 1e-3),(-0.3, 1e-3)]
  bounds = [(1e-3, None),(None, 1e-3), (None, 1e-3),(1e-3, None)]
  estimador = opt.minimize(divergencia_weibull, theta_inicial, args=args, bounds = bounds, method = 'L-BFGS-B') #
  return estimador.x



###### CALCULO DIC WEIBULL ######


def s_i_weibull(theta, IT, s1, s2):
  list_ITi = np.array(list_IT(IT, s1, s2))  
  alphai =list_weibull_alpha_i(theta, s1,s2, IT)
  nui =list_weibull_nu_i(theta, s1,s2, IT)
  
  si=[]
  for i in range(len(list_ITi)):
      w =np.log(list_ITi[i])
      mu = np.log(alphai[i])
      sigma = 1/nui[i]
      xi = np.exp((w-mu)/sigma)
      si.append(xi*np.exp(-xi)*np.log(xi))
  return si 


def l_i_weibull(theta, IT, s1, s2):
    
  list_ITi = np.array(list_IT(IT, s1, s2))  
  alphai =list_weibull_alpha_i(theta, s1,s2, IT)
  nui =list_weibull_nu_i(theta, s1,s2, IT)
  
  
  li=[]
  for i in range(len(list_ITi)):
      w =np.log(list_ITi[i])
      mu = np.log(alphai[i])
      sigma = 1/nui[i]
      xi = np.exp((w-mu)/sigma)
      li.append(xi*np.exp(-xi))
  return np.array(li)


def Psi_i_weibull(theta, IT, s1, s2):

  l = l_i_weibull(theta, IT, s1, s2)
  s = s_i_weibull(theta, IT, s1, s2)
  x=list_x(IT,s1,s2,x_1,x_2)

  Psi_matrices =[]
  for i in range(len(l)):
    Psi_matrices.append(([[l[i]**2 * np.dot(x[i], x[i]), l[i]* s[i] * np.dot(x[i],x[i])],
                  [l[i] * s[i] * np.dot(x[i], x[i]), s[i]**2 * np.dot(x[i],x[i])]]))
  return np.array(Psi_matrices)

def J_alpha_weibull(theta, alpha, IT, s1, s2, K, muestra):

  K_tot = K*len(muestra)
  pi_theta1 = probabilidad_weibull(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada_weibull(muestra,K)
  p2 = 1 - p1
  Psi = Psi_i_weibull(theta, IT, s1, s2)
  
  J_alpha = np.zeros((2,2))  # Initialize with zeros
  eps=1e-8    
  for i in range(len(muestra)):
    if np.any(np.isclose([pi_theta1[i], pi_theta2[i]] ,0, atol=eps)):
        term = ((pi_theta1[i]+eps) ** (alpha - 1)) + ((pi_theta2[i]+eps) ** (alpha - 1))
          
    else:  
        term = ((pi_theta1[i]) ** (alpha - 1)) + ((pi_theta2[i]) ** (alpha - 1))
    J_alpha += (K/K_tot) * Psi[i] * term
  return J_alpha

def K_alpha_weibull(theta, alpha, IT, s1, s2, K, muestra):

  K_tot = K*len(muestra)
  pi_theta1 = probabilidad_weibull(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada_weibull(muestra,K)
  p2 = 1 - p1
  K_alpha = np.zeros((2,2))
  Psi = Psi_i_weibull(theta, IT, s1, s2)

  eps=1e-8
  for i in range(len(muestra)):
     if np.any(np.isclose([pi_theta1[i], pi_theta2[i]] ,0, atol=eps)):
         term1 = pi_theta1[i] * pi_theta2[i]
         term2 = ((pi_theta1[i]+eps) ** (alpha - 1)) + ((pi_theta2[i]+eps) ** (alpha - 1))
     else:
         term1 = pi_theta1[i] * pi_theta2[i]
         term2 = ((pi_theta1[i]) ** (alpha - 1)) + ((pi_theta2[i]) ** (alpha - 1))
         
     K_alpha += (K / K_tot) * Psi[i] * term1 * term2 ** 2
  return K_alpha

def compute_DIC_weibull(theta, alpha, IT, s1, s2, K, muestra):
  
  n =len(muestra)
  div_alpha = divergencia_weibull(theta, alpha, IT, s1, s2, K, muestra)
  K_a =K_alpha_weibull(theta, alpha, IT, s1, s2, K, muestra)
  J_a =J_alpha_weibull(theta, alpha, IT, s1,s2, K, muestra)

  J_a_inv = np.linalg.pinv(J_a)  # Inverse of J_alpha
  trace_term = np.trace(K_a @ J_a_inv)  # Trace of product
  DIC = div_alpha + ((alpha + 1) / n )* trace_term
  return DIC
####################################
####################################

####################################LOGNORMAL####################################

#parámetro de escala
def lognorm_lambda_i(theta, si1, si2):

  a0 = theta[0]
  a1 = theta[1]
  lambdai =[]

  for i in si1:
    lambdai.append(np.exp(a0 + a1*i))

  return np.array(lambdai)

#parámetro de forma
def lognorm_sigma_i(theta, si1, si2):
  b0 = theta[2]
  b1 = theta[3]
  sigma = []

  for i in si1:

    sigma.append(np.exp(b0 + b1*i))
  return np.array(sigma)

#lista con los parámetros de forma
def list_lognorm_lambda_i(theta, s1,s2, IT):
    a0 = theta[0]
    a1 = theta[1]
  
    lambdai =[]
    for i in s1:
        for j in range(len(IT)):
            lambdai.append( np.exp(a0 + a1*i))
    return np.array(lambdai)

#lista con los parámetros de escala
def list_lognorm_sigma_i(theta, s1,s2, IT):
    b0 = theta[2]
    b1 = theta[3]
 
    sigmai = []
    for i in s1:
        for j in range(len(IT)):
            sigmai.append(np.exp(b0 + b1*i))
    return np.array(sigmai)


########ESTIMACIÓN#######

#Funcion de distribución lognormal
def lognorm_distribucion(t, theta, si1, si2): #Función de distribución

  sigmai =lognorm_sigma_i(theta, si1, si2)
  lambdai =lognorm_lambda_i(theta, si1, si2)

  return stat.lognorm.cdf(t, sigmai, scale = lambdai)

#Cálculo de probabilidad de fallo en el el momento de inspección IT_i
def probabilidad_lognorm(theta, IT, s1, s2): 
  probabilidades1 = []
  for l in range(len(IT)):
    probabilidades1.extend(lognorm_distribucion(IT[l], theta, s1 , s2))
  return np.array(probabilidades1)

#Generación de la muestra 
def gen_muestra_binomial_lognorm(theta_0, IT, s1, s2, K, seed):
  n_i =  []
  pi_theta1 = probabilidad_lognorm(theta_0, IT, s1, s2)
  np.random.seed(seed)
  for i in range(len(pi_theta1)):
        n_i.append(np.random.binomial(K, pi_theta1[i]))
  return np.array(n_i)

#Cálculo del vector de probabilidades de fallo para la muestra
def probabilidad_estimada_lognorm(muestra, K):
  p1 = []
  p2 = []
  for i in range(len(muestra)):
    p1.append(muestra[i]/K)
    p2.append(1 - muestra[i]/K)
  return np.array(p1)

#Divergencia de Kullback-Leibler
def divergencia_KL_lognorm(theta, IT, s1, s2, muestra, K):
  pi_theta1 = probabilidad_lognorm(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada_lognorm(muestra, K)
  p2 = 1 - p1
  div_KL = []
  eps = 1e-8
  for i in range(len(muestra)):
      if np.any(np.isclose([pi_theta1[i], pi_theta2[i], p1[i], p2[i]], 0, atol=eps)):
          div_KL.append(K*(((p1[i]+eps)* np.log((p1[i]+eps)/(pi_theta1[i]+eps))) + ((p2[i]+eps)* np.log((p2[i]+eps)/(pi_theta2[i]+eps)))))
      else:
          div_KL.append(K*((p1[i]* np.log(p1[i]/pi_theta1[i])) + (p2[i]* np.log(p2[i]/pi_theta2[i]))))
  return div_KL

#Divergencia de densidad de potencia en función del parámetro alpha
def divergencia_lognorm(theta, alpha, IT, s1, s2, K, muestra):
  pi_theta1 = probabilidad_lognorm(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada_lognorm(muestra, K)
  p2 = 1 - p1
  K_total = len(muestra)*K
  div_alpha = []
  
  if alpha == 0:
      for i in range(len(muestra)) :
          div = divergencia_KL_lognorm(theta, IT, s1, s2, muestra, K)
          div_alpha.append(div)
 
  else:
    for i in range(len(muestra)):
        term1 =(pi_theta1[i]**(1+ alpha) + pi_theta2[i]**(1+ alpha))
        term2 = (1 + 1/alpha)*((p1[i])*(pi_theta1[i])**alpha + (p2[i])*(pi_theta2[i])**alpha)
        term3 = (1/alpha)*((p1[i])**(1+alpha)+(p2[i])**(1+alpha))
        div_alpha.append(K*(term1-term2+term3))
  
  div_alpha_pond = (np.sum(div_alpha))/K_total
  return div_alpha_pond

#Cálculo del EMDP
def emdp_lognorm(theta_inicial, alpha, IT, s1, s2, K, muestra):
  args = (alpha, IT, s1,s2, K,  muestra)
  #bounds = [(1e-5,1),(-0.2, 1e-3), (-1, 1e-3),(-0.3, 1e-3)]
  bounds = [(1e-3, None),(None, 1e-3), (None, 1e-3),(1e-3, None)]
  estimador = opt.minimize(divergencia_lognorm, theta_inicial, args=args, bounds = bounds, method = 'L-BFGS-B') #bounds = bounds,
  return estimador.x



###### CALCULO DIC LOGNORMAL ######

#cálculo de delta_i2
def s_i_lognorm(theta, IT, s1, s2):
  list_ITi = np.array(list_IT(IT, s1, s2))  
  lambdai =list_lognorm_lambda_i(theta, s1,s2, IT)
  sigmai =list_lognorm_sigma_i(theta, s1,s2, IT)
  
  si=[]
  for i in range(len(list_ITi)):
      si.append(((np.log(lambdai[i]*list_ITi[i]))/sigmai[i])*(stat.norm.cdf(((np.log(lambdai[i]*list_ITi[i]))/sigmai[i]),sigmai[i],scale = lambdai[i])))
  return np.array(si)

#cálculo de delta_i1
def l_i_lognorm(theta, IT, s1, s2):
    
  list_ITi = np.array(list_IT(IT, s1, s2))  
  lambdai =list_lognorm_lambda_i(theta, s1,s2, IT)
  sigmai =list_lognorm_sigma_i(theta, s1,s2, IT)
  
  
  li=[]
  for i in range(len(list_ITi)):
      li.append((stat.norm.cdf(((np.log(lambdai[i]*list_ITi[i]))/sigmai[i]),sigmai[i],scale = lambdai[i])))
  return np.array(li)

#calculo de delta1
def Psi_i_lognorm(theta, IT, s1, s2):

  l = l_i_lognorm(theta, IT, s1, s2)
  s = s_i_lognorm(theta, IT, s1, s2)
  x=list_x(IT,s1,s2,x_1,x_2)

  Psi_matrices =[]
  for i in range(len(l)):
    Psi_matrices.append(([[l[i]**2 * np.dot(x[i], x[i]), l[i]* s[i] * np.dot(x[i],x[i])],
                  [l[i] * s[i] * np.dot(x[i], x[i]), s[i]**2 * np.dot(x[i],x[i])]]))
  return np.array(Psi_matrices)

def J_alpha_lognorm(theta, alpha, IT, s1, s2, K, muestra):

  K_tot = K*len(muestra)
  pi_theta1 = probabilidad_lognorm(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada_lognorm(muestra,K)
  p2 = 1 - p1
  Psi = Psi_i_lognorm(theta, IT, s1, s2)
  
  J_alpha = np.zeros((2,2))  # Initialize with zeros
  eps=1e-8    
  for i in range(len(muestra)):
    if np.any(np.isclose([pi_theta1[i], pi_theta2[i]] ,0, atol=eps)):
        term = ((pi_theta1[i]+eps) ** (alpha - 1)) + ((pi_theta2[i]+eps) ** (alpha - 1))
          
    else:  
        term = ((pi_theta1[i]) ** (alpha - 1)) + ((pi_theta2[i]) ** (alpha - 1))
    J_alpha += (K/K_tot) * Psi[i] * term
  return J_alpha

def K_alpha_lognorm(theta, alpha, IT, s1, s2, K, muestra):

  K_tot = K*len(muestra)
  pi_theta1 = probabilidad_lognorm(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada_lognorm(muestra,K)
  p2 = 1 - p1
  K_alpha = np.zeros((2,2))
  Psi = Psi_i_lognorm(theta, IT, s1, s2)
  eps=1e-8
  for i in range(len(muestra)):
     if np.any(np.isclose([pi_theta1[i], pi_theta2[i]] ,0, atol=eps)):
         term1 = pi_theta1[i] * pi_theta2[i]
         term2 = ((pi_theta1[i]+eps) ** (alpha - 1)) + ((pi_theta2[i]+eps) ** (alpha - 1))
     else:
         term1 = pi_theta1[i] * pi_theta2[i]
         term2 = ((pi_theta1[i]) ** (alpha - 1)) + ((pi_theta2[i]) ** (alpha - 1))
         
     K_alpha += (K / K_tot) * Psi[i] * term1 * term2 ** 2
  return K_alpha

def compute_DIC_lognorm(theta, alpha, IT, s1, s2, K, muestra):
  
  n =len(muestra)
  div_alpha = divergencia_lognorm(theta, alpha, IT, s1, s2, K, muestra)
  K_a =K_alpha_lognorm(theta, alpha, IT, s1, s2, K, muestra)
  J_a =J_alpha_lognorm(theta, alpha, IT, s1,s2, K, muestra)

  J_a_inv = np.linalg.pinv(J_a)  # Inverse of J_alpha
  trace_term = np.trace(K_a @ J_a_inv)  # Trace of product
  DIC = div_alpha + ((alpha + 1) / n )* trace_term
  return DIC


######################################
######################################

def simulacion(R, theta_0, theta_inicial_gamma, theta_inicial_weibull, theta_inicial_lognorm, IT,s1,s2, K, alphas):
    
    #Se simula una muestra sin contaminar  y una muestra contaminada en función de un parámetro theta contaminado
    #Devuelve el EMDP para la muestra sin contaminar y para la muestra contaminada, así como el RMSE de ambos estimadores.
    
    media_DIC_gamma =[]
    media_DIC_weibull= []
    media_DIC_lognorm = []

    
    for alpha in alphas:
      DIC_gamma= []
      DIC_weibull = []
      DIC_lognorm =[]
      
      for j in range(R):
          
        muestra = gen_muestra_binomial_lognorm(theta_0, IT, s1, s2, K, j)  
       
        
        theta_estimador_gamma = emdp_gamma(theta_inicial_gamma, alpha, IT, s1, s2, K, muestra)
        DIC_gamma.append(compute_DIC_gamma(theta_estimador_gamma, alpha, IT, s1, s2, K, muestra))
        
        theta_estimador_weibull = emdp_weibull(theta_inicial_weibull, alpha, IT, s1, s2, K, muestra)
        DIC_weibull.append(compute_DIC_weibull(theta_estimador_weibull, alpha, IT, s1, s2, K, muestra))
        
        theta_estimador_lognorm = emdp_lognorm(theta_inicial_lognorm, alpha, IT, s1, s2, K, muestra)
        DIC_lognorm.append(compute_DIC_lognorm(theta_estimador_lognorm, alpha, IT, s1, s2, K, muestra))

     #Se ecalcula la media del emdp sin contaminar
      mean_DIC_gamma = np.mean(DIC_gamma)
      mean_DIC_weibull = np.mean(DIC_weibull)
      mean_DIC_lognorm = np.mean(DIC_lognorm)
      
      #Se calcula la media del emdp contaminado
      media_DIC_gamma.append(mean_DIC_gamma)
      media_DIC_weibull.append(mean_DIC_weibull)
      media_DIC_lognorm.append(mean_DIC_lognorm)    
    
    df_DIC = pd.DataFrame({"alpha": alphas, "DIC_gamma": media_DIC_gamma, "DIC_weibull": media_DIC_weibull, "DIC_lognorm":media_DIC_lognorm})
    df_DIC.to_csv("DIC.csv", index=False)


    print("CSV file saved: 'DIC.csv'")

    return np.array(media_DIC_gamma), np.array(media_DIC_weibull), np.array(media_DIC_lognorm)


theta_0 = np.array([4.3, -0.185, -0.5, 0.135])
theta_inicial_gamma = np.array([4.5, -0.06, -0.46, 0.05])
theta_inicial_weibull = np.array([4.4, -0.1, -0.6, 0.08])
theta_inicial_lognorm = np.array([4.3, -0.3, -0.2, 0.3])

IT = np.array([25,30,35,40,45])
s1= [30,40]
s2 = [0,0]

x_1= [1,30,0]
x_2= [1,40,0]
K=100
R=1000
alphas = np.array([0.2,0.4,0.6,0.8])

media_DIC_gamma, media_DIC_weibull, media_DIC_lognorm =simulacion(R, theta_0, theta_inicial_gamma, theta_inicial_weibull, theta_inicial_lognorm, IT,s1,s2, K, alphas)





