import numpy as np
import matplotlib . pyplot as plt
import random
import math
import pandas as pd
import scipy.stats as stat
import scipy.optimize as opt
import scipy.special as sp
from mpmath import hyper



def list_x(x_1,x_2,x_3):
    list_x =[x_1,x_2,x_3,x_1,x_2,x_3,x_1,x_2,x_3]
  
    return np.array(list_x)

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


########ESTIMACIÓN#######

def gamma_distribucion(theta, t, s1, s2):
  alphai =gamma_alpha_i(theta, s1, s2)
  lambdai = gamma_lambda_i(theta, s1, s2)
  dist = []
  for i in range(len(muestra)):
      dist.append(stat.gamma.cdf(t[i], alphai[i], scale = lambdai[i]))
  return np.array(dist)
    
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
  
  pi_theta1 = gamma_distribucion(theta, IT, s1, s2)
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
  pi_theta1 = gamma_distribucion(theta, IT, s1, s2)
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

  alphai =gamma_alpha_i(theta, s1,s2)
  lambdai =gamma_lambda_i(theta, s1,s2)


  return stat.gamma.pdf(IT, alphai, scale = lambdai)

def s_i_gamma(theta, IT, s1, s2):

  densidad_gammai = densidad_gamma(theta, IT, s1, s2)  

    
  si = -densidad_gammai * IT

  return si 

def hyperf (IT, theta, s1, s2):

  alphai =gamma_alpha_i(theta, s1,s2)
  lambdai =gamma_lambda_i(theta, s1,s2)

  hyperf=[]

  for i in range(len(muestra)):
    z = -IT[i]/lambdai[i]
    hyperf.append(np.float64(hyper([alphai[i],alphai[i]],[1+alphai[i],1+alphai[i]], z)))

  return hyperf

def l_i_gamma(theta, IT, s1, s2):
    
  alphai =gamma_alpha_i(theta, s1,s2)
  lambdai =gamma_lambda_i(theta, s1,s2)
  

  pi_theta1 = gamma_distribucion(theta,IT, s1, s2)
  #pi_theta2 = 1 - pi_theta1
  #p1 = probabilidad_estimada(muestra)
  #p2 = 1 - p1
  l_i=[]
  
  hypergeom_term = hyperf (IT, theta, s1, s2)  # Hypergeometric function

  for i in range(len(muestra)):
    digamma_alpha = sp.digamma(alphai[i])
    l_i.append(alphai[i]*(-digamma_alpha * pi_theta1[i] + np.log(IT[i] / lambdai[i]) *pi_theta1[i] - ((IT[i] / lambdai[i]) ** alphai[i]) /( (alphai[i] ** 2) * sp.gamma(alphai[i])) * hypergeom_term[i]))

  return l_i

def Psi_i_gamma(theta, IT, s1, s2):

  l = l_i_gamma(theta, IT, s1, s2)
  s = s_i_gamma(theta, IT, s1, s2)
  x=list_x(x_1,x_2,x_3)

  Psi_matrices =[]
  for i in range(len(muestra)):
    Psi_matrices.append(([[l[i]**2 * np.dot(x[i], x[i]), l[i]* s[i] * np.dot(x[i],x[i])],
                  [l[i] * s[i] * np.dot(x[i], x[i]), s[i]**2 * np.dot(x[i],x[i])]]))
  return np.array(Psi_matrices)

def J_alpha_gamma(theta, alpha, IT, s1, s2, K, muestra):

  K_tot = K*len(muestra)
  
  pi_theta1 = gamma_distribucion(theta, IT, s1, s2)
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
  pi_theta1 = gamma_distribucion(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada_gamma(muestra,K)
  p2 = 1 - p1
  K_alpha = np.zeros((2,2))
  Psi = Psi_i_gamma(theta, IT, s1, s2)
  eps=1e-8
  for i in range(len(muestra)):
        # Compute term1 and term2 for the i-th index
        if np.any(np.isclose([pi_theta1[i], pi_theta2[i]], 0, atol=eps)):
            t1 = pi_theta1[i] * pi_theta2[i]
            t2 = ((pi_theta1[i] + eps) ** (alpha - 1)) + ((pi_theta2[i] + eps) ** (alpha - 1))
        else:
            t1 = pi_theta1[i] * pi_theta2[i]
            t2 = (pi_theta1[i] ** (alpha - 1)) + (pi_theta2[i] ** (alpha - 1))
        
        
        scalar_factor =(t1 * (t2 ** 2))
        K_alpha += (K / K_tot) * scalar_factor * Psi[i]
        
  return  K_alpha

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

########ESTIMACIÓN#######

#Funcion de distribución weibull
def weibull_distribucion( theta,t, s1, s2): 
  alphai = weibull_alpha_i(theta, s1, s2)
  nui = weibull_nu_i(theta, s1, s2)
  dist = []
  for i in range(len(muestra)):
      dist.append(stat.weibull_min.cdf(t[i], nui[i], scale = alphai[i]))
  return np.array(dist)
  

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
  pi_theta1 = weibull_distribucion(theta, IT, s1, s2)
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
  pi_theta1 = weibull_distribucion(theta, IT, s1, s2)
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
   
  alphai =weibull_alpha_i(theta, s1,s2)
  nui =weibull_nu_i(theta, s1,s2)
  
  si=[]
  for i in range(len(muestra)):
      w =np.log(IT[i])
      mu = np.log(alphai[i])
      sigma = 1/nui[i]
      xi = np.exp((w-mu)/sigma)
      si.append(xi*np.exp(-xi)*np.log(xi))
  return si 


def l_i_weibull(theta, IT, s1, s2):
    
  
  alphai =weibull_alpha_i(theta, s1,s2)
  nui =weibull_nu_i(theta, s1,s2)
  
  
  li=[]
  for i in range(len(muestra)):
      w =np.log(IT[i])
      mu = np.log(alphai[i])
      sigma = 1/nui[i]
      xi = np.exp((w-mu)/sigma)
      li.append(xi*np.exp(-xi))
  return np.array(li)


def Psi_i_weibull(theta, IT, s1, s2):

  l = l_i_weibull(theta, IT, s1, s2)
  s = s_i_weibull(theta, IT, s1, s2)
  x=list_x(x_1,x_2,x_3)

  Psi_matrices =[]
  for i in range(len(muestra)):
    Psi_matrices.append(([[l[i]**2 * np.dot(x[i], x[i]), l[i]* s[i] * np.dot(x[i],x[i])],
                  [l[i] * s[i] * np.dot(x[i], x[i]), s[i]**2 * np.dot(x[i],x[i])]]))
  return np.array(Psi_matrices)

def J_alpha_weibull(theta, alpha, IT, s1, s2, K, muestra):

  K_tot = K*len(muestra)
  pi_theta1 = weibull_distribucion(theta, IT, s1, s2)
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
  pi_theta1 = weibull_distribucion(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada_weibull(muestra,K)
  p2 = 1 - p1
  K_alpha = np.zeros((2,2))
  Psi = Psi_i_weibull(theta, IT, s1, s2)
  eps=1e-8
  for i in range(len(muestra)):
        # Compute term1 and term2 for the i-th index
        if np.any(np.isclose([pi_theta1[i], pi_theta2[i]], 0, atol=eps)):
            t1 = pi_theta1[i] * pi_theta2[i]
            t2 = ((pi_theta1[i] + eps) ** (alpha - 1)) + ((pi_theta2[i] + eps) ** (alpha - 1))
        else:
            t1 = pi_theta1[i] * pi_theta2[i]
            t2 = (pi_theta1[i] ** (alpha - 1)) + (pi_theta2[i] ** (alpha - 1))
        
        
        scalar_factor =(t1 * (t2 ** 2))
        K_alpha += (K / K_tot) * scalar_factor * Psi[i]
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
def lognorm_lambda_i(theta, s1, s2):

  a0 = theta[0]
  a1 = theta[1]
  lambdai =[]

  for i in s1:
    lambdai.append(np.exp(a0 + a1*i))

  return np.array(lambdai)

#parámetro de forma
def lognorm_sigma_i(theta, s1, s2):
  b0 = theta[2]
  b1 = theta[3]
  sigma = []

  for i in s1:

    sigma.append(np.exp(b0 + b1*i))
  return np.array(sigma)


########ESTIMACIÓN#######

#Funcion de distribución lognormal
def lognorm_distribucion(theta,t, s1, s2): #Función de distribución

  sigmai =lognorm_sigma_i(theta, s1, s2)
  lambdai =lognorm_lambda_i(theta, s1, s2)
  dist = []
  for i in range(len(muestra)):
      dist.append(stat.lognorm.cdf(t[i], sigmai[i], scale = lambdai[i]))
  return np.array(dist)

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
  pi_theta1 = lognorm_distribucion(theta, IT, s1, s2)
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
  pi_theta1 = lognorm_distribucion(theta, IT, s1, s2)
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
   
  lambdai =lognorm_lambda_i(theta, s1,s2)
  sigmai =lognorm_sigma_i(theta, s1,s2)
  
  si=[]
  for i in range(len(muestra)):
      si.append(((np.log(lambdai[i]*IT[i]))/sigmai[i])*(stat.norm.cdf(((np.log(lambdai[i]*IT[i]))/sigmai[i]),sigmai[i],scale = lambdai[i])))
  return np.array(si)

#cálculo de delta_i1
def l_i_lognorm(theta, IT, s1, s2):
    
  lambdai =lognorm_lambda_i(theta, s1,s2)
  sigmai =lognorm_sigma_i(theta, s1,s2)
  
  
  li=[]
  for i in range(len(muestra)):
      li.append((stat.norm.cdf(((np.log(lambdai[i]*IT[i]))/sigmai[i]),sigmai[i],scale = lambdai[i])))
  return np.array(li)

#calculo de delta1
def Psi_i_lognorm(theta, IT, s1, s2):

  l = l_i_lognorm(theta, IT, s1, s2)
  s = s_i_lognorm(theta, IT, s1, s2)
  x=list_x(x_1,x_2,x_3)

  Psi_matrices =[]
  for i in range(len(muestra)):
    Psi_matrices.append(([[l[i]**2 * np.dot(x[i], x[i]), l[i]* s[i] * np.dot(x[i],x[i])],
                  [l[i] * s[i] * np.dot(x[i], x[i]), s[i]**2 * np.dot(x[i],x[i])]]))
  return np.array(Psi_matrices)

def J_alpha_lognorm(theta, alpha, IT, s1, s2, K, muestra):

  K_tot = K*len(muestra)
  pi_theta1 = lognorm_distribucion(theta, IT, s1, s2)
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
  pi_theta1 = lognorm_distribucion(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada_lognorm(muestra,K)
  p2 = 1 - p1
  K_alpha = np.zeros((2,2))
  Psi = Psi_i_lognorm(theta, IT, s1, s2)
  
  eps=1e-8
  for i in range(len(muestra)):
        # Compute term1 and term2 for the i-th index
        if np.any(np.isclose([pi_theta1[i], pi_theta2[i]], 0, atol=eps)):
            t1 = pi_theta1[i] * pi_theta2[i]
            t2 = ((pi_theta1[i] + eps) ** (alpha - 1)) + ((pi_theta2[i] + eps) ** (alpha - 1))
        else:
            t1 = pi_theta1[i] * pi_theta2[i]
            t2 = (pi_theta1[i] ** (alpha - 1)) + (pi_theta2[i] ** (alpha - 1))
        
        
        scalar_factor = float(t1 * (t2 ** 2))
        K_alpha += (K / K_tot) * scalar_factor * Psi[i]
        
  return  K_alpha

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

def estimacion(theta_inicial_gamma, theta_inicial_weibull, theta_inicial_lognorm, IT,s1,s2, K, muestra,alphas):
    
    #Se simula una muestra sin contaminar  y una muestra contaminada en función de un parámetro theta contaminado
    #Devuelve el EMDP para la muestra sin contaminar y para la muestra contaminada, así como el RMSE de ambos estimadores.
   
    for alpha in alphas:
      DIC_gamma= []
      DIC_weibull = []
      DIC_lognorm =[]
     
      theta_estimador_gamma = emdp_gamma(theta_inicial_gamma, alpha, IT, s1, s2, K, muestra)
      #estimador_gamma.append(theta_estimador_gamma)
      DIC_gamma.append(compute_DIC_gamma(theta_estimador_gamma, alpha, IT, s1, s2, K, muestra))
        
      theta_estimador_weibull = emdp_weibull(theta_inicial_weibull, alpha, IT, s1, s2, K, muestra)
      #estimador_gamma.append(theta_estimador_weibull)
      DIC_weibull.append(compute_DIC_weibull(theta_estimador_weibull, alpha, IT, s1, s2, K, muestra))
        
      theta_estimador_lognorm = emdp_lognorm(theta_inicial_lognorm, alpha, IT, s1, s2, K, muestra)
      #estimador_gamma.append(theta_estimador_lognorm)
      DIC_lognorm.append(compute_DIC_lognorm(theta_estimador_lognorm, alpha, IT, s1, s2, K, muestra))


    df_DIC = pd.DataFrame({"alpha": alphas, "DIC_gamma": DIC_gamma, "DIC_weibull": DIC_weibull, "DIC_lognorm":DIC_lognorm})
    df_DIC.to_csv("DIC_real_data.csv", index=False)


    print("CSV file saved: 'DIC_real_DATA.csv'")

    return np.array(DIC_gamma), np.array(DIC_weibull), np.array(DIC_lognorm)


theta_0 = np.array([4.3, -0.185, -0.5, 0.135])
theta_inicial_gamma = np.array([5.5, -0.01, -0.6, 0.004])
theta_inicial_weibull = np.array([5.5, -0.01, -0.6, 0.004])
#theta_inicial_weibull = np.array([4.5, -0.065, -0.46, 0.05])
#theta_inicial_lognorm = np.array([4.3, -0.185, -0.5, 0.135])
theta_inicial_lognorm = np.array([5.5, -0.01, -0.6, 0.004])

IT = [10,10,10,20,20,20,30,30,30]
s1= [308,318,328,308,318,328,308,318,328]
s2 =[0,0,0,0,0,0,0,0,0]

x_1= [1,308,0]
x_2= [1,318,0]
x_3=[1,328,0]
K=10
R=1000
alphas = np.array([0.2,0.4,0.6,0.8,1])
muestra = [3,1,6,3,5,7,7,7,9]

print(len(muestra))
print(len(IT))
print(len(s1))



DIC_gamma,DIC_weibull, DIC_lognorm =estimacion(theta_inicial_gamma, theta_inicial_weibull, theta_inicial_lognorm, IT,s1,s2, K, muestra, alphas)

