import numpy as np
import matplotlib . pyplot as plt
import random
import math
import pandas as pd
import scipy.stats as stat
import scipy.optimize as opt

      
#parámetro de escala
def weibull_alpha_i(theta, si1, si2):
  a0 = theta[0]
  a1 = theta[1]
  alphai =[]
  for i in si1:
    alphai.append(np.exp(a0 + a1*i))
  return np.array(alphai)

#parámetro de forma
def weibull_nu_i(theta, si1, si2):
  b0 = theta[2]
  b1 = theta[3]
  nu = []
  for i in si1:
    nu.append(np.exp(b0 + b1*i))
  return np.array(nu)

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
def probabilidad_estimada(muestra, K):
  p1 = []
  p2 = []
  for i in range(len(muestra)):
    p1.append(muestra[i]/K)
    p2.append(1 - muestra[i]/K)
  return np.array(p1)

#Divergencia de Kullback-Leibler
def divergencia_KL(theta, IT, s1, s2, muestra, K):
  pi_theta1 = probabilidad_weibull(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada(muestra, K)
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
  p1 = probabilidad_estimada(muestra, K)
  p2 = 1 - p1
  K_total = len(muestra)*K
  div_alpha = []
  
  if alpha == 0:
    for i in range(len(muestra)) :
      div = divergencia_KL(theta, IT, s1, s2, muestra, K)
      div_alpha.append(div)
      
  else:
    for i in range(len(muestra)) :
      div_alpha.append(K*((pi_theta1[i]**(1+ alpha) + pi_theta2[i]**(1+ alpha)) - (1 + 1/alpha)*((p1[i])*(pi_theta1[i])**alpha + (p2[i])*(pi_theta2[i])**alpha)))

  div_alpha_pond = (np.sum(div_alpha))/K_total
  return div_alpha_pond

#Cálculo del EMDP
def emdp(theta_inicial, alpha, IT, s1, s2, K, muestra):
  args = (alpha, IT, s1,s2, K,  muestra)
  #bounds = [(1e-3, 10),(-0.2, 1e-3), (-2, 1e-3),(-0.2, 1e-3)]
  #4.5, -0.065, -0.46, 0.05
  estimador = opt.minimize(divergencia_weibull, theta_inicial, args=args, method = 'Nelder-Mead') # bounds = bounds,
  
  return estimador.x

#Simulación

def simulacion(R, theta_0, theta_inicial, theta_cont, IT,s1,s2, K, alphas):
    
    #Se simula una muestra sin contaminar  y una muestra contaminada en función de un parámetro theta contaminado
    #Devuelve el EMDP para la muestra sin contaminar y para la muestra contaminada, así como el RMSE de ambos estimadores.
    
    media_estimador =[]
    media_estimador_cont = []
    rmse_values = []
    rmse_cont_values = []
    
    for alpha in alphas:
      estimador = []
      estimador_cont = []
      
      for j in range(R):
          
        #Se estima el valor del emdp para la muestra sin contaminar  
        muestra = gen_muestra_binomial_weibull(theta_0, IT, s1, s2, K, j)
        theta_estimador = emdp(theta_inicial, alpha, IT, s1, s2, K, muestra)
        estimador.append(theta_estimador)
        
        #Se estima el valor del emdp para la muestra contaminada
        muestra_cont = gen_muestra_binomial_weibull(theta_cont, IT, s1, s2, K, j)
        theta_estimador_cont = emdp(theta_inicial, alpha, IT, s1, s2, K, muestra_cont)
        estimador_cont.append(theta_estimador_cont)

        #Se ecalcula la media del emdp sin contaminar
      mean_estimator = np.mean(estimador, axis = 0)
      mean_estimator_cont = np.mean(estimador_cont, axis = 0)
      
      #Se calcula la media del emdp contaminado
      media_estimador.append(mean_estimator)
      media_estimador_cont.append(mean_estimator_cont)

        #Cálculo del RMSE para la muestra sin contaminar
      mse = np.mean((theta_0 - mean_estimator) ** 2)
      rmse = np.sqrt(mse)
      rmse_values.append(rmse)

        #Cálculo del RMSE para la muestra contamindada
      mse_cont = np.mean((theta_0 - mean_estimator_cont) ** 2)
      rmse_cont = np.sqrt(mse_cont)
      rmse_cont_values.append(rmse_cont)

    # Convert results to DataFrames
    df_estimators = pd.DataFrame(media_estimador, columns=[f"param_{i+1}" for i in range(len(theta_0))])
    df_estimators["alpha"] = alphas
    df_estimators_cont = pd.DataFrame(media_estimador_cont, columns=[f"param_cont_{i+1}" for i in range(len(theta_0))])
    df_estimators_cont["alpha"] = alphas

    df_rmse = pd.DataFrame({"alpha": alphas, "rmse": rmse_values, "rmse_cont": rmse_cont_values})

    # Guardar en CSV
    df_estimators.to_csv("estimators.csv", index=False)
    df_estimators_cont.to_csv("estimators_cont.csv", index=False)
    df_rmse.to_csv("rmse.csv", index=False)

    print("CSV files saved: 'estimators.csv', 'estimators_cont.csv', and 'rmse.csv'")

    return np.array(media_estimador), np.array(media_estimador_cont), np.array(rmse_values), np.array(rmse_cont_values)




#Datos para la simulación
R = 1000 #Número de simulaciones
IT_weibull = np.array([8,16,24,32,40]) #Instantes de inspección
K = 100 #Número de dispositivos
s1_weibull = np.array([30,40,50])  #niveles de estrés
s2_weibull = np.array([0,0,0])
theta_0_weibull = np.array([5.3,-0.08,0.4,-0.03]) #Theta_0
theta_inicial_weibull = np.array([5.1,-0.08,-0.4,-0.03]) #Theta inicial para la función de minimización
theta_cont_weibull = np.array([5.3,-0.065,-0.4,-0.03])#Theta contaminada para generar la muestra contaminada
alphas= np.array([0, 0.2, 0.4, 0.6, 0.8, 1]) #parámetros alpha de los que depende la DPD


media_estimador_weibull, media_estimador_cont_weibull, rmse_values_weibull, rmse_cont_values_weibull = simulacion(R,theta_0_weibull, theta_inicial_weibull, theta_cont_weibull, IT_weibull, s1_weibull, s2_weibull, K, alphas)

##################################################


#Cálculo de la fiabilidad

IT1 = np.array([8,16,24,32,40])


s_prueba = 40

def distribucion1(t, theta,s): #Función de distribución lognormal

  a0 = theta[0]
  a1 = theta[1]
  b0 = theta[2]
  b1 = theta[3]

  alphai =np.exp(a0 + a1*s_prueba)
  nui =np.exp(b0 + b1*s_prueba)

  return stat.weibull_min.cdf(t, nui, scale = alphai)

def fiabilidad(theta, IT, s): #Probabilidad de fallo para cada intervalo


  probabilidades1 = []
  probabilidades2 = []

  for l in range(len(IT)):

    probabilidades1.append(distribucion1(IT[l], theta, s))
    probabilidades2.append(1 - distribucion1(IT[l], theta, s))


  return np.array(probabilidades2)

print(fiabilidad(theta_0_weibull, IT1, s_prueba))

df = pd.read_csv("/Users/javi/TFG_MATEMATICAS/WEIBULL/estimators.csv")  # Replace with the actual CSV file path
results_list = []

for index, row in df.iterrows():
    theta = row[0:6].values  # Extract the 6 estimated parameters from each row
    alpha_value = row.iloc[-1]  # Extract alpha value

    prob_vector = fiabilidad(theta, IT1, s_prueba)  # Compute fiabilidad function

    # Store results in a dictionary format
    results_list.append([alpha_value] + prob_vector.tolist())

columns = ["alpha"] + [f"R{IT1[i]}" for i in range(len(IT1))]
results_df = pd.DataFrame(results_list, columns=columns)

# Step 6: Save results to a separate CSV file
results_df.to_csv("fiabilidad_results.csv", index=False)

# Print confirmation
print("Results saved in 'fiabilidad_results.csv'.")

df = pd.read_csv("/Users/javi/TFG_MATEMATICAS/WEIBULL/estimators_cont.csv")  # Replace with the actual CSV file path
results_list = []

for index, row in df.iterrows():
    theta = row[0:6].values  # Extract the 6 estimated parameters from each row
    alpha_value = row.iloc[-1]  # Extract alpha value

    prob_vector = fiabilidad(theta, IT1, s_prueba)  # Compute fiabilidad function

    # Store results in a dictionary format
    results_list.append([alpha_value] + prob_vector.tolist())

columns = ["alpha"] + [f"R{IT1[i]}" for i in range(len(IT1))]
results_df = pd.DataFrame(results_list, columns=columns)

# Step 6: Save results to a separate CSV file
results_df.to_csv("fiabilidad_results_cont.csv", index=False)

# Print confirmation
print("Results saved in 'fiabilidad_results_cont.csv'.")
