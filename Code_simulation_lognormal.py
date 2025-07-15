import numpy as np
import matplotlib . pyplot as plt
import random
import math
import pandas as pd
import scipy.stats as stat
import scipy.optimize as opt
 
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

#Funcion de distribución lognormal
def lognorm_distribucion(t, theta, s1, s2): #Función de distribución

  sigmai =lognorm_sigma_i(theta, s1, s2)
  lambdai =lognorm_lambda_i(theta, s1, s2)

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
def probabilidad_estimada(muestra, K):
  p1 = []
  p2 = []
  for i in range(len(muestra)):
    p1.append(muestra[i]/K)
    p2.append(1 - muestra[i]/K)
  return np.array(p1)

'''
#Divergencia de Kullback-Leibler
def divergencia_KL(theta, IT, s1, s2, muestra, K):
  pi_theta1 = probabilidad_lognorm(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada(muestra, K)
  p2 = 1 - p1
  div_KL = []
  eps = 1e-10
  for i in range(len(muestra)):
      if np.any(np.isclose([pi_theta1[i], pi_theta2[i], p1[i], p2[i]], 0, atol=eps)):
          div_KL.append(K*(((p1[i]+eps)* np.log((p1[i]+eps)/(pi_theta1[i]+eps))) + ((p2[i]+eps)* np.log((p2[i]+eps)/(pi_theta2[i]+eps)))))
      else:
          div_KL.append(K*((p1[i]* np.log(p1[i]/pi_theta1[i])) + (p2[i]* np.log(p2[i]/pi_theta2[i]))))
  return div_KL
'''

def divergencia_KL(theta, IT, s1, s2, muestra, K):
  pi_theta1 = probabilidad_lognorm(theta, IT, s1, s2)
  pi_theta2 = 1 - pi_theta1
  p1 = probabilidad_estimada(muestra, K)
  p2 = 1 - p1
  div_KL = []
  eps = 1e-10
  
  pi_theta1 = np.where(pi_theta1 == 0, eps, pi_theta1)
  pi_theta2 = np.where(pi_theta2 == 0, eps, pi_theta2)
  p1 = np.where(p1 == 0, eps, p1)
  p2 = np.where(p2 == 0, eps, p2)

  for i in range(len(muestra)):
      div_KL.append(K*((p1[i]* np.log(p1[i]/pi_theta1[i])) + (p2[i]* np.log(p2[i]/pi_theta2[i]))))
  return np.array(div_KL)

'''
#Divergencia de densidad de potencia en función del parámetro alpha
def divergencia_lognorm(theta, alpha, IT, s1, s2, K, muestra):
  pi_theta1 = probabilidad_lognorm(theta, IT, s1, s2)
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
'''

def divergencia_lognorm(theta, alpha, IT, s1, s2, K, muestra):
  eps = 1e-10 # Un epsilon pequeño
  pi_theta1 = probabilidad_lognorm(theta, IT, s1, s2)
  
  # APLICA EL CLIP AQUÍ
  pi_theta1 = np.clip(pi_theta1, eps, 1.0 - eps)

  pi_theta2 = 1.0 - pi_theta1
  
  p1 = probabilidad_estimada(muestra, K)
  p1 = np.clip(p1, eps, 1.0 - eps)
  p2 = 1.0 - p1
  K_total = len(muestra)*K
  
  # El resto del código puede simplificarse usando operaciones vectoriales de numpy
  if alpha == 0:
    div_KL = K * (p1 * np.log(p1 / pi_theta1) + p2 * np.log(p2 / pi_theta2))
    div_alpha_pond = np.sum(div_KL) / K_total
  else:
    term1 = pi_theta1**(1 + alpha) + pi_theta2**(1 + alpha)
    term2 = (1 + 1/alpha) * (p1 * pi_theta1**alpha + p2 * pi_theta2**alpha)
    #term3 = term3 = (1/alpha)*((p1)**(1+alpha)+(p2)**(1+alpha))
    div_alpha = K * (term1 - term2)
    div_alpha_pond = np.sum(div_alpha) / K_total
    
  return div_alpha_pond

#Cálculo del EMDP
def emdp(theta_inicial, alpha, IT, s1, s2, K, muestra):
  args = (alpha, IT, s1,s2, K,  muestra) 
  #bounds = [(5.0, 7.0), (-0.5, 0.0),  (-1.0,1.0), (0.0, 0.2)]
  
  estimador = opt.minimize(divergencia_lognorm, 
                           theta_inicial, 
                           args=args, 
                           method='Nelder-Mead')#'L-BFGS-B', # USA ESTE OPTIMIZADOR
                           #bounds=bounds)     # Y USA BOUNDSs
  
  return estimador.x

#Simulación

def simulacion(R, theta_0, theta_inicial, theta_cont, IT,s1,s2, K, alphas):
    #Se simula una muestra sin contaminar y una muestra contaminada en función de un parámetro theta contaminado para la primera celda
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
        muestra = gen_muestra_binomial_lognorm(theta_0, IT, s1, s2, K, j)
        theta_estimador = emdp(theta_inicial, alpha, IT, s1, s2, K, muestra)
        estimador.append(theta_estimador)
        
        #Se estima el valor del emdp para la muestra contaminada
        muestra_cont = gen_muestra_binomial_lognorm(theta_cont, IT, s1, s2, K, j)
        muestra[0] = muestra_cont[0]
        theta_estimador_cont = emdp(theta_inicial, alpha, IT, s1, s2, K, muestra)
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

   # Convertir a DataFrames
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
R_lognorm = 1000 #Número de simulaciones
IT_lognorm = np.array([8,16,24,36]) #Instantes de inspección
K = 100 #Número de dispositivos 
s1_lognorm = np.array([30,40,50]) #niveles de estrés
s2_lognorm = np.array([0,0,0])
theta_0_lognorm = np.array([6,-0.1,-0.6,0.02]) #Theta_0
theta_inicial_lognorm = np.array([5.8,-0.08,-0.5,0.02]) #Theta inicial para la función de minimización
theta_cont_lognorm = np.array([7,-0.1,0.4,0.02])#Theta contaminada para generar la muestra contaminada
alphas= np.array([0, 0.2, 0.4, 0.6, 0.8, 1]) #parámetros alpha de los que depende la DPD

media_estimador_lognorm, media_estimador_cont_lognorm, rmse_values_lognorm, rmse_cont_values_lognorm = simulacion(R_lognorm,theta_0_lognorm, theta_inicial_lognorm, theta_cont_lognorm, IT_lognorm, s1_lognorm, s2_lognorm, K, alphas)

##################################################

#Cálculo de la fiabilidad

IT1 = np.array([10,20,30])

s_prueba = 30

def distribucion1(t, theta,s): #Función de distribución lognormal

  a0 = theta[0]
  a1 = theta[1]
  b0 = theta[2]
  b1 = theta[3]

  lambdai =np.exp(a0 + a1*s_prueba)
  sigmai =np.exp(b0 + b1*s_prueba)

  return stat.lognorm.cdf(t, sigmai, scale = lambdai)

def fiabilidad(theta, IT, s): #Probabilidad de fallo para cada intervalo

  probabilidades1 = []
  probabilidades2 = []

  for l in range(len(IT)):

    probabilidades1.append(distribucion1(IT[l], theta, s))
    probabilidades2.append(1 - distribucion1(IT[l], theta, s))

  return np.array(probabilidades2)

lista_probs = fiabilidad(theta_0_lognorm, IT1, s_prueba)

df = pd.read_csv("C:/Users/J.ESPLUGUESGARCIA/OneDrive - Zurich Insurance/Uni/TFG_matematicas_Code/lognorm/estimators.csv")  # Replace with the actual CSV file path
results_list = []

for index, row in df.iterrows():
    theta = row[0:4].values  # Extract the 4 estimated parameters from each row
    alpha_value = row.iloc[-1]  # Extract alpha value

    prob_vector = fiabilidad(theta, IT1, s_prueba)  # Compute fiabilidad function

    # Store results in a dictionary format
    results_list.append([alpha_value] + prob_vector.tolist())

columns = ["alpha"] + [f"R{IT1[i]}" for i in range(len(IT1))]
columnas = [f"R{IT1[i]}" for i in range(len(IT1))]
results_df1 = pd.DataFrame(results_list, columns=columns)
for i, col in enumerate(columnas):
    results_df1[col] = results_df1[col] - lista_probs[i]
# Step 6: Save results to a separate CSV file
results_df1.to_csv("C:/Users/J.ESPLUGUESGARCIA/OneDrive - Zurich Insurance/Uni/TFG_matematicas_Code/lognorm/fiabilidad_results.csv", index=False)

# Print confirmation
print("Results saved in 'fiabilidad_results.csv'.")

df = pd.read_csv("C:/Users/J.ESPLUGUESGARCIA/OneDrive - Zurich Insurance/Uni/TFG_matematicas_Code/lognorm/estimators_cont.csv")  # Replace with the actual CSV file path
results_list = []

for index, row in df.iterrows():
    theta = row[0:4].values  # Extract the 4 estimated parameters from each row
    alpha_value = row.iloc[-1]  # Extract alpha value

    prob_vector = fiabilidad(theta, IT1, s_prueba)  # Compute fiabilidad function

    # Store results in a dictionary format
    results_list.append([alpha_value] + prob_vector.tolist())

columns = ["alpha"] + [f"R{IT1[i]}" for i in range(len(IT1))]
columnas = [f"R{IT1[i]}" for i in range(len(IT1))]
results_df2 = pd.DataFrame(results_list, columns=columns)
for i, col in enumerate(columnas):
    results_df2[col] = results_df2[col] - lista_probs[i]

# Step 6: Save results to a separate CSV file
results_df2.to_csv("C:/Users/J.ESPLUGUESGARCIA/OneDrive - Zurich Insurance/Uni/TFG_matematicas_Code/lognorm/fiabilidad_results_cont.csv", index=False)
df2_sin_primera = results_df2.iloc[:, 1:]

# Print confirmation
print("Results saved in 'fiabilidad_results_cont.csv'.")

df_combinado = pd.concat([results_df1, df2_sin_primera], axis=1)
latex_table=df_combinado.to_latex(index=False)
df_rmse = pd.read_csv("C:/Users/J.ESPLUGUESGARCIA/OneDrive - Zurich Insurance/Uni/TFG_matematicas_Code/lognorm/rmse.csv")
tabla_latex_rmse = df_rmse.to_latex(index=False)

# Print 
print(latex_table)

print(tabla_latex_rmse)





print(lognorm_lambda_i(theta_cont_lognorm,s1_lognorm, s2_lognorm))
print(lognorm_sigma_i(theta_cont_lognorm, s1_lognorm, s2_lognorm))
#[20.08553692  7.3890561   2.71828183]
#[1.         1.22140276 1.4918247 ]


#cont
#[16.44464677  6.04964746  2.22554093]
#[1.         1.22140276 1.4918247 ]

x = np.linspace(0, 50, 1000)

scale,c = 7.3890561,1.22140276

cdf = stat.lognorm.cdf(x,c, scale = scale)

# Crear el gráfico
plt.plot(x, cdf, label=f'lognorm(c={c}, scale={scale})')
plt.title('Distribución lognorm')
plt.xlabel('x')
plt.ylabel('Densidad de probabilidad')
plt.grid(True)
plt.legend()
plt.show()

print(probabilidad_lognorm(theta_cont_lognorm, IT_lognorm,s1_lognorm, s2_lognorm))
print(probabilidad_lognorm(theta_0_lognorm, IT_lognorm,s1_lognorm, s2_lognorm))
