import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stat
import scipy.optimize as opt
import os

# --- Parámetros de la distribución Lognormal ---

def lognorm_lambda_i(theta, s1, s2):
  """Calcula el parámetro de escala lambda de Lognormal."""
  a0, a1 = theta[0], theta[1]
  return np.exp(a0 + a1 * s1)

def lognorm_sigma_i(theta, s1, s2):
  """Calcula el parámetro de forma sigma de Lognormal."""
  b0, b1 = theta[2], theta[3]
  return np.exp(b0 + b1 * s1)

# --- Funciones de Probabilidad ---

def probabilidad_lognorm(theta, IT, s1, s2):
  """
  Calcula la probabilidad de fallo para cada combinación de tiempo de inspección (IT)  y nivel de estrés (s1)
  """
  sigmai = lognorm_sigma_i(theta, s1, s2)
  lambdai = lognorm_lambda_i(theta, s1, s2)

  IT_col = IT[:, np.newaxis]

  prob_matrix = stat.lognorm.cdf(IT_col, s=sigmai, scale=lambdai)

  return prob_matrix.flatten()

def gen_muestra_binomial_lognorm(theta, IT, s1, s2, K, seed):
  """Genera una muestra binomial."""
  pi_theta = probabilidad_lognorm(theta, IT, s1, s2)
  np.random.seed(seed)
  return np.random.binomial(K, pi_theta)

def probabilidad_estimada(muestra, K):
  """Calcula el vector de probabilidades de fallo estimadas a partir de la muestra."""
  return muestra / K

# --- Funciones de Divergencia ---

def divergencia_lognorm(theta, alpha, IT, s1, s2, K, muestra):
  """
  Calcula la divergencia de densidad de potencia (DPD) de forma vectorizada.
  El caso alpha=0 corresponde a la divergencia de Kullback-Leibler (KL).
  """
  eps = 1e-10

  # Probabilidades teóricas basadas en el theta actual
  pi_theta1 = probabilidad_lognorm(theta, IT, s1, s2)
  pi_theta1 = np.clip(pi_theta1, eps, 1.0 - eps)
  pi_theta2 = 1 - pi_theta1
  
  # Probabilidades empíricas de la muestra
  p1 = probabilidad_estimada(muestra, K)
  p1 = np.clip(p1, eps, 1.0 - eps)
  p2 = 1 - p1
  
  if alpha == 0:
    # Divergencia de Kullback-Leibler
    div_kl_vector = K * (p1 * np.log(p1 / pi_theta1) + p2 * np.log(p2 / pi_theta2))
    total_divergence = np.sum(div_kl_vector)
  else:
    # Divergencia de densidad de potencia
    term1 = pi_theta1**(1 + alpha) + pi_theta2**(1 + alpha)
    term2 = (1 + 1/alpha) * (p1 * pi_theta1**alpha + p2 * pi_theta2**alpha)
    div_alpha_vector = K * (term1 - term2)
    total_divergence = np.sum(div_alpha_vector)
    
  K_total = len(muestra) * K
  return total_divergence / K_total

# --- Estimador y Simulación ---

def emdp(theta_inicial, alpha, IT, s1, s2, K, muestra):
  """Encuentra el estimador de mínima divergencia de densidad de potencia (EMDP)."""
  args = (alpha, IT, s1, s2, K, muestra)
  result = opt.minimize(divergencia_lognorm, theta_inicial, args=args, method='Nelder-Mead')
  
  if not result.success:
      print(f"ADVERTENCIA: La optimización falló para alpha={alpha} con el mensaje: {result.message}")
      
  return result.x

def simulacion(R, theta_0, theta_inicial, theta_cont, IT, s1, s2, K, alphas):
    """
    Realiza la simulación para estimar parámetros con y sin contaminación.
    """
    num_alphas = len(alphas)
    num_params = len(theta_0)
    
    estimators_clean = np.zeros((num_alphas, R, num_params))
    estimators_cont = np.zeros((num_alphas, R, num_params))
    
    print("Iniciando simulación Lognormal...")
    for i, alpha in enumerate(alphas):
        print(f"Procesando alpha = {alpha} ({i+1}/{num_alphas})")
        for j in range(R):
            # 1. Muestra sin contaminar
            muestra_clean = gen_muestra_binomial_lognorm(theta_0, IT, s1, s2, K, seed=j)
            
            estimators_clean[i, j, :] = emdp(theta_inicial, alpha, IT, s1, s2, K, muestra_clean)
            
            # 2. Muestra contaminada
            muestra_cont_source = gen_muestra_binomial_lognorm(theta_cont, IT, s1, s2, K, seed=j)
            muestra_contaminated = np.copy(muestra_clean)
            muestra_contaminated[0] = muestra_cont_source[0] # Contaminamos solo el primer punto

            estimators_cont[i, j, :] = emdp(theta_0, alpha, IT, s1, s2, K, muestra_contaminated)

    # --- Procesamiento de Resultados ---
    
    media_estimators_clean = np.mean(estimators_clean, axis=1)
    media_estimators_cont = np.mean(estimators_cont, axis=1)
    
    se_clean = (estimators_clean - theta_0)**2
    se_cont = (estimators_cont - theta_0)**2
    
    # MSE = Media de los SE sobre las simulaciones (R) y los parámetros.
    mse_clean = np.mean(se_clean, axis=(1, 2))
    mse_cont = np.mean(se_cont, axis=(1, 2))
    
    rmse_clean = np.sqrt(mse_clean)
    rmse_cont = np.sqrt(mse_cont)

    # --- Crear DataFrames para guardar los resultados ---
    df_estimators = pd.DataFrame(media_estimators_clean, columns=[f"param_{i+1}_mean" for i in range(num_params)])
    df_estimators["alpha"] = alphas
    
    df_estimators_cont = pd.DataFrame(media_estimators_cont, columns=[f"param_cont_{i+1}_mean" for i in range(num_params)])
    df_estimators_cont["alpha"] = alphas

    df_rmse = pd.DataFrame({"alpha": alphas, "rmse_clean": rmse_clean, "rmse_cont": rmse_cont})

    # --- Guardar en CSV ---
    try:
        output_path = "resultados_lognormal/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        df_estimators.to_csv(os.path.join(output_path, "estimators_clean.csv"), index=False)
        df_estimators_cont.to_csv(os.path.join(output_path, "estimators_cont.csv"), index=False)
        df_rmse.to_csv(os.path.join(output_path, "rmse.csv"), index=False)
        print(f"Archivos CSV guardados en la carpeta: '{output_path}'")
    except Exception as e:
        print(f"Error al guardar los archivos CSV: {e}")

    return df_estimators, df_estimators_cont, df_rmse

# --- ANÁLISIS DE FIABILIDAD---

def calcular_fiabilidad(theta, IT, s1, s2):
    """
    Calcula la fiabilidad R(t) = 1 - F(t) de forma vectorial.
    """
   
    prob_fallo_matrix = probabilidad_lognorm(theta, IT, s1, s2)

    fiabilidad_matrix = 1 - prob_fallo_matrix
  
    return fiabilidad_matrix.flatten()

def analizar_sesgo_fiabilidad(df_estimadores, theta_0, IT_test, s1_test, s2):
    """
    Calcula el sesgo de la fiabilidad para un conjunto de estimadores.
    """

    fiabilidad_verdadera = calcular_fiabilidad(theta_0, IT_test, s1_test, s2)

    num_params = len(theta_0)
    thetas_estimados = df_estimadores.iloc[:, 0:num_params]
  
    fiabilidades_estimadas = thetas_estimados.apply(
        lambda row: calcular_fiabilidad(row.values, IT_test, s1_test, s2),
        axis=1,
        result_type='expand'
    )
    
    df_sesgo = fiabilidades_estimadas.subtract(fiabilidad_verdadera, axis='columns')
    
    df_sesgo.columns = [f"Bias_R(t={t})" for t in IT_test]
    df_sesgo["alpha"] = df_estimadores["alpha"].values
    
    columnas_ordenadas = ["alpha"] + [col for col in df_sesgo.columns if col != "alpha"]
    return df_sesgo[columnas_ordenadas]

# --- DATOS PARA LA SIMULACIÓN Y EL ANÁLISIS ---

# Parámetros de simulación
R = 1000
K = 100
IT_lognorm = np.array([8, 16, 24, 36])
s1_lognorm = np.array([30, 40, 50])
s2_lognorm = np.array([0, 0, 0]) # s2 no se usa, pero la función lo espera
alphas = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])

# Parámetros del modelo
theta_0_lognorm = np.array([6, -0.1, -0.6, 0.02])
theta_inicial_lognorm = np.array([5.8, -0.08, -0.5, 0.02])
theta_cont_lognorm = np.array([7, -0.1, -0.6, 0.02])

# --- EJECUCIÓN ---

# 1. Ejecutar la simulación principal
df_est_clean, df_est_cont, df_rmse = simulacion(
    R, theta_0_lognorm, theta_inicial_lognorm, theta_cont_lognorm,
    IT_lognorm, s1_lognorm, s2_lognorm, K, alphas
)

# 2. Realizar el análisis de fiabilidad
print("\n--- Iniciando Análisis de Fiabilidad ---")

# Parámetros para el test de fiabilidad
IT_test = np.array([8,12,16])
s1_test = np.array([30]) # El nivel de estrés de prueba, como array

# Sesgo resultados de la muestra limpia
df_sesgo_clean = analizar_sesgo_fiabilidad(
    df_est_clean, theta_0_lognorm, IT_test, s1_test, s2_lognorm
)

# Sesgo de la muestra contaminada
df_sesgo_cont = analizar_sesgo_fiabilidad(
    df_est_cont, theta_0_lognorm, IT_test, s1_test, s2_lognorm
)

# Guardar los resultados del sesgo en archivos CSV
output_path = "resultados_lognormal/"
df_sesgo_clean.to_csv(os.path.join(output_path, "fiabilidad_sesgo_clean.csv"), index=False)
df_sesgo_cont.to_csv(os.path.join(output_path, "fiabilidad_sesgo_cont.csv"), index=False)
print(f"Archivos CSV de sesgo de fiabilidad guardados en '{output_path}'")

# 3. Generar Tablas LaTeX para el informe

# Combinar los resultados de sesgo en una sola tabla
df_sesgo_cont_sin_alpha = df_sesgo_cont.drop('alpha', axis=1)
df_sesgo_combinado = pd.concat([df_sesgo_clean, df_sesgo_cont_sin_alpha], axis=1)

# Generar código LaTeX
latex_table_sesgo = df_sesgo_combinado.to_latex(index=False, float_format="%.4f")
latex_table_rmse = df_rmse.to_latex(index=False, float_format="%.4f")

# Imprimir resultados finales
print("\n--- Resultados de la Simulación ---")
print("\nValores medios de los estimadores (Muestra Limpia):")
print(df_est_clean)
print("\nValores medios de los estimadores (Muestra Contaminada):")
print(df_est_cont)


print("\n--- Resultados del Análisis de Fiabilidad (Sesgo) ---")
print("\nSesgo para Muestra Limpia:")
print(df_sesgo_clean)
print("\nSesgo para Muestra Contaminada:")
print(df_sesgo_cont)

print("\n--- Tablas LaTeX Generadas ---")
print("\nTabla de Sesgo de Fiabilidad:")
print(latex_table_sesgo)
print("\nTabla de RMSE:")
print(latex_table_rmse)

print("\n--- Analizando la ubicación de IT_test en la distribución ---")
fiabilidad_verdadera_test = 1 - probabilidad_lognorm(theta_0_lognorm, IT_test, s1_test, s2_lognorm).flatten()
df_fiabilidad_test = pd.DataFrame({
    'IT_test': IT_test,
    'Fiabilidad Verdadera R(t)': fiabilidad_verdadera_test
})
print(df_fiabilidad_test)
