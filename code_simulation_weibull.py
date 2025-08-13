
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stat
import scipy.optimize as opt
import os
# --- Parámetros de la distribución Weibull ---

def weibull_alpha_i(theta, s1, s2):
  """Calcula el parámetro de escala alpha de Weibull de forma vectorizada."""
  a0 = theta[0]
  a1 = theta[1]
  return np.exp(a0 + a1 * s1)

def weibull_nu_i(theta, s1, s2):
  """Calcula el parámetro de forma nu de Weibull de forma vectorizada."""
  b0 = theta[2]
  b1 = theta[3]
  return np.exp(b0 + b1 * s1)

# --- Funciones de Probabilidad (Vectorizadas) ---

def probabilidad_weibull(theta, IT, s1, s2):
  """
  Calcula la probabilidad de fallo para cada combinación de tiempo de inspección (IT)
  y nivel de estrés (s1) usando broadcasting de NumPy.
  
  Devuelve:
      Un array 1D con todas las probabilidades.
  """
  alphai = weibull_alpha_i(theta, s1, s2)
  nui = weibull_nu_i(theta, s1, s2)
  
  # Reshape IT a un vector columna para que NumPy haga broadcasting contra los arrays de parámetros (que son vectores fila).
  # El resultado será una matriz de (len(IT), len(s1)).
  IT_col = IT[:, np.newaxis]
  
  prob_matrix = stat.weibull_min.cdf(IT_col, nui, scale=alphai)

  # Aplanamos la matriz para obtener un único vector de probabilidades
  return prob_matrix.flatten()

def gen_muestra_binomial_weibull(theta, IT, s1, s2, K, seed):
  """
  Genera una muestra binomial de forma vectorizada.
  
  np.random.binomial puede tomar un array de probabilidades.
  """
  pi_theta = probabilidad_weibull(theta, IT, s1, s2)
  np.random.seed(seed)
  return np.random.binomial(K, pi_theta)

def probabilidad_estimada(muestra, K):
  """Calcula el vector de probabilidades de fallo estimadas a partir de la muestra."""
  return muestra / K

# --- Funciones de Divergencia (Vectorizadas) ---

def divergencia_weibull(theta, alpha, IT, s1, s2, K, muestra):
  """
  Calcula la divergencia de densidad de potencia (DPD).
  El caso alpha=0 corresponde a la divergencia de Kullback-Leibler (KL).
  """
  eps = 1e-10  
  
  # Probabilidades teóricas basadas en el theta actual
  pi_theta1 = probabilidad_weibull(theta, IT, s1, s2)
  pi_theta1 = np.clip(pi_theta1, eps, 1.0 - eps) # Asegurar que estén en [eps, 1-eps]
  pi_theta2 = 1 - pi_theta1
  
  # Probabilidades empíricas de la muestra
  p1 = probabilidad_estimada(muestra, K)
  p1 = np.clip(p1, eps, 1.0 - eps)
  p2 = 1 - p1
  
  if alpha == 0:
    # Divergencia de Kullback-Leibler 
    div_kl = K * (p1 * np.log(p1 / pi_theta1) + p2 * np.log(p2 / pi_theta2))
    total_divergence = np.sum(div_kl)
  else:
    # Divergencia de densidad de potencia 
    term1 = pi_theta1**(1 + alpha) + pi_theta2**(1 + alpha)
    term2 = (1 + 1/alpha) * (p1 * pi_theta1**alpha + p2 * pi_theta2**alpha)
    div_alpha = K * (term1 - term2)
    total_divergence = np.sum(div_alpha)
    
  K_total = len(muestra) * K
  return total_divergence / K_total

# --- Estimador y Simulación ---

def emdp(theta_inicial, alpha, IT, s1, s2, K, muestra):
  """
  Encuentra el estimador de mínima divergencia de densidad de potencia (EMDP).
  """
  args = (alpha, IT, s1, s2, K, muestra)
  # Límites para los parámetros theta para ayudar al optimizador
  #bounds = [(4.5, 6.0),(-0.1, 0.0),(-1.5, 0.0),(0.0, 0.1)]
  result = opt.minimize(divergencia_weibull, theta_inicial, args=args,
                          method='Nelder-Mead')#'L-BFGS-B')#,bounds=bounds)
  
  if not result.success:
      print(f"ADVERTENCIA: La optimización falló para alpha={alpha} con el mensaje: {result.message}")
      
  return result.x

def simulacion(R, theta_0, theta_inicial, theta_cont, IT, s1, s2, K, alphas):
    """
    Realiza la simulación para estimar parámetros con y sin contaminación.
    
    Devuelve:
        Tres DataFrames de pandas con los resultados.
    """
    num_alphas = len(alphas)
    num_params = len(theta_0)
    
    # Pre-alojar arrays para guardar los resultados es más eficiente
    estimators_clean = np.zeros((num_alphas, R, num_params))
    estimators_cont = np.zeros((num_alphas, R, num_params))
    
    print("Iniciando simulación weibull...")
    for i, alpha in enumerate(alphas):
        print(f"Procesando alpha = {alpha} ({i+1}/{num_alphas})")
        for j in range(R):
            # 1. Muestra sin contaminar
            muestra_clean = gen_muestra_binomial_weibull(theta_0, IT, s1, s2, K, seed=j)
            estimators_clean[i, j, :] = emdp(theta_inicial, alpha, IT, s1, s2, K, muestra_clean)
            
            # 2. Muestra contaminada
            # Se genera una muestra con un theta diferente y se usa para contaminar
            # un punto de la muestra original.
            muestra_cont_source = gen_muestra_binomial_weibull(theta_cont, IT, s1, s2, K, seed=j)
            muestra_contaminated = np.copy(muestra_clean)
            muestra_contaminated[0] = muestra_cont_source[0] # Contaminamos el primer punto
            
            # CORRECCIÓN: Usar siempre theta_inicial como punto de partida, no theta_0
            estimators_cont[i, j, :] = emdp(theta_0, alpha, IT, s1, s2, K, muestra_contaminated)

    # --- Procesamiento de Resultados ---
    
    # Calcular medias de los estimadores para cada alpha
    media_estimators_clean = np.mean(estimators_clean, axis=1)
    media_estimators_cont = np.mean(estimators_cont, axis=1)
    
    # Calcular Error Cuadrático Medio (MSE) y RMSE
    # MSE = mean((estimador - valor_real)^2)
    se_clean = (estimators_clean - theta_0)**2
    se_cont = (estimators_cont - theta_0)**2
    
    # Tomamos la media sobre los parámetros y las R simulaciones
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

    # --- Guardar en CSV (IMPORTANTE: Cambia la ruta a una válida en tu PC) ---
    try:
        output_path = "resultados_weibull/" # Crea una carpeta para los resultados
        import os
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        df_estimators.to_csv(os.path.join(output_path, "estimators_clean.csv"), index=False)
        df_estimators_cont.to_csv(os.path.join(output_path, "estimators_cont.csv"), index=False)
        df_rmse.to_csv(os.path.join(output_path, "rmse.csv"), index=False)
        print(f"Archivos CSV guardados en la carpeta: '{output_path}'")
    except Exception as e:
        print(f"Error al guardar los archivos CSV: {e}")
        print("Asegúrate de que la ruta es correcta y tienes permisos de escritura.")

    return se_clean, se_cont,df_estimators, df_estimators_cont, df_rmse

# --- ANÁLISIS DE FIABILIDAD---


def calcular_fiabilidad(theta, IT, s1, s2):
    """
    Calcula la fiabilidad R(t) = 1 - F(t) de forma vectorial.
    Reutiliza la función 'probabilidad_weibull' (que calcula el CDF).
    
    Devuelve:
        Un array 1D con los valores de fiabilidad para cada tiempo en IT.
    """
    # probabilidad_weibull devuelve la prob. de fallo F(t) en una matriz
    prob_fallo_matrix = probabilidad_weibull(theta, IT, s1, s2)
    
    # Fiabilidad R(t) = 1 - F(t)
    fiabilidad_matrix = 1 - prob_fallo_matrix
    
    # Aplanamos para obtener un vector 1D, ya que s1 es un solo valor
    return fiabilidad_matrix.flatten()

def analizar_sesgo_fiabilidad(df_estimadores, theta_0, IT_test, s1_test, s2):
    """
    Calcula el sesgo de la fiabilidad para un conjunto de estimadores.
    
    Args:
        df_estimadores (DataFrame): DataFrame con los parámetros estimados (ej. df_est_clean).
        theta_0 (np.array): Parámetros verdaderos para calcular la fiabilidad real.
        IT_test (np.array): Tiempos de inspección para el análisis.
        s1_test (np.array): Nivel de estrés para el análisis.
        s2 (np.array): Segundo nivel de estrés (dummy en este caso).

    Returns:
        DataFrame con el sesgo de la fiabilidad para cada alpha.
    """
    # 1. Calcular la fiabilidad verdadera una sola vez
    fiabilidad_verdadera = calcular_fiabilidad(theta_0, IT_test, s1_test, s2)
    
    # 2. Extraer solo las columnas de parámetros del DataFrame
    num_params = len(theta_0)
    thetas_estimados = df_estimadores.iloc[:, 0:num_params]
    
    # 3. Aplicar la función de fiabilidad a cada fila de parámetros estimados
    # .apply con axis=1 es ideal para operaciones por fila.
    # result_type='expand' convierte la lista devuelta por cada fila en columnas del nuevo DataFrame.
    fiabilidades_estimadas = thetas_estimados.apply(
        lambda row: calcular_fiabilidad(row.values, IT_test, s1_test, s2),
        axis=1,
        result_type='expand'
    )
    
    # 4. Calcular el sesgo (Bias = Estimado - Verdadero) de forma vectorial
    # La resta se aplica a cada fila del DataFrame.
    df_sesgo = fiabilidades_estimadas.subtract(fiabilidad_verdadera, axis='columns')
    
    # 5. Formatear el DataFrame final
    df_sesgo.columns = [f"Bias_R(t={t})" for t in IT_test]
    df_sesgo["alpha"] = df_estimadores["alpha"].values
    
    # Reordenar para que 'alpha' sea la primera columna
    columnas_ordenadas = ["alpha"] + [col for col in df_sesgo.columns if col != "alpha"]
    return df_sesgo[columnas_ordenadas]

# --- DATOS PARA LA SIMULACIÓN Y EL ANÁLISIS ---

# Parámetros de simulación
R = 1000
K = 100
IT_weibull = np.array([8, 16, 24])
s1_weibull = np.array([30, 40, 50])
s2_weibull = np.array([0]) # s2 no se usa, pero la función lo espera
alphas = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])

# Parámetros del modelo
theta_0_weibull = np.array([5.3, -0.05, -0.6, 0.03])
theta_cont_weibull = np.array([5.3, -0.025, -0.6, 0.03])
theta_inicial_weibull = np.array([5.2, -0.06, -0.5, 0.04])

# --- EJECUCIÓN ---

# 1. Ejecutar la simulación principal
se_clean, se_cont,df_est_clean, df_est_cont, df_rmse = simulacion(
    R, theta_0_weibull, theta_inicial_weibull, theta_cont_weibull,
    IT_weibull, s1_weibull, s2_weibull, K, alphas
)

# 2. Realizar el análisis de fiabilidad
print("\n--- Iniciando Análisis de Fiabilidad ---")

# Parámetros para el test de fiabilidad
IT_test = np.array([25,30,35])
s1_test = np.array([40]) # El nivel de estrés de prueba, como array

# Analizar resultados de la muestra limpia
df_sesgo_clean = analizar_sesgo_fiabilidad(
    df_est_clean, theta_0_weibull, IT_test, s1_test, s2_weibull
)

# Analizar resultados de la muestra contaminada
df_sesgo_cont = analizar_sesgo_fiabilidad(
    df_est_cont, theta_0_weibull, IT_test, s1_test, s2_weibull
)

# Guardar los resultados del sesgo en archivos CSV
output_path = "resultados_weibull/"
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
