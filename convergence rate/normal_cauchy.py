import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from numpy import array

# Graficando histograma
#mu, sigma = 0, 0.2 # media y desvio estandar
#datos = np.random.normal(mu, sigma, 1000) #creando muestra de datos
n, p = 30, 0.4
# histograma de distribución normal.
#cuenta, cajas, ignorar = plt.hist(datos, 20)
#plt.ylabel('frequencia')
#plt.xlabel('valores')
#plt.title('Histograma')
#plt.show()

# Graficando Poisson
x = np.linspace(stats.cauchy.ppf(0.01),
               stats.cauchy.ppf(0.99), 100)
x_1 = np.linspace(stats.norm.ppf(0.01),
                  stats.norm.ppf(0.99), 100)

fda_cauchy = stats.cauchy.pdf(x, 2.75, 2.75) # Función de Distribución Acumulada
fda_normal = stats.norm.pdf(x_1,0.5, 2) # Función de Distribución Acumulada
fda_prod = fda_cauchy * fda_normal 
plt.plot(x, fda_cauchy, '--', label='FDA Cauchy')
plt.plot(x, fda_normal, label='FDA normal')
plt.plot(x, fda_prod, label='FDA Product Normal - Cauchy ')
plt.title('Función de Distribución Acumulada')
plt.ylabel('probabilidad')
plt.xlabel('valores')
plt.legend(loc=4)
plt.show()