"""
@author: Nico
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2


nb_sims = 10**6
df = 9 # degrees of freedom for student an chi-square
dist_name = 'chi-square' # student normal exponential uniform chi-square
dist_type = 'simulated RV' # real or custom


if dist_name == 'normal':
    x = np.random.standard_normal(nb_sims)
    x_description = dist_type + ' ' + dist_name
elif dist_name == 'exponential':
    x = np.random.standard_exponential(nb_sims)
    x_description = dist_type + ' ' + dist_name
elif dist_name == 'uniform':
    x = np.random.uniform(0,1,nb_sims)
    x_description = dist_type + ' ' + dist_name
elif dist_name == 'student':
    x = np.random.standard_t(df=df, size=nb_sims)
    x_description = dist_type + ' ' + dist_name + ' | df = ' + str(df)
elif dist_name == 'chi-square':
    x = np.random.chisquare(df=df, size=nb_sims)
    x_description = dist_type + ' ' + dist_name + ' | df = ' + str(df)
    
'''
Let's create a Jarque-Bera normality test
'''

x_mean = np.mean(x)
x_std = np.std(x)
x_skew = skew(x)
x_kurtosis = kurtosis(x) # 4th moment. Excess kurtosis. Tales
x_jb = nb_sims/6*(x_skew**2 + 1/4*x_kurtosis**2)
x_p_value = 1 - chi2.cdf(x_jb, df=2) # normal is p>0.05. 2 to be distributed as chi-square
x_is_normal = (x_p_value > 0.05) # equivalently jb < 6

print(x_description)
print('mean is ' + str(x_mean))
print('standard deviation is ' + str(x_std))
print('skewness is ' + str(x_skew))
print('kurtosis is ' + str(x_kurtosis))
print('JB statistic is ' + str(x_jb))
print('p-value ' + str(x_p_value))
print('is normal ' + str(x_is_normal))

# plot histogram
plt.figure()
plt.hist(x,bins=100)
plt.title(x_description)
plt.show()



# ================Some comments to understand the code=========================
# Test used to know whether sample data have the skewness
# and kurosis matching a normal distr.
# In other words: how far are you from normality 
#
# A rnv are symmetric and hence has odd moments equal to 0
#
# Kurtosis: used to describe the size of the tails on a distribution. 
#
# Excess kurtosis: helps determine how much risk is involved in a specific investment
#
# Excess kurtosis = Kurtosis - 3
#
# Skewness: degree of asymmetry observed in a probability distribution.
# 
# Null hyp: Skew and ExcKur equals to 0 
#
# P-alue: find a num such that if <p is normal and if >p not normal
# Probability of obtaining test results (JB stat) at least as extreme as the result 
# actually observed under the assumption that the null hypothesis is correct.
# Small p => such an extreme observed outcome (JB) would be very unlikely under the 
# null hypothesis.
# Does not prove normality, rather it attempts to reject normality
#
# =============================================================================

