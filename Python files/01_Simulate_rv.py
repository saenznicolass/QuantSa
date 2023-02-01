"""
@author: Nico
"""

import numpy as np
import matplotlib.pyplot as plt

nb_sims = 10**6
df = 9 # degrees of freedon
dist_name = 'chi-square' # student normal exponential uniform chi-square

# Take a look at numpy documentation to see wich distributions are availiable 

if dist_name == 'normal':
    x = np.random.standard_normal(nb_sims)
    x_description = dist_name
elif dist_name == 'exponential':
    x = np.random.standard_exponential(nb_sims)
    x_description = dist_name
elif dist_name == 'uniform':
    x = np.random.uniform(0,1,nb_sims)
    x_description = dist_name
elif dist_name == 'student':
    x = np.random.standard_t(df=df, size=nb_sims)
    x_description = dist_name + ' | df = ' + str(df)
elif dist_name == 'chi-square':
    x = np.random.chisquare(df=df, size=nb_sims)
    x_description = dist_name + ' | df = ' + str(df)
    
# plot hist 
plt.figure()
plt.hist(x,bins=100)
plt.title(x_description)
plt.show()

