"""
@author: Nico
"""
import importlib
import file_classes
importlib.reload(file_classes)

inputs = {"data_type" : 'real',
          "variable_name" : '^STOXX50E',
          "degrees_freedom" : 9,
          "nb_sims" : 10**6}

manager = file_classes.manager(inputs) # initialise constructor
manager.load_timeseries() # get the timeseries
manager.compute() # compute returns and all different risk metrics
manager.plot_histogram()
print(manager)


# ================Some comments to understand the code=========================
# Implement the newly created class. It is much cleaner
# =============================================================================


