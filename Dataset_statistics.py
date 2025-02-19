# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd
from scipy.stats import skew
from tabulate import tabulate

# Path to the source and destination folders
Data_folder = './Data_Instances'

#Displaying statistics as a table
def display_statistics(data, headers):
    # Displaying the selected columns as a table
    print(tabulate(data, headers))

list_of_instances = [30, 40, 50, 60, 70, 80]
instance_statistics = []
for num_inst in list_of_instances:
  instance_file_name = f'instance_{num_inst}.npz'
  instance_file_path = os.path.join(Data_folder, instance_file_name)
  # Load the instance data
  instance_data = np.load(instance_file_path)
  #affinity score matrix and t
  matrix=instance_data['affinity_scores']
  t = int(instance_data['number_t'])
  n,m=matrix.shape[0],matrix.shape[1]
  flattened_matrix=matrix.flatten()
  matrix_mean = np.mean(flattened_matrix)
  matrix_median = np.median(flattened_matrix)
  matrix_std = np.std(flattened_matrix)
  matrix_min = np.min(flattened_matrix)
  matrix_max = np.max(flattened_matrix)
  matrix_skewness = skew(flattened_matrix)
  sparsity_index = np.sum(matrix == 0) / matrix.size
  instance_number=f'Instance_{num_inst}'
  Data_stats=[instance_number,n,m,t,matrix_mean,matrix_median,matrix_std,matrix_min,matrix_max,matrix_skewness,sparsity_index]
  instance_statistics.append(Data_stats)

headers = ['Instance', 'n', 'm', 't', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Skewness', 'Sparsity Index']

display_statistics(instance_statistics, headers)
