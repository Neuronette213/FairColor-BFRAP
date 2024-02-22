# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from scipy.stats import skew
import os
from tabulate import tabulate

# Path to the source and destination folders
Data_folder = './Data_Instances'
Faircolor_results_folder = './FairColor_Results'
fairflow_results_folder = './FairFlow_Results'
fairIR_results_folder = './FairIR_Results'

path_results = {
    'FairFlow': fairflow_results_folder,
    'FairIR': fairIR_results_folder,
}

#Calculating the proportional fairness
def prop_fairness(pap_scores):
  fairness = np.sum(np.log(pap_scores))
  return fairness

def extract_reviewers_indices(binary_matrix, t):
    # Get the shape of the input matrix
    n, m = binary_matrix.shape

    # Initialize an empty output matrix
    assignment_matrix = np.zeros((n, t), dtype=int)

    # Iterate over each row of the binary matrix
    for i in range(n):
        # Find the indices of non-zero elements in the current row
        nonzero_indices = np.where(binary_matrix[i] != 0)[0]

        # If the number of non-zero elements is not equal to t, raise an error
        if len(nonzero_indices) != t:
            raise ValueError(f"Row {i+1} does not have exactly {t} non-zero elements.")

        # Store the non-zero indices in the output matrix
        assignment_matrix[i] = nonzero_indices

    return assignment_matrix

def calculate_metrics(mat_scores, mat_assignments):
    """
    Calculate the paper scores based on the assigned reviewers and return the minimum score and the sum of paper scores.

    Args:
    - mat_scores (ndarray): Array of shape (n, m) containing scores for each reviewer-paper pair.
    - mat_assignments (ndarray): Array of shape (n, t) containing indices of assigned reviewers for each paper.

    Returns:
    - Tuple (float, float): Minimum of the array of paper scores and the sum of elements of paper scores.
    """
    # Calculate the paper scores based on the assigned reviewers
    paper_scores = np.sum(mat_scores[np.arange(len(mat_scores))[:, None], mat_assignments], axis=1)

    # Calculate the sum of paper scores
    sum_paper_scores = np.sum(paper_scores)

    # Find the minimum of the array of paper scores
    min_score = np.min(paper_scores)

    #Calculate the proportional fairness
    prop = prop_fairness(paper_scores)

    return min_score, sum_paper_scores, prop

def balanced(mat_assignments):
    """
    Count the number of papers assigned to each reviewer and return the difference between the maximum and minimum counts.

    Args:
    - mat_assignments (ndarray): Array of shape (n, t) containing indices of assigned reviewers for each paper.

    Returns:
    - Tuple (ndarray, int): Array of shape (m,) containing the number of papers assigned to each reviewer, and the difference
                            between the maximum and minimum counts.
    """
    # Get the maximum index (number of reviewers)
    max_index = np.max(mat_assignments) + 1

    # Count the occurrences of each index along the rows
    counts = np.bincount(mat_assignments.flatten(), minlength=max_index)

    # Calculate the difference between the maximum and minimum counts
    difference = np.max(counts) - np.min(counts)

    return difference

import glob
def get_files(instance_number):
    # Define the file pattern
    file_pattern = os.path.join(Faircolor_results_folder, f"instance_{instance_number}_*_FairColor_result.npz")

    # Use glob to find matching files
    matching_files = glob.glob(file_pattern)

    return matching_files

def display_results(data):
    headers = ['Assignment', 'MS', 'Min PS', 'PF', 'Balance','RT' ]
    print(tabulate(data, headers))

import re
def results_faircolor(instance_number):
    all_solutions = []
    files = get_files(instance_number)
    for instance_file_path in files:
        instance_file_name = os.path.basename(instance_file_path)
        idx = int(re.match(f'instance_\d+_(?P<idx>\d+)_FairColor_result\.npz', instance_file_name).group('idx'))
        faircolor_results = np.load(instance_file_path,allow_pickle=True)
        faircolor_assignment = faircolor_results['assignment']
        faircolor_time = faircolor_results['time']
        ps,ms,prop=calculate_metrics(mat_scores, faircolor_assignment)
        balance=balanced(faircolor_assignment)
        ass=f'FairColor_Assignment_{idx}'
        all_solutions.append([ass,ms,ps,prop,balance,faircolor_time])
    return all_solutions

def alg_results(instance_number, algname):
    instance_file_name = f'instance_{instance_number}_{algname}_result.npz'
    instance_file_path = os.path.join(path_results[algname], instance_file_name)
    results = np.load(instance_file_path)
    matrix_assignment = results['matrix_assign']
    run_time = results['time']
    alg_assignment=extract_reviewers_indices(matrix_assignment,t)
    ps,ms,prop=calculate_metrics(mat_scores, alg_assignment)
    balance=balanced(alg_assignment)
    ass=f'{algname}_Assignment'
    solution=[[ass,ms,ps,prop,balance,run_time]]
    return solution

"""# Results"""

#Select an instance
#80=ICLR'18, 70=CVPR'18 extended (more than 10000 papers), 60=ICA2IT'19,50=CVPR'18, 40=CVPR'17, 30=MIDL
num_inst = 60
#Load data instance
instance_file_name = f'instance_{num_inst}.npz'
instance_file_path = os.path.join(Data_folder, instance_file_name)
# Load the instance data
instance_data = np.load(instance_file_path)
#Matching score matrix and t
mat_scores=instance_data['affinity_scores']
t = int(instance_data['number_t'])
n,m = mat_scores.shape[0],mat_scores.shape[1]

#Load FairColor's results
print(f'FairColor Results for Instance_{num_inst}')
print("-" * 100)
faircolor_solutions = results_faircolor(num_inst)
display_results(faircolor_solutions)

for line in range(3):
  print("-" * 100)
#Load Fairflow's results
print(f'FairFlow Results for Instance_{num_inst}')
print("-" * 100)
algname = 'FairFlow'
fairflow_solution = alg_results(num_inst, algname)
display_results(fairflow_solution)

for line in range(3):
  print("-" * 100)
#Load FairIR's results
print(f'FairIR Results for Instance_{num_inst}')
print("-" * 100)
algname = 'FairIR'
fairIR_solution = alg_results(num_inst, algname)
display_results(fairIR_solution)
