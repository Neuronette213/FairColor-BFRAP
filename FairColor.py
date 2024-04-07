# -*- coding: utf-8 -*-

import math
import numpy as np
import time
from timeit import default_timer as timer
import os
from tabulate import tabulate
import matplotlib.pyplot as plt

"""# Paths to Folders for Data Importation and Saving Results"""

Data_folder = './Data_Instances'
Results_folder = './FairColor_Results'
Plots_folder = './FairColor_Plots'

os.makedirs(Results_folder, exist_ok=True)
os.makedirs(Plots_folder, exist_ok=True)

"""# Functions"""

def set_largest_values_to_1(input_matrix,t):
    # Initialize the binary matrix with zeros of the same shape as the input matrix
    binary_matrix = np.zeros_like(input_matrix, dtype=int)

    # Loop through each row in the input matrix
    for i in range(input_matrix.shape[0]):
        # Find the indices of the three largest values in the current row
        largest_indices = np.argsort(input_matrix[i])[-t:]
        # Choose a random element from the NumPy array
        #random_index = np.random.choice(largest_indices)


        # Set the corresponding elements in the binary matrix to 1
        binary_matrix[i, largest_indices] = 1

    return binary_matrix

#Getting the list of critical papers
def critical_papers(matrix,t):
    min_values = np.min(matrix, axis=1)
    sorted_values = np.sort(matrix, axis=1)[:, ::-1]
    theta_values = sorted_values[:, t-1]
    c_papers = np.where(theta_values == min_values)[0]

    return c_papers,theta_values

#Filtering the critical papers that are assigned to a reviwer r_k
def filter_critical_papers(c_papers, r_k, m, theta_values):
    valid_indices = np.where(m[c_papers, r_k] > theta_values[c_papers])[0]
    filtered_c_papers = c_papers[valid_indices]
    return filtered_c_papers

#Calculating the proportional fairness
def prop_fairness(pap_scores):
  fairness=np.sum(np.log(pap_scores))
  return fairness

#Displaying solutions found during the local search process
def display_selected_columns(data,headers,column_indices):
    # Selecting specific columns to display
    selected_columns = [[row[i] for i in column_indices] for row in data]
    # Displaying the selected columns as a table
    print(tabulate(selected_columns, headers))

#Ploting the matching scores and min papers scores for soultions obtained during the local search process
def plot_best_solutions(solutions, number_inst):
    # Extracting x, y1, and y2 values from the list of tuples
    x_values, y1_values, y2_values, a, b = zip(*solutions)

    # Creating the first plot with y1 values
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Minimum Paper Score', color=color)
    ax1.plot(x_values, y1_values, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Creating a second y-axis for y2 values
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Overall Matching Score', color=color)
    ax2.plot(x_values, y2_values, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Get the instance name corresponding to the instance number
    instance_name = instance_info[number_inst]

    # Display the plot with dynamic title
    plt.title(f'Best Solutions Plot - {instance_name}')

    # Save the plot as an image with a dynamic filename
    plot_file_name = f'best_solutions_plot_instance_{number_inst}.png'
    #plot_file_path = Plots_folder + result_file_name
    plot_filename = os.path.join(Plots_folder, f'best_solutions_plot_instance_{number_inst}.png')
    plt.savefig(plot_filename)

    # Close the plot to free memory
    plt.close()

    # Print a message indicating where the plot was saved
    #print(f"Plot saved as: {plot_filename}")

#Building assignment matrix containing only reviewers indices for each paper
def extract_reviewers_indices(binary_matrix,t):
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

#Pruning criteria strategy of Pareto set
def sort_criteria(x, criteria):
    """
    Define sorting criteria based on the specified parameter.

    Args:
    - x: A tuple representing an element to be sorted.
    - criteria (int): An integer specifying the sorting criteria:
        - 1: Sort by the minimum of paper score.
        - 2: Sort by the overall matching score.

    Returns:
    - Value based on the specified sorting criteria.
    """
    if criteria == 1:
        return x[0]
    elif criteria == 2:
        return x[1]
    else:
        raise ValueError("Invalid sorting criteria. Use 1 or 2.")
#Constructing the Pareto set and filtering the percentage according to selected criteria
def pareto_efficient_solutions(original_list, keep_percentage,pruning_criteria):
    """
    Identify Pareto-efficient solutions from a list of tuples and sort them by descending order of the first objective.

    Args:
    - original_list (list): A list of tuples where each tuple contains the values of the two objectives and the corresponding solution.

    Returns:
    - pareto_solutions (list): A list containing Pareto-efficient solutions sorted by descending order of the first objective.
    """
    # Identify Pareto-efficient solutions
    print('For decision Making, we provide a pruned Pareto set ')
    pareto_solutions = []
    for i, (obj1, obj2, solution) in enumerate(original_list):
        is_pareto_efficient = True
        for j, (other_obj1, other_obj2, _) in enumerate(original_list):
            if obj1 < other_obj1 and obj2 < other_obj2:
                is_pareto_efficient = False
                break
        if is_pareto_efficient:
            pareto_solutions.append((obj1, obj2, solution))

    # Sort Pareto-efficient solutions by descending order of the first objective
    sorted_pareto_solutions = sorted(pareto_solutions, key=lambda x: sort_criteria(x, pruning_criteria), reverse=True)

    # Calculate the number of elements to keep based on the percentage
    num_elements_to_keep = int(len(sorted_pareto_solutions) * (keep_percentage / 100))

    # Prune the list to keep only the specified percentage of elements
    pruned_pareto_solutions = sorted_pareto_solutions[:num_elements_to_keep]

    # Print pruned Pareto-efficient solutions
    print(f"Pruned Pareto-efficient solutions ({keep_percentage}%):")
    headers1=["Minimum paper score", "Matching score"]
    column_indices1 = [0, 1]
    display_selected_columns(pruned_pareto_solutions,headers1,column_indices1)

    return pruned_pareto_solutions

#Displaying Statistics about the obtained soultions
def statistics(solutions,upper_ms,upper_minps,running_time):
  print('Running time:',running_time)
  print('Upper bounds for Matching score:',upper_ms)
  print('Upper Bound for Minimum paper score:', upper_minps)
  min_paper_score_info = max(solutions, key=lambda x: x[1])
  max_matching_score_info = max(solutions, key=lambda x: x[2])
  stats_info=[max_matching_score_info,min_paper_score_info]
  print('information about the best solutions obtained in term of the efficiency and maxmin criterion, respectively')
  headers2=["Number iteration", "Minimum paper score", "Matching score", "Proportional Fairness", "Time to best"]
  column_indices = [0, 1, 2, 3, 4]
  display_selected_columns(stats_info,headers2,column_indices)

#Faircolor code
def Faircolor(num_inst, maxiter, keep_percentage,pruning_criteria):
  instance_file_name = f'instance_{num_inst}.npz'
  instance_file_path = os.path.join(Data_folder, instance_file_name)
  # Load the instance data
  instance_data = np.load(instance_file_path)
  #affinity score matrix and t
  matrix=instance_data['affinity_scores']
  t = int(instance_data['number_t'])
  #computation of n,m, upper and lower bounds l1 and l2
  n , m = matrix.shape[0], matrix.shape[1]
  l1 = math.floor(n * t / m)
  l2 = math.ceil(n * t / m)
  print(n,m,t,l1,l2)
  #Starting time
  start=timer()

  #Initialization and m-coloring construction
  assign_matrix = set_largest_values_to_1(matrix,t)
  init_assign_matrix=assign_matrix
  assign_score=matrix*assign_matrix
  paper_score=np.sum(assign_score,axis=1)
  init_paper_score=np.copy(paper_score)
  #Upper bounds for overall Matching Scores and the min of paper scores, respectively
  upper_ms=np.sum(assign_score)
  upper_minps=np.min(paper_score)

  #Find critical papers
  critic_papers,theta=critical_papers(matrix,t)
  class_size=np.sum(assign_matrix,axis=0)

  #Step 2: Make the m-coloring equitable
  upper=l2
  V_R=[]
  #mask = np.ones(len(class_size), dtype=bool)
  while len(V_R) != m:
    r_k=np.argmax(class_size)
    if class_size[r_k]>upper:
      over=class_size[r_k]-upper
      #search of all possible moves from one class r_k to other classes
      paper_moves_from_r_k=[]
      critic_papers_r_k=filter_critical_papers(critic_papers, r_k, matrix, theta)
      paper_indices = np.where((assign_matrix[:, r_k] == 1) & ~np.isin(np.arange(assign_matrix.shape[0]), critic_papers_r_k))[0]
      for i in paper_indices:
        other_reviewers= np.where((assign_matrix[i] == 0) & ~np.isin(np.arange(assign_matrix.shape[1]), V_R))[0]
        if len(other_reviewers)>0:
          j = other_reviewers[np.argmin(matrix[i, r_k] - matrix[i, other_reviewers])]
          best_move_paper=(i,j,matrix[i,r_k]-matrix[i,j])
          paper_moves_from_r_k.append(best_move_paper)
      best_moves = sorted(paper_moves_from_r_k, key=lambda x: (x[2],class_size[j]))
      #Balancing the color class r_k
      best_moves_from_r_k=best_moves[:over]
      if len(best_moves_from_r_k)>0:
        paper_vertices=[t[0] for t in best_moves_from_r_k]
        color_classes=[t[1] for t in best_moves_from_r_k]
        assign_matrix[paper_vertices, r_k] = 0
        assign_matrix[paper_vertices,color_classes]=1
        class_size[color_classes]=assign_matrix[:,color_classes].sum(axis=0)
        class_size[r_k]=0
        paper_score[paper_vertices]+=matrix[paper_vertices,color_classes]-matrix[paper_vertices,r_k]
    V_R.append(r_k)
    class_size[r_k]=0
    if len(V_R)==n*t-m*l1:
      upper=l1

  #Step~3: Local search to improve maxmin fairness
  #Best solutions list initialization
  best_solutions=[]
  assignments=[]
  best_assign_matrix=assign_matrix
  assignment = extract_reviewers_indices(best_assign_matrix,t)
  best_paper_score=np.sum(matrix*best_assign_matrix,axis=1)
  best_assign_score=matrix*best_assign_matrix
  end=timer()
  time_to_best=end-start
  fair=prop_fairness(best_paper_score)
  best_solutions.append((0,np.min(best_paper_score), np.sum(best_paper_score),fair,time_to_best))
  assignments.append((np.min(best_paper_score), np.sum(best_paper_score),assignment))

  #Sigma initialization and stepsize
  sigma=np.percentile(matrix,25,axis=1) #first quartile Q1
  stepsize=(np.max(init_paper_score)-np.min(sigma))/200
  #Local search Loop
  for a in range(1,maxiter): #100 for all instances
    #print('I am at iteration ',a)
    assign_score=matrix*assign_matrix
    paper_score=np.sum(assign_score,axis=1)
    #Saving the best solutions that improve maxmin
    if (np.min(paper_score) > np.min(best_paper_score)):
      best_assign_matrix=np.copy(assign_matrix)
      assignment = extract_reviewers_indices(best_assign_matrix,t)
      best_assign_score=matrix*best_assign_matrix
      best_paper_score=np.sum(best_assign_score,axis=1)
      #print(np.min(best_paper_score), np.sum(best_paper_score))
      #print('number of iterations:',a)
      time_B=timer()
      time_to_best=time_B-start
      fair=prop_fairness(best_paper_score)
      best_solutions.append((a,np.min(best_paper_score), np.sum(best_paper_score),fair,time_to_best))
      assignments.append((np.min(best_paper_score), np.sum(best_paper_score),assignment))
    #Detecting unfair papers wrt thresholds
    threshold_score=(a-1)*stepsize+sigma
    unfair_papers = np.where((paper_score<threshold_score) & (threshold_score<=init_paper_score))[0]
    #finding the best swaps for unfair papers
    if len(unfair_papers)>0:
      for paper in unfair_papers:
        swaps_paper=[]
        reviewers= np.where(assign_matrix[paper,:] == 1)[0]
        other_reviewers=np.where(assign_matrix[paper,:]==0)[0]
        for r_k in reviewers:
          for r_j in other_reviewers:
            alpha=paper_score[paper]+matrix[paper,r_j] - matrix[paper,r_k]
            if alpha>=threshold_score[paper]:
              paper_indices=np.where((assign_matrix[:,r_j]==1) & (assign_matrix[:,r_k]==0)
              & (paper_score+matrix[:,r_k]-matrix[:,r_j]>=threshold_score))[0]
              if len(paper_indices)>0:
                p=paper_indices[np.argmax(paper_score[paper_indices]-matrix[paper_indices,r_j]+matrix[paper_indices,r_k])]
                swap_move=(r_k,p,r_j,alpha)
                swaps_paper.append(swap_move)
        if len(swaps_paper)>0:
          best_swap_paper = max(swaps_paper, key=lambda x: x[3])
          r_k=best_swap_paper[0]
          r_j=best_swap_paper[2]
          p=best_swap_paper[1]
          assign_matrix[paper,r_j]=1
          assign_matrix[paper,r_k]=0
          assign_matrix[p,r_j]=0
          assign_matrix[p,r_k]=1

  # Computing the running time
  end = timer()
  running_time=end-start
  #Displaying statitics about the solutions found during the local search process
  statistics(best_solutions,upper_ms,upper_minps,running_time)
  #Ploting and saving overall matching scores and minimum paper scores for solutions found during the local search process
  plot_best_solutions(best_solutions,num_inst)

  # Getting percentage% of best solutions wrt pruning criteria from pareto set
  pruned_pareto_set=pareto_efficient_solutions(assignments,keep_percentage,pruning_criteria)

  # Saving Results, each assignment from the pruned Pareto set individually to the results folder
  save_solutions(pruned_pareto_set,num_inst,running_time)

#Saving assignments individually.
def save_solutions(pruned_pareto_set, instance_number,run_time):
    """
    Save each solution in the pruned Pareto set as an NPZ file.

    Args:
    - pruned_pareto_set (list of tuples): List of tuples containing solutions.
    - instance_number (int): Number of the instance.
    """
    # Iterate over each solution in the pruned Pareto set
    for idx, (_, _, assignment) in enumerate(pruned_pareto_set):
        # Create a unique filename for each solution
        filename = os.path.join(Results_folder, f"instance_{instance_number}_{idx}_FairColor_result.npz")
        
        # Save the assignment and running time as an NPZ file
        np.savez(filename, assignment=assignment, run_time=run_time)

"""# Main Code"""

if __name__ == '__main__':
    #Select an instance
    #80=ICLR'18, 70=CVPR'18 extended (more than 10000 papers), 60=ICA2IT'19,50=CVPR'18, 40=CVPR'17, 30=MIDL
    instance_info = {30: "Midl'18", 40: "CVPR'17", 50: "CVPR'18", 60: "ICA2IT'19", 70: "CVPR'18Extd", 80: "ICLR'18"}
    num_inst=30 #Number of instance
    maxiter=100 #Number of iterations during the local search process
    keep_percentage=70 #This is the percentage of solutions that are kept from the pareto set sorted wrt the pruning criteria, such that
    '''
      * pruning criteria=1 ---> the pruning is done on the pareto set solutions when sorted in descending
        order according to the minimum of papers' scores.
      * pruning_criteria=2 ---> the pruning is done on the pareto set solutions when sorted in descending
        order according to the overall matching scores.
    '''
    pruning_criteria=1
    Faircolor(num_inst, maxiter, keep_percentage,pruning_criteria)
