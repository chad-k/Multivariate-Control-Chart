
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:02:01 2025

@author: ckaln
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.multivariate.pca import PCA
from statsmodels.multivariate.manova import MANOVA
import seaborn as sns

def simulate_manufacturing_data(phase1_size=30, phase2_size=70):
    """Simulates manufacturing data with speed, temperature, and pressure.
    Args:
        phase1_size (int): The number of samples in Phase I data.
        phase2_size (int): The number of samples in Phase II data.
    Returns:
       pd.DataFrame: Combined DataFrame with Phase I and II data and process labels.
    """
    np.random.seed(42)  # for reproducibility

    # Phase I (In-Control data)
    mean_phase1 = [50, 150, 200]
    cov_phase1 = [[10, 5, 3], [5, 20, 8], [3, 8, 30]]  # covariance matrix
    phase1_data = np.random.multivariate_normal(mean_phase1, cov_phase1, size=phase1_size)
    phase1_df = pd.DataFrame(phase1_data, columns=['Speed', 'Temperature', 'Pressure'])
    phase1_df['Phase'] = 'Phase I'

    # Phase II (Introduce a shift in Temperature for 30 samples)
    mean_phase2 = [50, 150, 200]
    mean_phase2[1] = 165  #Shift in Temperature
    cov_phase2 = [[10, 5, 3], [5, 20, 8], [3, 8, 30]]  # covariance matrix
    phase2_data = np.random.multivariate_normal(mean_phase2, cov_phase2, size=phase2_size)
    phase2_df = pd.DataFrame(phase2_data, columns=['Speed', 'Temperature', 'Pressure'])
    phase2_df['Phase'] = 'Phase II'

    combined_df = pd.concat([phase1_df, phase2_df], ignore_index=True)
    return combined_df


def hotelling_t2_control_chart(data, variables, phase1_size=30, alpha=0.0027):
    """Creates a Hotelling's T-squared control chart.
      Args:
        data (pd.DataFrame): DataFrame containing the data.
        variables (list): List of column names to be used for T-squared calculations.
        phase1_size (int): Number of samples to use for calculating the control limits
                           (typically based on "in-control" data).
        alpha (float): The significance level for the control limits.

    Returns:
      dict: Dictionary containing central lines, upper and lower control limits and T-squared values
    """

    # 1. Separate Phase I (training) and Phase II (monitoring) data
    phase1_data = data.iloc[:phase1_size, :][variables]
    phase2_data = data.iloc[phase1_size:, :][variables]

    # 2. Calculate mean vector and covariance matrix from Phase I data
    mean_vector = phase1_data.mean().values
    covariance_matrix = phase1_data.cov().values
    covariance_matrix_inv = np.linalg.inv(covariance_matrix)


    # 3. Calculate T-squared statistic for Phase II data
    t2_values = []
    for index, row in phase2_data.iterrows():
       sample_vector = row.values
       deviation_vector = sample_vector - mean_vector
       t2_stat = np.dot(np.dot(deviation_vector.T, covariance_matrix_inv), deviation_vector)
       t2_values.append(t2_stat)


    # 4. Calculate control limits
    p = len(variables)  # number of variables
    n = phase1_size  # number of samples in Phase I
    f_value = stats.f.ppf(1 - alpha, p, n - p)
    upper_control_limit = ((p * (n - 1)) / (n - p)) * f_value
    lower_control_limit = 0.0  # Lower limit for T-squared is always 0


    control_limit_data = {
        't2_values': t2_values,
        'upper_control_limit': upper_control_limit,
        'lower_control_limit': lower_control_limit
    }
    return control_limit_data


def create_hotelling_plot(control_limit_data, phase1_size, plot_title = 'Hotelling\'s T-squared Control Chart'):
    """Creates Hotelling's T-squared plot
       Args:
        control_limit_data (dict): Data of T-squared values and control limits
        phase1_size (int): Number of samples in Phase I
        plot_title (str): Title of the control chart plot

    Returns:
      None (Displays the plot)
    """
    t2_values = control_limit_data['t2_values']
    upper_control_limit = control_limit_data['upper_control_limit']
    lower_control_limit = control_limit_data['lower_control_limit']

    plt.figure(figsize=(10, 6))
    plt.plot(range(phase1_size + 1, phase1_size + len(t2_values) + 1), t2_values, marker='o', linestyle='-', label='T-squared')
    plt.axhline(y=upper_control_limit, color='r', linestyle='--', label=f'UCL: {upper_control_limit:.2f}')
    plt.axhline(y=lower_control_limit, color='g', linestyle='--', label=f'LCL: {lower_control_limit:.2f}')
    plt.xlabel('Sample Number')
    plt.ylabel('Hotelling\'s T-squared Value')
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_contributions(data, control_limit_data, variables, phase1_size):
    """ Calculates variable contributions to the out-of-control T-squared values.
        Args:
            data (pd.DataFrame): DataFrame containing the data.
            control_limit_data(dict): Data of T-squared values and control limits
            variables (list): List of column names to be used for T-squared calculations.
            phase1_size (int): Number of samples in phase I
        Returns:
           pd.DataFrame: Contributions for each variable to each out-of-control sample.
        """
    # 1. Separate Phase I (training) and Phase II (monitoring) data
    phase1_data = data.iloc[:phase1_size, :][variables]
    phase2_data = data.iloc[phase1_size:, :][variables]

    # 2. Calculate mean vector and covariance matrix from Phase I data
    mean_vector = phase1_data.mean().values
    covariance_matrix = phase1_data.cov().values
    covariance_matrix_inv = np.linalg.inv(covariance_matrix)

    # 3. Get the T-Squared Values and UCL
    t2_values = control_limit_data['t2_values']
    upper_control_limit = control_limit_data['upper_control_limit']


    # 4. Calculate contributions for out-of-control points
    out_of_control_indices = [i for i, t2 in enumerate(t2_values) if t2 > upper_control_limit]

    if not out_of_control_indices:
       print("There are no out-of-control points. Contributions not calculated")
       return None

    contribution_list = []
    for index in out_of_control_indices:
        sample_vector = phase2_data.iloc[index].values
        deviation_vector = sample_vector - mean_vector

        # Calculate individual variable contributions
        contributions = []
        for i in range(len(variables)):
            # Create a vector where all other components have a 0 mean.
            deviation_vector_copy = deviation_vector.copy()
            for j in range(len(variables)):
                if j != i:
                   deviation_vector_copy[j] = 0
            t2_stat = np.dot(np.dot(deviation_vector_copy.T, covariance_matrix_inv), deviation_vector_copy)
            contributions.append(t2_stat)
        contribution_list.append(contributions)

    contribution_df = pd.DataFrame(contribution_list, columns=variables, index = out_of_control_indices)
    return contribution_df

def plot_contributions(contributions_df):
    """Plots variable contributions for out-of-control points.

     Args:
        contributions_df (pd.DataFrame): Contributions of each variable.

    Returns:
      None (Displays the contribution plots)
    """
    if contributions_df is None:
       return

    for index, row in contributions_df.iterrows():
      plt.figure(figsize=(8,6))
      plt.bar(row.index, row.values)
      plt.title(f'Contributions for out-of-control sample: {index}')
      plt.ylabel('Contribution to T-squared')
      plt.xlabel('Variable')
      plt.show()

def perform_pca(data, variables):
    """Performs PCA (Principal Component Analysis).

    Args:
        data (pd.DataFrame): DataFrame with data and variables to be used.
        variables (list): List of column names to perform PCA

    Returns:
       statsmodels.multivariate.pca.PCA: Result object containing PCA results.
    """

    pca = PCA(data[variables])
    return pca


def plot_pca_loadings(pca, variables):
    """Plots the PCA loadings for principal components.
       Args:
        pca (statsmodels.multivariate.pca.PCA): Result object from PCA calculation
        variables (list): List of column names to perform PCA
      Returns:
        None (Displays the plot)
    """
    loadings = pca.loadings
    num_components = loadings.shape[1]
    for i in range(num_components):
        plt.figure(figsize=(8, 6))
        plt.bar(variables, loadings[:, i])
        plt.title(f'Loadings of Principal Component {i + 1}')
        plt.xlabel('Variables')
        plt.ylabel('Loading')
        plt.show()

def perform_manova(data, variables, phase_variable='Phase'):
     """Performs MANOVA (Multivariate Analysis of Variance).
         Args:
            data (pd.DataFrame): DataFrame with data and variables to be used.
            variables (list): List of column names of dependant variables for MANOVA
            phase_variable (str): Name of the variable to be used as independant factor in MANOVA
         Returns:
           statsmodels.multivariate.manova.MANOVA: Object containing results of the MANOVA Analysis.
     """
     formula = f'{variables[0]} + {variables[1]} + {variables[2]} ~ C({phase_variable})'
     manova = MANOVA.from_formula(formula, data=data)
     return manova


# Main program execution
if __name__ == '__main__':
    # 1. Generate simulated data
    data = simulate_manufacturing_data()

    # 2. Define the variables we want to monitor
    variables_to_monitor = ['Speed', 'Temperature', 'Pressure']

    # 3. Perform Hotelling's T-squared analysis
    control_limit_data = hotelling_t2_control_chart(data, variables_to_monitor)
    create_hotelling_plot(control_limit_data, data[data['Phase'] == 'Phase I'].shape[0], plot_title = 'Hotelling\'s T-Squared Chart for Manufacturing Process')

    # 4. Calculate and plot contributions
    contributions_df = calculate_contributions(data, control_limit_data, variables_to_monitor, data[data['Phase'] == 'Phase I'].shape[0])
    plot_contributions(contributions_df)

    # 5. Perform PCA and plot loadings
    pca_result = perform_pca(data, variables_to_monitor)
    plot_pca_loadings(pca_result, variables_to_monitor)

    # 6. Perform MANOVA and print the result
    manova = perform_manova(data, variables_to_monitor)
    print("\nMANOVA Results:\n", manova.mv_test())
