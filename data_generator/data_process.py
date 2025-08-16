"""
Module for generating synthetic treatment response data with drift simulation
and visualization. Includes functions for probability modeling, data sampling,
and 3D visualization of response surfaces.
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Compute the sigmoid activation function.
    
    Args:
        x (ndarray): Input array
        
    Returns:
        ndarray: Sigmoid-transformed values in range (0, 1)
    """
    return 1 / (1 + np.exp(-x))

def P_func0(X, Y, d1):
    """
    Compute response probability surface for treatment 0 with concept drift parameter.
    
    Args:
        X (ndarray): Feature matrix dimension 1
        Y (ndarray): Feature matrix dimension 2
        d1 (float): Drift parameter controlling distribution shift
        
    Returns:
        ndarray: Probability matrix with Gaussian noise (μ=0, σ=0.0001)
    """
    noise = np.random.normal(0, 0.0001)
    return sigmoid(0.6*X**2 + X*Y**2 -(0.5-d1)*X -(0.6-0.5*d1)*Y + d1 * 0.2 + noise)

def P_func1(X, Y, d2):
    """
    Compute response probability surface for treatment 1 with concept drift parameter.
    
    Args:
        X (ndarray): Feature matrix dimension 1
        Y (ndarray): Feature matrix dimension 2
        d2 (float): Drift parameter controlling distribution shift
        
    Returns:
        ndarray: Probability matrix with Gaussian noise (μ=0, σ=0.0001)
    """
    noise = np.random.normal(0, 0.0001)
    return sigmoid(0.7*X**2 + 0.1*Y**2 + X*Y**2 -(0.5-d2)*X - (0.5-0.5*d2)*Y + d2 * 0.2 + noise)

def P_func2(X, Y, d3):
    """
    Compute response probability surface for treatment 2 with concept drift parameter.
    
    Args:
        X (ndarray): Feature matrix dimension 1
        Y (ndarray): Feature matrix dimension 2
        d3 (float): Drift parameter controlling distribution shift
        
    Returns:
        ndarray: Probability matrix with Gaussian noise (μ=0, σ=0.0001)
    """
    noise = np.random.normal(0, 0.0001)
    return sigmoid(0.9*X**2 + 0.15*Y**2 + X*Y**2 -(0.5-d3)*X - (0.5-0.5*d3)*Y + d3 * 0.2 + noise)

def exact_curve_plot(drift0, drift1, drift2, save_fig_path=None, show=True):
    """
    Generate 3D visualization of treatment response probability surfaces.
    
    Args:
        drift0 (float): Concept drift parameter for treatment 0
        drift1 (float): Concept drift parameter for treatment 1
        drift2 (float): Concept drift parameter for treatment 2
        save_fig_path (str, optional): Path to save output figure
        show (bool, optional): Whether to display plot interactively
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    
    # Create coordinate grid
    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 1, 101)
    X, Y = np.meshgrid(x, y)
    
    # Calculate response surfaces
    P0 = P_func0(X, Y, drift0)
    P1 = P_func1(X, Y, drift1)
    P2 = P_func2(X, Y, drift2)
    
    # Plot surfaces with stylistic settings
    ax1.plot_surface(X, Y, P0, edgecolor='black', color='blue', 
                    lw=0.1, rstride=5, cstride=5, alpha=0.4)
    ax1.plot_surface(X, Y, P1, edgecolor='black', color='red', 
                    lw=0.1, rstride=5, cstride=5, alpha=0.4)
    ax1.plot_surface(X, Y, P2, edgecolor='black', color='lightgreen', 
                    lw=0.1, rstride=5, cstride=5, alpha=0.4)
    
    plt.title(f"$d = [{drift0:.2f}, {drift1:.2f}, {drift2:.2f}]$")
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_zlabel('$p$')
    
    # Output handling
    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300)
    if show:
        plt.show()
    plt.close()
    
def choose_treatment(X1, X2):
    """
    Assign treatments using binomial selection with region-based bias.
    
    Implements a two-binomial experimental design introducing selection bias
    based on feature space partitioning.
    
    Args:
        X1 (ndarray): Feature vector dimension 1
        X2 (ndarray): Feature vector dimension 2
        
    Returns:
        ndarray: Treatment assignments (0, 1, or 2) with shape (n, 1)
    """
    n = len(X1)
    T = np.zeros((n, 1))
    
    for i in range(len(X1)):
        # Generate binomial random variables for selection
        p1 = np.random.binomial(1, 0.6)
        p2 = np.random.binomial(1, 0.5)

        # Region-based treatment assignment logic
        if X1[i] + X2[i] < 0.8:
            if p1 == 1: T[i] = 0
            elif p2 == 1: T[i] = 1
            else: T[i] = 2   
                      
        elif X1[i] + X2[i] > 1.2:
            if p1 == 1: T[i] = 1
            elif p2 == 1: T[i] = 0
            else: T[i] = 2  
      
        else:
            if p1 == 1: T[i] = 2
            elif p2 == 1: T[i] = 0
            else: T[i] = 1
            
    return T

def train_data_process(inputs, drift0, drift1, drift2):
    """
    Generate training dataset with selection bias.
    
    Args:
        inputs (ndarray): Input features (n_samples, 2)
        drift0 (float): Concept drift parameter for treatment 0
        drift1 (float): Concept drift parameter for treatment 1
        drift2 (float): Concept drift parameter for treatment 2
        
    Returns:
        tuple: 
            X (ndarray): Combined features, treatments, and outcomes
            P (ndarray): True probability matrices
            Y (ndarray): Potential outcomes for all treatments
    """
    X1 = inputs[:, 0:1]
    X2 = inputs[:, 1:2]
    
    # Calculate true probabilities
    P0 = P_func0(X1, X2, drift0)
    P1 = P_func1(X1, X2, drift1)
    P2 = P_func2(X1, X2, drift2)
    P = np.hstack((P0, P1, P2))
    
    # Generate binary outcomes
    Y0 = np.random.binomial(1, P0)
    Y1 = np.random.binomial(1, P1)
    Y2 = np.random.binomial(1, P2)
    Y = np.hstack((Y0, Y1, Y2))
    
    # Assign treatments with bias
    T = choose_treatment(X1, X2)
    
    # Select observed outcomes based on treatment
    Y_selected = np.select([T == 0, T == 1, T == 2], [Y0, Y1, Y2])

    # Combine features, treatments, and outcomes
    X = np.hstack((X1, X2, T, Y_selected.reshape(-1, 1)))
    
    return X, P, Y

def test_data_process(inputs, drift0, drift1, drift2):
    """
    Generate test dataset with random treatment assignment.
    
    Test points follow an identical distribution to the train points.
    
    Args:
        inputs (ndarray): Input features (n_samples, 2)
        drift0 (float): Drift parameter for treatment 0
        drift1 (float): Drift parameter for treatment 1
        drift2 (float): Drift parameter for treatment 2
        
    Returns:
        tuple: 
            X (ndarray): Combined features, treatments, and outcomes
            P (ndarray): True probability matrices
            Y (ndarray): Potential outcomes for all treatments
    """
    X1 = inputs[:, 0:1]
    X2 = inputs[:, 1:2]
    
    # Calculate true probabilities
    P0 = P_func0(X1, X2, drift0)
    P1 = P_func1(X1, X2, drift1)
    P2 = P_func2(X1, X2, drift2)
    P = np.hstack((P0, P1, P2))

    # Generate binary outcomes
    Y0 = np.random.binomial(1, P0)
    Y1 = np.random.binomial(1, P1)
    Y2 = np.random.binomial(1, P2)
    Y = np.hstack((Y0, Y1, Y2))

    # Assign treatments randomly
    T = np.random.choice([0, 1, 2], size=(len(Y), 1))

    # Select observed outcomes based on treatment
    Y_selected = np.select([T == 0, T == 1, T == 2], [Y0, Y1, Y2])

    # Combine features, treatments, and outcomes
    X = np.hstack((X1, X2, T, Y_selected.reshape(-1, 1)))
    
    return X, P, Y

def test_data_process_grid(drift0, drift1, drift2):
    """
    Generate test dataset with grid-sampled points.
    
    Test points are arranged as grid nodes spanning the sample space.
    
    Args:
        drift0 (float): Drift parameter for treatment 0
        drift1 (float): Drift parameter for treatment 1
        drift2 (float): Drift parameter for treatment 2
        
    Returns:
        tuple: 
            X (ndarray): Combined features, treatments, and outcomes
            P (ndarray): True probability matrices
            Y (ndarray): Potential outcomes for all treatments
    """
    # Create dense grid coordinates
    x1 = np.linspace(0, 1, 101)
    x2 = np.linspace(0, 1, 101)
    X1, X2 = np.meshgrid(x1, x2)
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)
    n = len(X1)

    # Replicate grid for multiple samples
    num_samples = 5
    X1 = np.tile(X1, (num_samples, 1))
    X2 = np.tile(X2, (num_samples, 1))
    
    # Calculate true probabilities
    P0 = P_func0(X1, X2, drift0)
    P1 = P_func1(X1, X2, drift1)
    P2 = P_func2(X1, X2, drift2)
    P = np.hstack((P0, P1, P2))

    # Generate binary outcomes
    Y0 = np.random.binomial(1, P0)
    Y1 = np.random.binomial(1, P1)
    Y2 = np.random.binomial(1, P2)
    Y = np.hstack((Y0, Y1, Y2))

    # Assign treatments randomly
    T = np.random.choice([0, 1, 2], size=(n * num_samples, 1))

    # Select observed outcomes based on treatment
    Y_selected = np.select([T == 0, T == 1, T == 2], [Y0, Y1, Y2])

    # Combine features, treatments, and outcomes
    X = np.hstack((X1, X2, T, Y_selected.reshape(-1, 1)))
    
    return X, P, Y