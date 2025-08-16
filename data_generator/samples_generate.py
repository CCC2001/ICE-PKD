import numpy as np
import matplotlib.pyplot as plt

def normal_clip(mean, cov, size): 
    """ Generate binary normally distributed data and clip to [0,1] """
    
    data = np.random.multivariate_normal(mean, cov, size)
    return np.clip(data, 0, 1)  

def samples_generate(N,save_fig_path=None,show=True):
    
    n = N // 5 #Number of each sampling set
    
    # Randomly selected means and variances
    means = [
        np.random.uniform(0.2, 0.8, 2),
        np.random.uniform(0.2, 0.8, 2),
        np.random.uniform(0.2, 0.8, 2),
        np.random.uniform(0.2, 0.8, 2)
    ]

    covariances = [
        np.eye(2)*np.random.uniform(0.01, 0.1, 2),
        np.eye(2)*np.random.uniform(0.01, 0.1, 2),
        np.eye(2)*np.random.uniform(0.01, 0.1, 2),
        np.eye(2)*np.random.uniform(0.01, 0.1, 2)
    ]

    # Generate train covariates
    x0 = normal_clip(means[0], covariances[0], n)
    x1 = normal_clip(means[1], covariances[1], n)
    x2 = normal_clip(means[2], covariances[2], n)
    x3 = normal_clip(means[3], covariances[3], n)
    x4 = np.random.rand(n, 2) 

    # Concat train covariates
    x = np.append(x0, x1, axis=0)
    x = np.append(x, x2, axis=0)
    x = np.append(x, x3, axis=0)
    x = np.append(x, x4, axis=0)
    
    # Generate test covariates
    x_test0 = normal_clip(means[0], covariances[0], n*10)
    x_test1 = normal_clip(means[1], covariances[1], n*10)
    x_test2 = normal_clip(means[2], covariances[2], n*10)
    x_test3 = normal_clip(means[3], covariances[3], n*10)
    x_test4 = np.random.rand(n*10, 2) 

    # Concat Test covariates
    x_test = np.append(x_test0, x_test1, axis=0)
    x_test = np.append(x_test, x_test2, axis=0)
    x_test = np.append(x_test, x_test3, axis=0)
    x_test = np.append(x_test, x_test4, axis=0)

    # Plot
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    labels = ['normal', 'normal', 'normal', 'normal', 'uniform[0,1]']

    for i, (data, mean, cov, color, label) in enumerate(zip([x0, x1, x2, x3], means, covariances, colors, labels)):
        plt.scatter(data[:, 0], data[:, 1], label=label+str(mean.round(2))+str([cov.round(2)[0][0],cov.round(2)[1][1]]), color=color)   
    plt.scatter(x4[:, 0], x4[:, 1],label=labels[-1],color=colors[-1])
    plt.legend()
    
    # Save fig
    if save_fig_path:
        plt.savefig(save_fig_path, dpi=300)
    
    # Show fit or not
    if show:
        plt.show()
    
    plt.close()
    
    return x,x_test