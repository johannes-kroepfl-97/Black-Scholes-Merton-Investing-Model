import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def rw1D(steps, p=0.5):
    """
    This function creates a 1-dimensional random walk of a given number of steps.
    
    Parameters:
    steps (int): The number of steps in the random walk.
    p (float): The probability of moving up (+1).

    Returns:
    numpy.ndarray: An array of tuples, where each tuple is a step in the random walk.
    """
    
    # Initialize the walk
    M = [(0,0)]
    # Generate each step of the walk
    for k in range(1, steps):
        # Decide the direction of the next step
        X_k = 1 if np.random.uniform(0,1) < p else -1
        # Add the new position to the walk
        M.append((k, M[k-1][1]+X_k))
    # Convert the walk to a numpy array and return it
    return np.array(M)

def plot_rw1D(steps, p=0.5, col='b-'):
    """
    This function generates and plots a 1-dimensional random walk.

    Parameters:
    steps (int): The number of steps in the random walk.
    p (float): The probability of moving up (+1).
    col (str): The color of the plot line.
    """
    # Generate the walk
    M = rw1D(steps)
    # Create the plot
    plt.figure(figsize=(9,6))
    plt.plot(M[:,0], M[:,1], col)
    plt.xlabel('k')
    plt.ylabel('Mk')
    plt.show()

def ssrw1D(n, T, p=0.5):
    """
    This function scales a random walk to simulate Brownian motion, a continuous-time stochastic process.
    
    Parameters:
    n (int): The number of steps in the random walk.
    T (int): The time period for the Brownian motion.
    p (float): The probability of moving up (+1) in the random walk.
    
    Returns:
    numpy.ndarray: An array of tuples, where each tuple is a step in the scaled random walk.
    """
    # Generate the random walk
    M = rw1D(n*T, p=0.5)
    # Scale the walk
    W = np.array(list(zip((1/n) * M[:,0], float(1 / np.sqrt(n)) * M[:,1])))
    return W

def plot_bm(num_tries=10000, n=1000, T=2, figsize=(12,6)):
    """
    This function generates multiple realizations of a Brownian motion process, 
    and plots their distribution at time T.

    Parameters:
    num_tries (int): The number of realizations to generate.
    n (int): The number of steps in each random walk.
    T (int): The time period for the Brownian motion.
    figsize (tuple): The size of the figure to plot.
    """
    fig, axs = plt.subplots(1,2, figsize=figsize)
    WT_list = list()
    for _ in range(num_tries):
        col = np.random.choice(['b-', '-g', '-r', '-c', '-m', '-y'])
        bm_1 = ssrw1D(n=n, T=T)
        WT_list.append(float(bm_1[-1][1]))
        axs[0].plot(bm_1[:,0],bm_1[:,1],col)
    
    axs[0].title.set_text(f'{n} SSRWs with T={T}, n={n}')
    
    mu = np.mean(WT_list)
    var = np.var(WT_list)
    sigma = np.sqrt(var)
    
    print(f'Distribution of paths at T={T}:')
    print(f'  mu: {round(mu, 3)}')
    print(f'  var: {round(var, 3)}\n')
    
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100000)
    axs[1].plot(x, stats.norm.pdf(x, mu, sigma))
    axs[1].hist(WT_list, density=True, bins=30)  
    axs[1].title.set_text(f'Distribution of paths at T={T}')
    axs[1].set_ylabel('Probability')
    axs[1].set_xlabel('Data')
    
    plt.show()

def plot_function(f, x=np.linspace(0,3,100)):
    """
    This function plots a given function f.

    Parameters:
    f (function): The function to plot.
    x (numpy.ndarray): The x values to use in the plot.
    """
    # the function, which is y = x^2 here
    y = f(x)

    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # plot the function
    plt.plot(x, y, 'b')
    
    plt.show()

def plot_indicator_for_bm(n=1000, T=100, k=5):
    """
    This function generates a realization of a Brownian motion process,
    and overlays it with a stepwise constant indicator function. 

    Parameters:
    n (int): The number of steps in the random walk.
    T (int): The time period for the Brownian motion.
    k (int): The frequency of the steps in the stepwise function.
    """
    bm= ssrw1D(n, T, p=0.5)
    x = bm[:,0].copy().tolist()
    y = bm[:,1].copy().tolist()
    
    #plot the BM
    fig = plt.figure(figsize=(16,10))
    plt.plot(x,y)

    indicator_f_x = [x[0]]
    indicator_f_y = [y[0]]

    for i, yi in enumerate(y):
        if i % (k*n) == 0:
            # plot the straight line as indicator and then reset
            plt.plot(indicator_f_x, indicator_f_y, 'r')
            # make closed point at beginning of line
            plt.plot(indicator_f_x[0], indicator_f_y[0], markersize=5,
                     marker='o', markeredgecolor='red', markerfacecolor='red')
            # make open point at end of line
            plt.plot(indicator_f_x[-1], indicator_f_y[-1], markersize=5,
                     marker='o', markeredgecolor='red', markerfacecolor='white')
            indicator_f_x = [x[i]]
            indicator_f_y = [y[i]]
        else:
            indicator_f_x.append(x[i])
            indicator_f_y.append(indicator_f_y[-1])

    plt.plot(indicator_f_x, indicator_f_y, 'r')
    plt.plot(indicator_f_x[0], indicator_f_y[0], markersize=5,
                     marker='o', markeredgecolor='red', markerfacecolor='red')
    plt.plot(indicator_f_x[-1], indicator_f_y[-1], markersize=5,
             marker='o', markeredgecolor='red', markerfacecolor='white')
    
    plt.xlabel('t')
    plt.ylabel('W(t) and  ∆t')
    plt.show()