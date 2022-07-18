# correlation consistent plots
import numpy as np
import matplotlib.pyplot as plt

def extract_steps(opt_results, key="fun"):
    """Get the given key value for each step
       in an opt_results dictionary
    """
    nsteps = len(opt_results)
    steps, values = [], []
    for k, d in opt_results.items():
        steps.append(int(k[9:]))
        values.append(d.get(key, 0.))
    return steps, np.array(values)
    
def plot_objective(basis, figsize=(9, 9),
                   x_transform=lambda x: x,
                   y_transform=lambda y: y):
    """ Create a matplotlib figure of the objective function value
        at each step of an optimization, separated by atom type if 
        multiple atoms given.
    
        Arguments:
            basis (Basis): basis object with opt_results attribute
            figsize (tuple): (width, height) of figure in inches
            x_transform, y_transform (callable): functions that take
                    a numpy array of values and return an array of the 
                    same size
            
        Returns:
            matplotlib (figure, axis) tuple
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Objective value")
    
    if hasattr(basis, 'opt_results'):
        steps = {}
        values = {}
        results = basis.opt_results
        for k, v in results.items():
            if 'atomicopt' in k:
                key = basis._symbol
                steps[key], values[key] = extract_steps(results,
                                                        key='fun')
                break
            else:
                steps[k], values[k] = extract_steps(v, key='fun')
        
        for k in steps.keys():
            ax.plot(x_transform(steps[k]), y_transform(values[k]), 
                    'x', ms=8, label=k)
        if (len(steps)) > 1:
            ax.legend()
    else:
        raise TypeError("Not a suitable Basis object")
    return fig, ax
    
def plot_exponents(basis, atoms=[], split_by_shell=True,
                   log_scale=True, figsize=(9, 9)):
    """ Creates event plots to visualize exponents in a basis set.
                   
        Arguments:
                basis (dict): internal basis object
                atoms (list): list of atoms to plot for
                split_by_shell (bool): if True, the event plots will be
                   split by shell, with a different plot for each atom
                log_scale (bool): if True, exponents will be in log_10
                figsize (tuple): (width, heigh) in inches of the figure
                   
        Returns:
                matplotlib figure, [list of matplotlib axes]
    """
    natoms = len(atoms)
    if natoms > 1 and split_by_shell:
        fig, axes = plt.subplots(ncols=natoms, sharey=True)
        to_build = [{k: basis[k.lower()]} for k in atoms]
    else:
        fig, ax = plt.subplots()
        axes = [ax]
        to_build = [{k: basis[k.lower()] for k in atoms}]
    fig.set_size_inches(figsize)
    
    def _single_plot(bas, ax):
        flat_bases = []
        for k, v in bas.items():
            flat_basis = [s.exps for s in v] 
            if log_scale:
                flat_basis = [np.log10(x) for x in flat_basis]
            if not split_by_shell:
                flat_basis = np.concatenate(flat_basis)
                flat_bases.append(flat_basis)
            else:
                flat_bases = flat_basis
        colors = ['C{}'.format(i) for i in range(len(flat_bases))]
        ax.eventplot(flat_bases, orientation='vertical',
                     linelengths=0.5, colors=colors)
            
        if split_by_shell:
            ax.set_xticks([i for i in range(len(flat_bases))])
            ax.set_xticklabels([s.l for v in bas.values() for s in v])
        else:
            ax.set_xticks([i for i in range(len(bas))])
            ax.set_xticklabels([k for k in bas.keys()])
        
        if log_scale:
            ax.set_ylabel(r"$\log_{10}$ (exponent)")
        else:
            ax.set_ylabel("Exponent")
    
    for bas, ax in zip(to_build, axes):
        _single_plot(bas, ax)    
        
    return fig, axes
    
        
             