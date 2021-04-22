# correlation consistent plots
import numpy as np
import matplotlib.pyplot as plt

def ccplot(results):
    """ Create a Dunning plot from a correlation consistent
        optimization.
    
        Arguments:
            results: OptResult object
    """
    for k, v in results.dunning():
        x = [n for (n, e) in v]
        y = [np.log10(e) for (n, e) in v]
        plt.plot(x, y, label=k, marker='x')
    
    plt.xlabel("No. of functions")
    plt.ylabel("log(Energy / Ha)")
    plt.legend()
    plt.show()
        
             