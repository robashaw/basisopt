# plot an orbital
import numpy as np
import matplotlib.pyplot as plt

def ccplot(results):
    for k, v in results.errors.items():
        x = [n for (n, e) in v]
        y = [np.log10(e) for (n, e) in v]
        plt.plot(x, y, label=k, marker='x')
    
    plt.xlabel("No. of functions")
    plt.ylabel("log(Energy / Ha)")
    plt.legend()
    plt.show()
        
             