import os
import matplotlib.pyplot as plt

# Plotter
def plot(data_name_, naming_, df_):

    plt.figure()
    plt.scatter(df_[naming_[-2]], df_[naming_[-1]], s = 10, label = data_name_) 

    plt.xlabel(naming_[-2] + " (years)")
    plt.ylabel(naming_[-1] + " ($)")

    plt.legend()
    plt.legend(loc = 'upper right')

    name = data_name_ + "_" + naming_[-1]
    path = "./plots/"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + name + ".pdf")
    