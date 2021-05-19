import csv
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['xtick.labelsize'] =20 
mpl.rcParams['ytick.labelsize'] =20 
xs = [0, 5, 10, 15, 20, 24]
d0 = [0.4617, 0.5509, 0.6103, 0.9711, 1, 1]
d1 = [0.484, 0.5126, 0.5528, 0.8397, 0.9979, 0.9989]
d2 = [0.4574, 0.4967, 0.5492, 0.7639, 0.9941, 0.9957]
d3 = [0.4475, 0.5217, 0.5416, 0.7967, 0.9913, 0.9932]
d5 = [0.432, 0.5267, 0.5611, 0.832, 0.9927, 0.9933]
fig, ax = plt.subplots(1, 1, figsize=(15,7.5))
ax.plot(xs, d0, linestyle='--', color='red', label='D<=0', marker='o', linewidth=4)
ax.plot(xs, d1, linestyle='--', color='blue', label='D<=1', marker='v', linewidth=4)
ax.plot(xs, d2, linestyle='--', color='cyan', label='D<=2', marker='*', linewidth=4)
ax.plot(xs, d3, linestyle='--', color='purple', label='D<=3', marker='D', linewidth=4)
ax.plot(xs, d5, linestyle='--', color='green', label='D<=5', marker='s', linewidth=4)

ax.set_ylabel('accuracy', fontweight="bold", fontsize=22)
ax.set_xlabel('#layers', fontweight="bold", fontsize=22)
ax.set_title('Accuracy vs. #Layers', fontweight="bold", fontsize=25)
ax.grid(True)

ax.legend(fontsize = 22)
plt.show()

fig.savefig("exp.pdf", bbox_inches='tight')

