import csv
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['xtick.labelsize'] =20 
mpl.rcParams['ytick.labelsize'] =20 

xs = [0, 1, 2, 3, 4]
race = [0.7678, 0.6419,0.6187,0.5664,0.5546]
d0 = [1,0.8107,0.7646,0.7411,0.698]
d1 = [0.9998, 0.9989, 0.9292, 0.8867, 0.8332]
d2 = [0.9997, 0.9964, 0.9957, 0.9895, 0.9713]
d3 = [0.9997, 0.9966, 0.9958, 0.9932, 0.9918]
d5 = [0.9999, 0.9955, 0.9945, 0.9931, 0.9933]
fig, ax = plt.subplots(1, 1, figsize=(15,7.5))
ax.plot(xs, race, linestyle='--', color='tan', label='race model', marker='1', linewidth=4)
ax.plot(xs, d0, linestyle='--', color='red', label='D<=0 model', marker='o', linewidth=4)
ax.plot(xs, d1, linestyle='--', color='blue', label='D<=1 model', marker='v', linewidth=4)
ax.plot(xs, d2, linestyle='--', color='cyan', label='D<=2 model', marker='*', linewidth=4)
ax.plot(xs, d3, linestyle='--', color='purple', label='D<=3 model', marker='D', linewidth=4)
ax.plot(xs, d5, linestyle='--', color='green', label='D<=5 model', marker='s', linewidth=4)
#ax.axhline(y=0.33333333333, color='gray', linestyle=':')

ax.set_ylabel('accuracy', fontweight="bold", fontsize=22)
ax.set_xlabel('#layers', fontweight="bold", fontsize=22)
ax.set_title('Accuracy vs. Tested on Different Datasets', fontweight="bold", fontsize=25)
ax.grid(True)
ax.set_xticks([0,1,2,3,4])
ax.set_xticklabels(['D<=0','D<=1','D<=2','D<=3','D<=5'])

ax.legend(loc=3, fontsize=22)
plt.show()

fig.savefig("exp1.pdf", bbox_inches='tight')

