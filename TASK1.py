import matplotlib.pyplot as plt
import numpy as np
import os.path


def func(x):
    A = 10
    return np.sin(A * np.pi * x)/(2 * x) + (x - 1)**4


xmin = -5.12
xmax = 5.12
step = 0.01
count = 1000

xlist = np.arange(xmin, xmax, step)
ylist = []

for i in range(len(xlist)):
    y = func(xlist[i])
    ylist.append(y)

if os.path.exists('results') == False:
    os.mkdir('results')

with open ("results/results.txt", "w") as f:
    textlist = {"x f(x)"}
    f.write("   x \t       f(x)\n")
    for i,j in zip(xlist,ylist):
        f.write("{:<12.5f}{}\n".format(i,j))

plt.plot(xlist, ylist)
plt.savefig('results/plot_1.png', dpi=50, bbox_inches='tight')
plt.show()