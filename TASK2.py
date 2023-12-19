import os.path
import requests as r
import re
import scipy.special as sci
import numpy as np
import matplotlib.pyplot as plt
import math



def JOrd(order, countt):
    jblist = (sci.spherical_jn(order, k[countt] * D / 2))
    return jblist


def HOrd(order, countt):
    yblist = (sci.spherical_yn(order, k[countt] * D / 2))
    hblist = (complex(JOrd(order, countt), yblist))
    return hblist


def AOrd(order, countt):
    anlist = (JOrd(order, countt) / HOrd(order, countt))
    return anlist


def BOrd(order, countt):
    bnlist = (((k[countt] * (D / 2) * JOrd(order - 1, countt) - order * JOrd(order, countt)) / (k[countt] * (D / 2) * HOrd(order - 1, countt) - order * HOrd(order, countt))))
    return bnlist


if os.path.exists('download') == False:
    os.mkdir('download')

url = 'https://jenyay.net/uploads/Student/Modelling/task_02_02.txt'

file = r.get(url)
if file.ok is False:
    print('Smth went wrong')
    quit()

with open('download/Task2Parametrs.txt', 'wb') as f:
    f.write(file.content)
with open('download/Task2Parametrs.txt', 'r') as f:
    data = f.readlines()

needvar = data[13]
print(needvar)

pattern = r'(?<!\d)\d+[.]\d+|\d+|-\d+(?!\d)'
zn = re.findall(pattern, str(needvar))
print(zn)

D = float(zn[1])*10**float(zn[2])
fmin = float(zn[3])*10**float(zn[4])
fmax = float(zn[5])*10**float(zn[6])
c = 3*10**8
print("Диаметр сферы D, м:",D)
print("fmin, Гц:", fmin)
print("fmax, Гц:", fmax)

count = 200
step = 25

flist = np.linspace(fmin, fmax, count)
k = []
for i in range(count):
    k.append(2 * math.pi * flist[i] / c)

j = 1
ssuumm = [0]*count
while j <= step:
    for i in range(count):
        ssuumm[i] += ((-1)**j * (j+0.5) * (BOrd(j, i) - AOrd(j, i)))
    j += 1

epr = []
lam = []
for i in range(count):
    epr.append((c ** 2 / (math.pi * flist[i] ** 2)) * (abs(ssuumm[i]) ** 2))
    lam.append(c/flist[i])

if os.path.exists('results2') == False:
    os.mkdir('results2')

flist1 = flist.tolist()

with open('results2/results_2.json', 'w') as f:
    f.write("\"data\": [\n")
    for i,j,k in zip(flist1, lam, epr):
        f.write( "\t  \"freq\": {:<20.5f} \"lambda\": {:<12.5f} \"rcs\": {:<12.12f}\n".format(i,j,k))
    f.write("]")

plt.plot(flist, epr)
plt.show()
