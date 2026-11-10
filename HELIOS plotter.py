import numpy as np
import matplotlib.pyplot as plt
import helios_parser as hp
import h5py
from pathlib import Path

folder = Path("DHe3ab")

with h5py.File("C:/Users/Raphael/Desktop/NeSST output/0_50DHe3ab1.h5", "r") as f3:
    dt_0_50 = f3["nesst_inputs"]["dt"][()]

def read_final_yield(file):
    ddny = None
    dhpy = None
    ddn_name = "2.45 MeV neutrons"
    prefixn = f"{ddn_name} =  "
    dhp_name = "14.67 MeV protons"
    prefixp = f"{dhp_name}  =  "

    with open(file, 'r') as file:
        for line in reversed(list(file)):
            if ddny is None and prefixn in line:
                ddny = float(line.split("=")[1].strip())
            elif dhpy is None and prefixp in line:
                dhpy = float(line.split("=")[1].strip())
            if ddny is not None and dhpy is not None:
                break
    return ddny, dhpy

ddnr = []
tn_prev = 0
dhpr = []
tp_prev = 0
dt_iter = iter(dt_0_50)
tn = None
tp = None
with open(f"{folder}/0_50DHe3ab1.log", 'r') as file:
    ddn_name = "2.45 MeV neutrons"
    prefixn = f"{ddn_name} =  "
    dhp_name = "14.67 MeV protons"
    prefixp = f"{dhp_name}  =  "
    for line in list(file):
        if prefixn in line:
            tn = float(line.split("=")[1].strip())
        elif prefixp in line:
            tp = float(line.split("=")[1].strip())
        if tn is not None and tp is not None:
            dti = next(dt_iter)
            ddnr.append((tn - tn_prev)/dti)
            tn_prev = tn
            dhpr.append((tp - tp_prev)/dti)
            tp_prev = tp
            tn = None
            tp = None

middle = "DHe3"
l_ab1 = []
l_ab01 = []
l_ab02 = []

for file in folder.glob("*.log"):
    filename = file.name
    if middle not in filename:
        continue
    raw_prefix, raw_suffix = filename.split(middle, 1)
    prefix_float = float(raw_prefix.replace("_", "."))
    ddny, dhpy = read_final_yield(file)
    row = [prefix_float, ddny, dhpy]
    suffix = raw_suffix.replace(".log", "")
    if "ab1" in suffix:
        l_ab1.append(row)
    elif "ab01" in suffix:
        l_ab01.append(row)
    elif "ab02" in suffix:
        l_ab02.append(row)
    ab1 = np.array(l_ab1).T
    ab01 = np.array(l_ab01).T
    ab02 = np.array(l_ab02).T

#print(ab1[1])
#print(ab1[2])
#print(ab01[1])
#print(ab01[2])
#print(ab02[1])
#print(ab02[2])

#file1 = (f"{folder}/0_50DHe3ab1.log")
#file2 = (f"{folder}/0_50DHe3ab01.log")
#file3 = (f"{folder}/0_50DHe3ab02.log")
data1 = hp.get_helios_2d_arrays(f"{folder}/1_00DHe3ab1.log")
data05 = hp.get_helios_2d_arrays(f"{folder}/0_50DHe3ab1.log")
#data3 = hp.get_helios_2d_arrays(f"{folder}/0_50DHe3ab02.log")
dt1 = np.gradient(data1["times"])
dt05 = np.gradient(data05["times"])
#dt3 = np.gradient(data3["times"])
yield1 = np.sum(dt1[:, None] * data1['neutron_rate'].T)
yield05 = np.sum(dt05[:, None] * data05['neutron_rate'].T)
#yield3 = np.sum(dt3[:, None] * data3['neutron_rate'].T)
print(yield1)
print(yield05)
#print(yield3)

fig2n = np.loadtxt(open("fig2_DD-n.csv", "rb"), delimiter=",")
fig2p = np.loadtxt(open("fig2_DHe3-p.csv", "rb"), delimiter=",")
fig3n = np.loadtxt(open("fig3_DD-n.csv", "rb"), delimiter=",")
fig3p = np.loadtxt(open("fig3_DHe3-p.csv", "rb"), delimiter=",")

x = np.linspace(0, 1, 100)

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('time (ns)')
ax1.set_xlim(x[0]+1,x[-1]+1.5)
ax1.set_ylabel('DHe3-p rate (s\u207B\u00B9)', color=color)
ax1.set_ylim(1e16,2e19)
ax1.semilogy((data05["times"]*10**9)-0.18,dhpr, '+m', label="HELIOS DHe3-p -0.18 ns")
ax1.semilogy(fig2p[:,0],fig2p[:,1], 'r', label="Rygg DHe3-p")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('DD-n rate (s\u207B\u00B9)', color=color)
ax2.set_ylim(3e17,6e20)
ax2.semilogy((data05["times"]*10**9)-0.18,ddnr, '+c', label="HELIOS DD-n -0.18 ns")
ax2.semilogy(fig2n[:,0],fig2n[:,1], 'b.', label="Rygg DD-n")
ax2.tick_params(axis='y', labelcolor=color)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

fig.tight_layout()

plt.show()

plt.figure(dpi=200)
plt.semilogy(data05["times"]*10**9,data05['neutron_rate'].sum(axis=0), 'xg', label="NeSST DD-n rate")

plt.xlim(x[0],x[-1]*3)
#plt.ylim(1e16,1e21)
plt.xlabel("time")
plt.ylabel("rate")
plt.legend()
plt.show()

plt.figure(dpi=200)

plt.plot(ab1[0],ab1[1]/max(ab1[1]), '+c', label="HELIOS DD-n")
plt.plot(ab1[0],ab1[2]/max(ab1[2]), '+m', label="HELIOS DHe3-p")
plt.plot(fig3n[:,0],fig3n[:,1], 'b', label="Rygg DD-n")
plt.plot(fig3p[:,0],fig3p[:,1], 'r.', label="Rygg DHe3-p")
"""
plt.semilogy(ab1[0],ab1[1], 'g', label="ab1n")
plt.semilogy(ab1[0],ab1[2], 'k', label="ab1p")
plt.semilogy(ab01[0],ab01[1], 'c', label="ab01n")
plt.semilogy(ab01[0],ab01[2], 'm', label="ab01p")
plt.semilogy(ab02[0],ab02[1], 'b', label="ab02n")
plt.semilogy(ab02[0],ab02[2], 'r', label="ab02p")
"""
plt.xlim(x[0]-0.01,x[-1]+0.05)
plt.ylim(-0.01,1.01)
plt.xlabel("f\u1D05")
plt.ylabel("Relative yield (Y/Y\u2098\u2090\u2093)")
plt.xticks(np.arange(0,1.5,0.5))
plt.xticks(np.arange(0,1,0.1), minor=True)
plt.yticks(np.arange(0,1,0.05), minor=True)
plt.legend(loc='upper left', fontsize='x-small')

plt.show()
