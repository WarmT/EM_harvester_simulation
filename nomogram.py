from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

f = np.logspace(0, 4, 40)
print "f = "
print f
#x_array = np.logspace(-8, 1, 80)
x_array = np.array([1, 2, 5, 10, 20, 50, 100]) * 1e-3
#a = np.logspace(-2, 5, 80)
a_array = np.array([1, 2, 5, 10])
#xd2 = a / (2 * np.pi * f)

print "f = "
print f
print "x_array = "
print x_array

fig = plt.figure(facecolor='white', figsize=(10, 10))
ax = plt.gca()


for i in xrange(9):
    for x in np.linspace(10**(i - 9), 10**(i - 8), 10):
        v = x * 2 * np.pi * f
        plt.plot(f, v, 'k')

for i in xrange(9):
    for a in np.linspace(10**(i - 3), 10**(i - 2), 10):
        v = a / (2 * np.pi * f)
        plt.plot(f, v, 'k')

f_size=13

ax.text(11e3, 6.5e-4, r"$0.1\,\mathrm{um}$", rotation=45, ha='left', va='bottom', fontsize=f_size)
ax.text(11e3, 6.5e-3,   r"$1\,\mathrm{um}$", rotation=45, ha='left', va='bottom', fontsize=f_size)
ax.text(11e3, 6.5e-2,  r"$10\,\mathrm{um}$", rotation=45, ha='left', va='bottom', fontsize=f_size)
ax.text(11e3, 6.5e-1, r"$100\,\mathrm{um}$", rotation=45, ha='left', va='bottom', fontsize=f_size)
ax.text(11e3, 6.5e-0,   r"$1\,\mathrm{mm}$", rotation=45, ha='left', va='bottom', fontsize=f_size)
ax.text(11e3, 6.5e1,   r"$10\,\mathrm{mm}$", rotation=45, ha='left', va='bottom', fontsize=f_size)
ax.text(1.7e2, 10,    r"$100\,\mathrm{mm}$", rotation=45, ha='left', va='bottom', fontsize=f_size)
ax.text(1.7e3, 10,     r"$10\,\mathrm{mm}$", rotation=45, ha='left', va='bottom', fontsize=f_size)

ax.text(1, 1.5e-4, r"$0.001\,\mathrm{m/s}^2$", rotation=-45, ha='right', va='bottom', fontsize=f_size)
ax.text(1, 1.5e-3, r"$0.01\,\mathrm{m/s}^2$",  rotation=-45, ha='right', va='bottom', fontsize=f_size)
ax.text(1, 1.5e-1, r"$1\,\mathrm{m/s}^2$",     rotation=-45, ha='right', va='bottom', fontsize=f_size)
ax.text(1, 1.5e-0, r"$10\,\mathrm{m/s}^2$",    rotation=-45, ha='right', va='bottom', fontsize=f_size)
ax.text(1.7e0, 10, r"$100\,\mathrm{m/s}^2$",   rotation=-45, ha='right', va='bottom', fontsize=f_size)
ax.text(1.7e1, 10, r"$1000\,\mathrm{m/s}^2$",  rotation=-45, ha='right', va='bottom', fontsize=f_size)
ax.text(1.7e2, 10, r"$10000\,\mathrm{m/s}^2$", rotation=-45, ha='right', va='bottom', fontsize=f_size)

f_severity = np.array([1, 30, 1000, 10e3])

v_1Hz   = 635e-6 / 2 * 2 * np.pi * 1
v_30Hz  = 635e-6 / 2 * 2 * np.pi * 30
a = v_30Hz * 2 * np.pi * 1000
v_10kHz = a / (2 * np.pi * 10e3)
v_l_extreme = np.array([v_1Hz, v_30Hz, v_30Hz, v_10kHz])

v_1Hz   = 266.7e-6 / 2 * 2 * np.pi * 1
v_30Hz  = 266.7e-6 / 2 * 2 * np.pi * 30
a = v_30Hz * 2 * np.pi * 1000
v_10kHz = a / (2 * np.pi * 10e3)
v_l_excessive = np.array([v_1Hz, v_30Hz, v_30Hz, v_10kHz])

v_1Hz   = 95.25e-6 / 2 * 2 * np.pi * 1
v_30Hz  = 95.25e-6 / 2 * 2 * np.pi * 30
a = v_30Hz * 2 * np.pi * 1000
v_10kHz = a / (2 * np.pi * 10e3)
v_l_tolerable  = np.array([v_1Hz, v_30Hz, v_30Hz, v_10kHz])

v_1Hz   = 38.1e-6 / 2 * 2 * np.pi * 1
v_30Hz  = 38.1e-6 / 2 * 2 * np.pi * 30
a = v_30Hz * 2 * np.pi * 1000
v_10kHz = a / (2 * np.pi * 10e3)
v_l_acceptable = np.array([v_1Hz, v_30Hz, v_30Hz, v_10kHz])

plt.plot(f_severity, v_l_acceptable, 'b', lw=3)
plt.plot(f_severity, v_l_tolerable,  'b', lw=3)
plt.plot(f_severity, v_l_excessive,  'b', lw=3)
plt.plot(f_severity, v_l_extreme,    'b', lw=3)

plt.text(200, 2e-3,  "Good",       color="blue", ha='center', fontsize=16, bbox=dict(boxstyle="square", ec=(1., 1., 1.), fc=(1., 1., 1.),))
ax.text(200, 5e-3,   "Acceptable", color="blue", ha='center', fontsize=16, bbox=dict(boxstyle="square", ec=(1., 1., 1.), fc=(1., 1., 1.),))
ax.text(200, 1.4e-2, "Tolerable",  color='blue', ha='center', fontsize=16, bbox=dict(boxstyle="square", ec=(1., 1., 1.), fc=(1., 1., 1.),))
ax.text(200, 3.5e-2, "Excessive",  color='blue', ha='center', fontsize=16, bbox=dict(boxstyle="square", ec=(1., 1., 1.), fc=(1., 1., 1.),))
ax.text(200, 9e-2, "Extreme",      color='blue', ha='center', fontsize=16, bbox=dict(boxstyle="square", ec=(1., 1., 1.), fc=(1., 1., 1.),))

plt.xscale('log')
plt.yscale('log')

ax.tick_params(width=1, which="both")
ax.tick_params(length=8, which="major")
ax.tick_params(length=4, which="minor")

plt.xlabel("Frequency [Hz]")
plt.ylabel("Velocity [m/s]")
plt.axis([1, 10e3, 0.1e-3, 10])
plt.axes().set_aspect('equal')
#plt.axis([1, 10, 100, 1000])
plt.grid(True, which="both")
plt.show(block=False)
plt.savefig("vibration_nomograph.pdf")
raw_input("haa")
plt.close()
