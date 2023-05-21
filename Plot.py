import matplotlib.pyplot as plt
import numpy as np

y = np.array([81.59091851555118, 81.59091851555118,81.59091851555118, 81.59091851555118, 81.59091851555118, 81.59091851555118,
              81.59091851555118, 81.59091851555118, 91.40331726313924, 91.40331726313924, 1100.3094831388373, 1100.3094831388373,
              1100.3094831388373, 1101.9906866160084, 1101.9906866160084, 1102.842199729604, 1102.842199729604, 1102.842199729604,
              1102.842199729604, 1102.842199729604, 1102.842199729604, 1102.842199729604, 1105.7478833825649, 1105.7478833825649,
              1105.7478833825649, 1105.7478833825649, 1105.7478833825649, 1105.7478833825649, 1105.7478833825649, 1105.7478833825649,
              1105.7478833825649, 1106.7384845653964, 1108.97454125622, 1108.97454125622, 1108.97454125622, 1108.97454125622,
              1108.97454125622, 1108.97454125622, 1108.97454125622, 1108.97454125622, 1108.97454125622, 1108.97454125622,
              1108.9783688036343, 1108.9783688036343, 1108.9783688036343, 1108.9783688036343, 1108.9783688036343, 1108.9783688036343,
              1108.9783688036343, 1108.9783688036343, 1108.9783688036343, 1108.9783688036343, 1108.9783688036343, 1108.9783688036343,
              1108.9783688036343, 1108.9783688036343, 1108.9783688036343, 1108.9783688036343, 1108.9783688036343])

xarr = []
for i in range (59):
    xarr.append(i)
x = np.array(xarr)
plt.plot(x, y)
plt.xlabel("Generation", fontsize=18)
plt.ylabel("Best Value", fontsize=18)
plt.show()
plt.savefig("Hackatum Statistic.png",format="png")