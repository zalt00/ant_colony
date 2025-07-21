import json
import numpy as np
import matplotlib.pyplot as plt
with open("data.json") as file:
    (v1, v2, v3) = json.load(file)

v1 = np.array(v1) / max(v1)
v2 = np.array(v2) / max(v2)
v3 = np.array(v3) / max(v3)

plt.plot(v1, label="mp")
plt.plot(v2, label="stretch")
plt.plot(v3, label="disto")

plt.plot(v1 / v3, label="mps")
plt.plot(v2 / v3, label="stres")

plt.legend(loc='best')
plt.show()


