import matplotlib.pyplot as plt
import numpy as np

dimensioni = ["FAMILIARITY", "MEANINGFULNESS", "DIFFICULTY", 
              "BODY_RELATEDNESS", "PHISICALITY", "IMAGEABILITY"]

RS = [0.1781, 0.2396, 0.0952, 0.8099, 0.4731, 0.4888]
RM = [0.1648, 0.3026, 0.1199, 0.8081, 0.4720, 0.4686]
M  = [0.2185, 0.3163, 0.1821, 0.8347, 0.5032, 0.5563]

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"] 

n_row = 3
indices_row1 = np.arange(n_row)
indices_row2 = np.arange(len(dimensioni) - n_row)

width = 0.15

fig, axs = plt.subplots(2, 1, figsize=(12, 8))

axs[0].bar(indices_row1 - width, RS[:n_row], width, color=colors[0], label="RS")
axs[0].bar(indices_row1, RM[:n_row], width, color=colors[1], label="RM")
axs[0].bar(indices_row1 + width, M[:n_row], width, color=colors[2], label="M")

axs[0].set_xticks(indices_row1)
axs[0].set_xticklabels(dimensioni[:n_row])
axs[0].set_ylim(0, 1)
axs[0].set_ylabel("Correlazione")
axs[0].legend()

axs[1].bar(indices_row2 - width, RS[n_row:], width, color=colors[0])
axs[1].bar(indices_row2, RM[n_row:], width, color=colors[1])
axs[1].bar(indices_row2 + width, M[n_row:], width, color=colors[2])

axs[1].set_xticks(indices_row2)
axs[1].set_xticklabels(dimensioni[n_row:])
axs[1].set_ylim(0, 1)
axs[1].set_ylabel("Correlazione")

plt.tight_layout()

plt.savefig("correlazioni_condizioni.png", dpi=300)
plt.show()
