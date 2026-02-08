import math
from scipy.stats import norm

# ---- tuoi dati ----
r1 = 0.5262648815105805
n1 = 90

r2 = 0.8186630063610546
n2 = 70

# ---- funzione Fisher z ----
def fisher_z(r):
    return 0.5 * math.log((1 + r) / (1 - r))

# trasformazione
z1 = fisher_z(r1)
z2 = fisher_z(r2)

# errore standard
se = math.sqrt(1/(n1 - 3) + 1/(n2 - 3))

# statistica z
z_stat = (z1 - z2) / se

# p-value due code
p_value = 2 * (1 - norm.cdf(abs(z_stat)))

print("z statistic:", z_stat)
print("p-value:", p_value)
