import numpy as np
import matplotlib.pyplot as plt

def potentiel_moyen_2d(m1, m2, nu, sigm):
    part1 = (3*nu*sigm + nu**3)*(m1**3 + m2**3) + 3*nu*(sigm + nu*nu)*(m1*m1*m2 + m1*m2*m2)
    part2 = (nu*nu*(m1+m2)**2 + sigm*(m1*m1 + m2*m2))**2
    #part3 = m1*m1 + m2*m2 - 2*sigm - 2*nu*(m1+m2)  # essai d'un terme quadratique
    return -part1/3.0 + 0.25*part2# - 0.05*part3

def potentiel_instant_x(m, thresh, x):
    return -(m.dot(x))**3 / 3.0 + 0.25*(m.dot(x))**2 * thresh

def potentiel_instant_c(c, thresh):
    return -c**3 / 3.0 + 0.25*thresh * (c**2)

# Potentiel moyen en fonction de m
def main_pot_moyen():
    m1g, m2g = np.meshgrid(np.arange(-1, 2, 0.1), np.arange(2, -1, -0.1))
    potgrid = potentiel_moyen_2d(m1g, m2g, 0.5, 0.25)
    print(m2g)
    logpotgrid = np.log10(potgrid - potgrid.min() + 1)
    fig, ax = plt.subplots()
    img1 = ax.imshow(potgrid,
        extent=(m1g[0, 0], m1g[0, -1], m2g[-1, 0], m2g[0, 0]))
    fig.colorbar(img1, ax=ax)
    ax.scatter([0, 0.5], [0, 0.5], c="w")
    plt.show()
    plt.close()


# Potentiel instantané en fonction de m.
# Montrer comment le potentiel évolue au fil du temps avec x qui varie vite.
def main_pot_variable():
    pass

# Potentiel instantané en fonction de c = m.x
# Le potentiel change lentement avec le seuil theta,
# c fluctue fortement et rapidement avec x
# (le point saute d'un endroit à l'autre dans le potentiel)
# et aussi lentement avec m.
def main_pot_c():
    pass
