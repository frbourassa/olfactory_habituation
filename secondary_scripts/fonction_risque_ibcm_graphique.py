import numpy as np
import matplotlib.pyplot as plt

def fonction_risque(c, theta, mu=1.0):
    return -mu * (c**3/3.0 - 0.25*theta**2)

def fonction_cout(c, theta, mu=1.0):
    return -mu * (c**3/3.0 - 0.25*theta * c**2)

if __name__ == "__main__":
    # Graphique qualitatif de la fonction de risque pour un certain theta
    # et un axe des x représentant la moyenne des c^3.
    c_axe = np.linspace(-0.5, 1.0, 201)
    theta_seuil = 1.0
    risque = fonction_cout(c_axe, theta_seuil, mu=1.0)

    fig, ax = plt.subplots()
    fig.set_size_inches(4, 2.75)
    ax.plot(c_axe, risque, color="k", lw=2.5, zorder=100)

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.plot([0.5*theta_seuil]*2,
        [0.0, fonction_cout(0.5*theta_seuil, theta_seuil, mu=1.0)],
        color="k", ls="--", lw=1.0)
    ax.set_xlabel(r"$c$", loc="right", labelpad=-10)
    ax.annotate(r"$L_m(c)$", xy=(c_axe[5], np.amax(risque)), ha="left", va="center")
    ax.set_xticks([-0.5*theta_seuil, 0.5*theta_seuil, 0.75*theta_seuil,
                    theta_seuil])
    ax.set_xticklabels([r"$-\frac{1}{2}\Theta$", r"$\frac{1}{2}\Theta$",
                        r"$\frac{3}{4}\Theta$", r"$\Theta$"])
    fig.tight_layout()

    # Draw a fictitious distribution of input values on top?
    c_axe = np.linspace(-0.5, 1.1, 221)
    gauss1 = 0.75*np.amax(risque) * np.exp(-c_axe**2 / 2 / (0.125*theta_seuil)**2)
    gauss2 = 0.5*np.amax(risque) * np.exp(-(c_axe - 0.9*theta_seuil)**2 / 2 / (0.05*theta_seuil)**2)
    ax.plot(c_axe, gauss1+gauss2, lw=1.0, color=(0.7,)*3)
    ax.fill_between(c_axe, 0.0, gauss1+gauss2, color=(0.7,)*3, alpha=0.4)
    ax.annotate(r"$P(c)$", xy=(0.05, np.amax(gauss1)), ha="left", va="bottom", color="grey")


    fig.savefig("figures/ibcm_loss_function_fixed_theta.pdf",
                transparent=True, bbox_inches="tight")
    plt.show()
    plt.close()
