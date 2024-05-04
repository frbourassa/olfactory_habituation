import numpy as np
from utils.metrics import l2_norm
from modelfcts.ideal import find_projector, find_parallel_component

import matplotlib.pyplot as plt


def generate_odor_set(n_b, n_r, rng):
    back_vecs = rng.exponential(size=[n_b, n_r])
    back_vecs /= l2_norm(back_vecs, axis=1)[:, None]
    return back_vecs


def check_xnort(n_components, n_orn, rgen, repeats=1000):

    # Generate 1000 backgrounds, 1000 new odors
    if n_components < 0 or n_components > n_orn:
        raise ValueError("Invalid number of components")
    n_new = repeats
    n_backs = repeats
    all_xort = []
    x_n_mean_elements = []
    for b in range(n_backs):
        back_components = generate_odor_set(n_components, n_orn, rgen)
        new_odors = generate_odor_set(n_new, n_orn, rgen)
        x_n_mean_elements.append(np.mean(new_odors))
        proj = find_projector(back_components.T)
        for n in range(n_new):
            x_ort = new_odors[n] - find_parallel_component(
                        new_odors[n], back_components, projector=proj
                    )
            all_xort.append(x_ort)
    x_n_mean_elements = np.mean(x_n_mean_elements)
    x_n_ort_mean_prediction = (1.0 - n_components/n_orn)*x_n_mean_elements
    all_xort = np.stack(all_xort, axis=0)
    xort_norm2_mean = np.mean(np.sum(all_xort**2, axis=1))
    x_n_ort_mean_numerical = np.mean(all_xort)
    print("Mean x_n_ort component, prediction:", x_n_ort_mean_prediction)
    print("Numerical:", x_n_ort_mean_numerical)
    print("x_n_ort norm2, computed:", xort_norm2_mean)
    return [x_n_ort_mean_prediction, x_n_ort_mean_numerical, xort_norm2_mean]

if __name__ == "__main__":
    rgen = np.random.default_rng(0x9751b262b38be1c3d0cf73e14e3e9a97)
    analytical_line = []
    numerical_line = []
    norm2_line = []
    n_orn = 25
    n_b_trials = np.arange(0, n_orn+1, 1)
    for n_b in n_b_trials:
        res = check_xnort(n_b, n_orn, rgen, repeats=200)
        analytical_line.append(res[0])
        numerical_line.append(res[1])
        norm2_line.append(res[2])

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes = axes.flatten()
    axes[0].plot(n_b_trials, analytical_line, label="Linear prediction")
    axes[0].plot(n_b_trials, numerical_line, label="Computed result")
    axes[1].plot(n_b_trials, 1.0 - np.asarray(norm2_line))
    axes[0].set(ylabel=r"$\langle \vec{x}_{n, \perp} \rangle$ element")
    axes[1].set(xlabel=r"$n_{B}$", ylabel=r"$\langle \vec{x}_{n, \parallel}^2 \rangle$")
    axes[0].legend()
    fig.tight_layout()
    plt.show()
    plt.close()
