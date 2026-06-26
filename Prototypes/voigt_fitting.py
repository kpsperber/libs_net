import numpy as np
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
from scipy.optimize import curve_fit



def voigt(x, amp, mu, sigma, gamma):
    return amp * voigt_profile(x - mu, sigma, gamma)

N = 500
x = np.linspace(-10, 10, N)
dx = x[1] - x[0]
y = np.exp(-0.5 * x ** 2)

amp_guess = np.max(y)
mu_guess = 0
sigma_guess = np.sqrt(np.sum((x - mu_guess) ** 2 * y * dx) / np.sum(y * dx))
gamma_guess = np.diff(x[np.where(np.abs(y - y.max() / 2) < 0.3)])[0]
initial_guess = [amp_guess, sigma_guess, gamma_guess]

bounds = ((0, 0, 0), (np.inf, np.inf, np.inf))

def voigt_fixed(x, amp, sigma, gamma):
    return voigt(x, amp, mu_guess, sigma, gamma)

popt, pcov = curve_fit(voigt_fixed, x, y, p0 = initial_guess, bounds = bounds)

x_fit = np.linspace(x.min(), x.max(), 1024)

fig, ax = plt.subplots(1)
ax.plot(x, y, "o")
ax.plot(x_fit, voigt_fixed(x_fit, *popt))

ax.grid(True)

plt.show()

print()