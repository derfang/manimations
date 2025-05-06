import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))  # sometimes required to run the notebook
import macromax
import numpy as np
import matplotlib.pyplot as plt

wavelength = 500e-9
source_polarization = np.array([0, 1, 0])[:, np.newaxis]  # y-polarized

# Set the sampling grid
nb_samples = 1024
sample_pitch = wavelength / 16  # overkill, just for display purposes 
x_range = sample_pitch * np.arange(nb_samples) - 4e-6

# define the medium
refractive_index = np.ones(len(x_range), dtype=np.complex64)
# define an absorbing boundary
absorbing_bound = macromax.bound.LinearBound(x_range, thickness=4e-6, max_extinction_coefficient=0.5)
# glass has a refractive index of about 1.5, also permittivity and/or permeability can be specified instead
refractive_index[(x_range >= 10e-6) & (x_range < 20e-6)] = 1.5

# point source at 0
current_density = np.abs(x_range) < sample_pitch / 2  # A single voxel with j = 1 A / m^2
current_density = source_polarization * current_density

# The actual work is done here!
solution = macromax.solve(x_range, vacuum_wavelength=wavelength, current_density=current_density,
                          refractive_index=refractive_index, bound=absorbing_bound)

fig, ax = plt.subplots(2, 1, frameon=False, figsize=(8, 6))

x_range = solution.grid[0]  # coordinates
E = solution.E[1, :]  # Electric field
H = solution.H[2, :]  # Magnetizing field
S = solution.S[0, :]  # Poynting vector
f = solution.f[0, :]  # Optical force
field_to_display = E  # The source is polarized along this dimension
max_val_to_display = np.maximum(np.amax(np.abs(field_to_display)), np.finfo(field_to_display.dtype).eps)  # avoid /0
poynting_normalization = np.amax(np.abs(S)) / max_val_to_display
ax[0].plot(x_range * 1e6, np.abs(field_to_display)**2 / max_val_to_display, color=[0, 0, 0])[0]
ax[0].plot(x_range * 1e6, np.real(S) / poynting_normalization, color=[1, 0, 1])[0]
ax[0].plot(x_range * 1e6, np.real(field_to_display), color=[0, 0.7, 0])[0]
ax[0].plot(x_range * 1e6, np.imag(field_to_display), color=[1, 0, 0])[0]
figure_title = "Iteration %d, " % solution.iteration
ax[0].set_title(figure_title)
ax[0].set_xlabel("x  [$\mu$m]")
ax[0].set_ylabel("I, E  [a.u.]")
ax[0].set_xlim(x_range[[0, -1]] * 1e6)

ax[1].plot(x_range[-1] * 2e6, 0, color=[0, 0, 0], label='I')
ax[1].plot(x_range[-1] * 2e6, 0, color=[1, 0, 1], label='$S_{real}$')
ax[1].plot(x_range[-1] * 2e6, 0, color=[0, 0.7, 0], label='$E_{real}$')
ax[1].plot(x_range[-1] * 2e6, 0, color=[1, 0, 0], label='$E_{imag}$')
ax[1].plot(x_range * 1e6, refractive_index.real, color=[0, 0, 1], label='$n_{real}$')
ax[1].plot(x_range * 1e6, refractive_index.imag, color=[0, 0.5, 0.5], label='$n_{imag}$')
ax[1].set_xlabel('x  [$\mu$m]')
ax[1].set_ylabel('n')
ax[1].set_xlim(x_range[[0, -1]] * 1e6)
ax[1].legend(loc='upper right')

plt.tight_layout()
plt.show()
# save the figure
fig.savefig('macromax_1d.png', dpi=300, bbox_inches='tight')
