try:
    import cupy as xp
    USE_CUDA = True
    print('lpshg running on GPU')
except:
    import numpy as xp
    USE_CUDA = False
    print('lpshg running on CPU')

import numpy as np
from scipy.constants import epsilon_0, c

class LPSHGsim:
    def __init__(self, I, X_size, Y_size, T_size, xtal_length=0.03):
        self.lam_fundamental = 1030e-9  # wavelength of fundamental [m]
        self.deff = 0.83e-12  # d_eff [m/V]
        self.n1 = 1.606  # refractive index of fundamental
        self.n2 = 1.606  # refractive index of second harmonic
        self.dk = (
            2 * 2 * xp.pi * self.n1 / self.lam_fundamental
            - 2 * xp.pi * self.n2 / (self.lam_fundamental / 2)
        ) # wavevector mismatch

        self.A1 = self.convert_intensity_to_amplitude(xp.array(I)) # Field amplitude of fundamental
        self.A2 = xp.zeros_like(self.A1) # Field amplitude of second harmonic

        self.Nz = 500 # number of steps along z
        self.Z = xtal_length # total distance along z

        self.X = X_size # size of simulation grid in x
        self.Y = Y_size # size of simulation grid in x
        self.T = T_size # size of simulation grid in t

        (self.Nx, self.Ny, self.Nt) = self.A1.shape # number of grid points along x, y and t

        self.energy_shg = [] # list to store energy during conversion
        self.energy_fund = [] # list to store energy during conversion

    def convert_intensity_to_amplitude(self, I):
        # convert intensity to a complex field amplitude
        return xp.sqrt(2 * I / (self.n1 * c * epsilon_0)) * xp.exp(-0 * 1j)

    def init_grid(self):
        # initialize the grid 
        self.dz = self.Z / self.Nz
        self.z = np.linspace(0, self.Z, self.Nz)

        self.dx = self.X / self.Nx
        ax = xp.linspace(-self.Nx / 2, self.Nx / 2, self.Nx + 1)[:-1]
        fx = ax / self.X

        self.dy = self.Y / self.Ny
        ay = xp.linspace(-self.Ny / 2, self.Ny / 2, self.Ny + 1)[:-1]
        fy = ay / self.Y
        print(fy)
        self.FX, self.FY = xp.meshgrid(fx, fy)
        
        self.dt = self.T / self.Nt

    def SHG_ode(self, _t, _y, n1, n2):
        # ODE that describes the nonlinear interaction
        y1 = _y[0]
        y2 = _y[1]

        b1 = (
            -1j
            * 2
            * xp.pi
            * c
            / self.lam_fundamental
            * self.deff
            / (n1 * c)
            * xp.exp(-1j * _t * self.dk)
        )
        b2 = (
            -1j
            * 2
            * xp.pi
            * c
            / self.lam_fundamental
            * self.deff
            / (n2 * c)
            * xp.exp(1j * _t * self.dk)
        )

        a1 = b1 * y2 * xp.conjugate(y1)
        a2 = b2 * y1**2

        return xp.array([a1, a2])

    def nonlinear_step(self):
        # do a single nonlinear step by solving the SHG ODE with runge-kutta
        sol = self.RK45_step(
            self.SHG_ode,
            xp.array([self.A1, self.A2]),
            self.dz,
            args=(self.n1, self.n2),
        )
        return sol[0], sol[1]

    def linear_propagation(self, A, lam, n):
        # linear propagation by distance self.dz
        A = xp.fft.fftshift(xp.fft.fft2(A, axes=(0, 1)), axes=(0, 1))
        A = (
            A
            * xp.exp(
                1j
                * 2
                * xp.pi
                * self.dz
                * n
                * xp.sqrt(1 / lam**2 - self.FX**2 - self.FY**2)
            )[:, :, None]
        )
        A = xp.fft.ifft2(xp.fft.ifftshift(A, axes=(0, 1)), axes=(0, 1))
        return A

    def RK45_step(self, f, y0, h, args):
        # 4th order Runge-Kutta for step length of h
        F1 = h * f(0, y0, *args)
        F2 = h * f((h / 2), (y0 + F1 / 2), *args)
        F3 = h * f((h / 2), (y0 + F2 / 2), *args)
        F4 = h * f((h), (y0 + F3), *args)

        y1 = y0 + 1 / 6 * (F1 + 2 * F2 + 2 * F3 + F4)
        return y1

    def save_energy(self):
        # save energy of fundamental and second harmonic
        E_1 = xp.asarray(
            xp.sum(xp.abs(self.A1) ** 2)
            * self.dx
            * self.dy
            * self.dt
            * (self.n1 * c * epsilon_0)
            / 2
        )
        E_2 = xp.asarray(
            xp.sum(xp.abs(self.A2) ** 2)
            * self.dx
            * self.dy
            * self.dt
            * (self.n2 * c * epsilon_0)
            / 2
        )

        self.energy_fund.append(E_1)
        self.energy_shg.append(E_2)

    def get_intensies(self):
        # return the 3D intensity profile
        if USE_CUDA:
            I1 = xp.asarray(xp.abs(self.A1) ** 2 * (c * epsilon_0 * self.n1) / 2).get()
            I2 = xp.asarray(xp.abs(self.A2) ** 2 * (c * epsilon_0 * self.n2) / 2).get()
        else:
            I1 = xp.asarray(
                xp.abs(self.A1) ** 2 * (c * epsilon_0 * self.n1) / 2
            )  # .get()
            I2 = xp.asarray(
                xp.abs(self.A2) ** 2 * (c * epsilon_0 * self.n2) / 2
            )  # .get()
        return I1, I2

    def run(self):
        # run the actual simulation
        self.init_grid()
        # loop over all steps along z
        for i in range(self.Nz):
            # do a nonlinear step
            self.A1, self.A2 = self.nonlinear_step()
            # linear propagation of fundamental field
            self.A1 = self.linear_propagation(self.A1, self.lam_fundamental, self.n1)
            # linear propagation of second harmonic field
            self.A2 = self.linear_propagation(
                self.A2, self.lam_fundamental / 2, self.n2
            )
            self.save_energy()

            print("{:.2f}% done".format(i / (self.Nz - 1) * 100), end="\r")

        if USE_CUDA:
            # if run on GPU, convert the recorded energy to numpy compatible array
            self.energy_fund = xp.array(self.energy_fund).get()
            self.energy_shg = xp.array(self.energy_shg).get()
