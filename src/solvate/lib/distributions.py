import numpy as np
from scipy.constants import N_A, epsilon_0, k as kB, e as e_el
from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity

N_A       *= Q_('1/mol')
epsilon_0 *= Q_('F/m')
e_el      *= Q_('C')
kB        *= Q_('J/K')


class PBTwoPlates():
    """
    Class to generate Poisson-Boltzmann distribution profiles for cations and anions
    between two charged plates.
    """


    def __init__(
        self,
        sigma_1: float,
        sigma_2: float,
        c_0: float,
        T: float,
        epsilon_r: float,
        q: float,
    ):

        # Convert inputs to pint quantities
        self._sigma_1 = Q_(sigma_1, 'elementary_charge/angstrom^2')
        self._sigma_2 = Q_(sigma_2, 'elementary_charge/angstrom^2')
        self._c_0 = Q_(c_0, 'mol/l')
        self._T = Q_(T, 'kelvin')
        self._epsilon_r = Q_(epsilon_r, 'dimensionless')
        self._q = Q_(q, 'elementary_charge')


    def _lambda_D(self, epsilon_r, T, c_0):
        """
        Calculate the Debye length.
        Positional arguments:
        epsilon_r   -- Relative permittivity of the medium.
        T           -- Temperature.
        c_0         -- Bulk concentration of cations.
        Returns:
        Debye length.
        """

        return np.sqrt(epsilon_r * epsilon_0 * kB * T / (2 * e_el**2 * c_0 * N_A))


    def _phi_0(self, sigma, lambda_D, epsilon_r):
        """
        Calculate the electrostatic potential at the plate.
        Positional arguments:
        sigma      -- Surface charge density.
        lambda_D   -- Debye length.
        epsilon_r  -- Relative permittivity of the medium.
        Returns:
        phi_0 -- Electrostatic potential at the plate.
        l_D -- Debye length.
        """

        return sigma / (epsilon_r * epsilon_0) * lambda_D
    

    def _phi_z(self, phi_0, lambda_D, z, z_0):
        """
        Calculate the electrostatic potential profile in z-direction.
        Positional arguments:
        sigma      -- Surface charge density.
        lambda_D   -- Debye length.
        z          -- Distance from the charged surface.
        epsilon_r  -- Relative permittivity of the medium.
        Returns:
        Electrostatic potential profile in z-direction.
        """

        return phi_0 * np.exp(-np.abs(z - z_0) / lambda_D)
    

    def _distribution_profile(self, q, T, phi_z):
        """
        Calculate the Poisson-Boltzmann factor profile in z-direction.
        Positional arguments:
        q    -- Charge of the ion.
        T    -- Temperature.
        phi  -- Electrostatic potential profile.
        Returns:
        Poisson-Boltzmann factor profile in z-direction.
        """

        p = np.exp(-q * phi_z / (kB * T))
        p /= np.sum(p)
        return p
    

    def calculate_p(self, z):
        """
        Calculate the Poisson-Boltzmann distribution profile between two charged plates.
        Positional arguments:
        zmin -- Minimum z-coordinate.
        zmax -- Maximum z-coordinate.
        Returns:
        Poisson-Boltzmann distribution profile between two charged plates.
        """

        z = z * ureg('angstrom')

        l_D = self._lambda_D(self._epsilon_r, self._T, self._c_0)
        print("Debye length in angstrom:", l_D.to('angstrom').magnitude)
        phi_0_1 = self._phi_0(self._sigma_1, l_D, self._epsilon_r)
        phi_0_2 = self._phi_0(self._sigma_2, l_D, self._epsilon_r)
        print("Electrostatic potential at plate 1 in V:", phi_0_1.to('volt').magnitude)
        print("Electrostatic potential at plate 2 in V:", phi_0_2.to('volt').magnitude)
        print("Potential difference between plates in V:", (phi_0_1 - phi_0_2).to('volt').magnitude)

        # Calculate the total electrostatic potential profile
        phi_z_1 = self._phi_z(phi_0_1, l_D, z, z[0])
        phi_z_2 = self._phi_z(phi_0_2, l_D, z, z[-1])
        phi_z_ = phi_z_1 + phi_z_2

        return self._distribution_profile(self._q, self._T, phi_z_)