import numpy as np
from scipy.constants import N_A, epsilon_0, k as kB, e as e_el
from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity

N_A       *= Q_('1/mol')
epsilon_0 *= Q_('F/m')
e_el      *= Q_('C')
kB        *= Q_('J/K')


def PBTwoPlates2(
    sigma_1: float,
    sigma_2: float,
    c_0: float,
    T: float,
    epsilon_r: float,
    q: float,
    n_points: int = 1000,
) -> np.ndarray:
    """
    Function to generate Poisson-Boltzmann distribution profiles for cations and anions
    between two charged plates.
    """


    def _lambda_D(epsilon_r, T, c_0):
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


    def _phi_0(sigma, lambda_D, epsilon_r):
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
    

    def _phi_z(phi_0, z, z_0):
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

        return phi_0 * np.exp(-np.abs(z - z_0))


    def _distribution_profile(q, T, phi_z):
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
    

    # Convert inputs to pint quantities
    _sigma_1 = Q_(sigma_1, 'elementary_charge/angstrom^2')
    _sigma_2 = Q_(sigma_2, 'elementary_charge/angstrom^2')
    _c_0 = Q_(c_0, 'mol/l')
    _T = Q_(T, 'kelvin')
    _epsilon_r = Q_(epsilon_r, 'dimensionless')
    _q = Q_(q, 'elementary_charge')

    l_D = _lambda_D(_epsilon_r, _T, _c_0)
    print("Debye length in angstrom:", l_D.to('angstrom').magnitude)
    phi_0_1 = _phi_0(_sigma_1, l_D, _epsilon_r)
    phi_0_2 = _phi_0(_sigma_2, l_D, _epsilon_r)
    print("Electrostatic potential at plate 1 in V:", phi_0_1.to('volt').magnitude)
    print("Electrostatic potential at plate 2 in V:", phi_0_2.to('volt').magnitude)

    # Calculate the total electrostatic potential profile
    zeta = np.linspace(0, 1/l_D.magnitude, n_points)
    phi_z_1 = _phi_z(phi_0_1, zeta, zeta[0])
    phi_z_2 = _phi_z(phi_0_2, zeta, zeta[-1])
    phi_z_ = phi_z_1 + phi_z_2
    return _distribution_profile(_q, _T, phi_z_)