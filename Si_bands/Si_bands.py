from __future__ import division
from ase import Atoms
from ase.dft.kpoints import special_paths
from gpaw import GPAW, PW, FermiDirac

def make_Si_cell():
    """Construct the ASE Atoms object for the Si lattice.
    """
    # Silicon has a diamond lattice, which is an fcc Bravais lattice with two atoms.
    # Lattice constant, Angstrom (cubic axis).
    a = 5.4307 
    # fcc lattice vectors, in convention of [SC10]:
    # Setyawan and Curtarolo, Computational Materials Science 49, 299 (2010).
    cell = [[0.0, a/2, a/2],
               [a/2, 0.0, a/2],
               [a/2, a/2, 0.0]]
    # Atom positions within cell, in lattice coordinates.
    pos = [[0.0, 0.0, 0.0],
           [1/4, 1/4, 1/4]]
    # Use periodic boundary conditions in all directions.
    pbc = [True, True, True]

    si = Atoms(['Si', 'Si'], cell=cell, scaled_positions=pos, pbc=pbc)
    return si

# Silicon band structure calculation.
# Based on the example at:
# https://wiki.fysik.dtu.dk/gpaw/tutorials/bandstructures/bandstructures.html#bandstructures
def _main():
    si = make_Si_cell()

    # First step: find the ground-state charge density.
    # Use plane-wave basis with 200 eV wavefunction cutoff energy.
    # Cutoff energy should be chosen such that the error introduced by the incompleteness
    # of the basis is not too high.
    # The appropriate energy is different for each atom.
    # See here: https://wiki.fysik.dtu.dk/gpaw/setups/Si.html
    calc = GPAW(mode=PW(200), 
            # PBE exchange-correlation functional
            xc='PBE',
            # Sample the Brillouin zone with 8 k-points in each reciprocal lattice direction
            kpts=(8, 8, 8),
            # Smear step functions appearing in Brillouin zone integrals using Fermi-Dirac
            # distribution at temperature k_B T = 0.01 eV.
            occupations=FermiDirac(0.01),
            # Start calculation using random wavefunctions.
            # We have a physically-reasonable initial value for the charge density based
            # on the atomic positions.
            # The Hamiltonian is diagonalized iteratively, so we also need an initial guess for
            # the wavefunctions. Random starting wavefunctions generally work well.
            random=True,
            # Set the log file for the ground-state DFT calculation.
            txt="Si_gs.txt")

    si.calc = calc
    # To make GPAW perform the DFT calculation, we need to ask it for something
    # that it needs to perform the calculation to know.
    si.get_potential_energy()
    # Save the result for the ground state so we can load it later.
    ground_state_file = "Si_gs.gpw"
    calc.write(ground_state_file)

    # Second step: using the converged ground-state charge density, calculate the
    # wavefunctions along a high-symmetry path in the Brilllouin zone.
    # The energies along this path give the band structure diagram.
    calc = GPAW(ground_state_file,
            # Include the 16 lowest-energy valence bands in the calculation.
            nbands=16,
            # Keep the charge density constant, using our converged ground-state density.
            fixdensity=True,
            # Don't consider symmetry in the k-point distribution: we want to use exactly
            # the k-points we specify.
            # (In the ground-state calculation, many k-points can be eliminated due to
            # being redundant by symmetry.)
            symmetry='off',
            # Use a high-symmetry k-point path.
            # High-symmetry paths following the [SC10] convention for band structure are
            # built-in (can use these if our unit cell also follows the [SC10] convention).
            # Use 300 k-points total.
            kpts={'path': special_paths['fcc'], 'npoints': 300},
            # Require that the energies of the lowest 8 bands are well-converged.
            convergence={'bands': 8},
            # Set the log file for the band-structure calculation.
            txt="Si_bands.txt")

    # Make GPAW perform the band-structure calculation.
    calc.get_potential_energy()

    # Plot the band structure.
    # Since we used kpoints={'path': ...}, we can use an ASE convenience function for this.
    bs = calc.band_structure()
    bs.plot(filename="Si_bands.png")

if __name__ == "__main__":
    _main()
