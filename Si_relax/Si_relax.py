from __future__ import division
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from gpaw import GPAW, PW, FermiDirac

def make_displaced_Si_cell():
    """Construct the ASE Atoms object for the Si lattice, with one atom
    displaced from its equilibrium position.
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
           [1/5, 1/5, 1/5]] # compare to equilibrium position = [1/4, 1/4, 1/4]
    # Use periodic boundary conditions in all directions.
    pbc = [True, True, True]

    si = Atoms(['Si', 'Si'], cell=cell, scaled_positions=pos, pbc=pbc)

    # We will keep the atom at [0.0, 0.0, 0.0] fixed and move the other
    # one toward the equilibrium position.
    c = FixAtoms(indices=[0])
    si.set_constraint(c)

    return si

# Silicon relaxation.
def _main():
    si = make_displaced_Si_cell()

    # Set up GPAW calculator for the ground-state charge density.
    # This will be used repeatedly as the structure is moved toward the
    # equilibrium structure.
    calc = GPAW(mode=PW(200), # plane-wave basis, 200 eV wavefunction cutoff energy
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
            txt="Si_relax.txt")

    si.calc = calc

    opt = QuasiNewton(si,
            # Set the (non-plaintext) log file for the structural optimization steps.
            trajectory="Si.traj")

    # Relax structure until all forces are less than 0.05 eV/Angstrom.
    opt.run(fmax=0.05)

if __name__ == "__main__":
    _main()
