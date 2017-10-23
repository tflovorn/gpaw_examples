from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.dft.stm import STM
from gpaw import GPAW, PW, FermiDirac

def make_graphene_cell():
    """Construct the ASE Atoms object for the graphene lattice.
    """
    # Graphene has a triangular Bravais lattice, with two atoms per unit cell,
    # together forming a honeycomb lattice.
    # Lattice constant, Angstrom.
    a = 2.46
    # We will include 20 Angstrom of vacuum in the calculation surrounding
    # the graphene sheet.
    vac = 20.0
    # Lattice vectors.
    cell = [[a/2, -np.sqrt(3)*a/2, 0.0],
            [a/2, np.sqrt(3)*a/2, 0.0],
            [0.0, 0.0, vac]]
    # Atom positions within cell, in lattice coordinates.
    pos = [[0.0, 0.0, 1/2],
           [1/3, 2/3, 1/2]]
    # Use periodic boundary conditions in the plane of the graphene, but
    # not perpendicular to it.
    pbc = [True, True, False]

    graphene = Atoms(['C', 'C'], cell=cell, scaled_positions=pos, pbc=pbc)
    return graphene

def run_ground_state(ground_state_file):
    graphene = make_graphene_cell()

    # First step: find the ground-state charge density.
    # Use plane-wave basis with 600 eV cutoff energy for wavefunction variation in the
    # plane of the graphene sheet. https://wiki.fysik.dtu.dk/gpaw/setups/C.html
    calc = GPAW(mode=PW(600),
            xc='PBE',
            # 0.2 Angstrom real-space basis grid spacing perpendicular to the plane
            h=0.2,
            # Sample the Brillouin zone with 25 k-points in each reciprocal lattice direction
            # (only the in-plane directions). For the triangular lattice, we need to
            # be sure to include the Gamma point k = (0, 0, 0) to have the k-point sampling
            # match the symmetry of the Brillouin zone.
            kpts={'size': (25, 25, 1), 'gamma': True},
            occupations=FermiDirac(0.01),
            random=True,
            symmetry='off',
            txt="graphene_gs.txt")

    graphene.calc = calc
    graphene.get_potential_energy()

    calc.write(ground_state_file, 'all')

def plot_stm(ground_state_file, bias, z):
    calc = GPAW(ground_state_file,
            txt="graphene_stm.txt")
    stm = STM(calc.get_atoms())
    # Get the average constant-height (height = z) current for a reasonable value
    # of the current to use when scanning.
    c = stm.get_averaged_current(bias, z)
    # Scan over the unit cell at constant current.
    # repeat=(3, 5) --> repeat over 3 copies of unit cell in one direction, 5 in another
    x, y, h = stm.scan(bias, c, repeat=(3, 5))

    plt.gca(aspect='equal')
    plt.contourf(x, y, h, 80) # 80 contour levels
    plt.hot()
    plt.colorbar()
    plt.savefig("graphene_stm_bias_{:.2f}_z_{:.2f}.png".format(bias, z), bbox_inches='tight', dpi=500)
    plt.clf()

# Graphene STM simulation example.
# Based on example at:
# https://wiki.fysik.dtu.dk/gpaw/tutorials/stm/stm.html
def _main():
    # First step: find the ground-state charge density.
    ground_state_file = "graphene_gs.gpw"
    run_ground_state(ground_state_file)

    # Second step: calculate STM image.
    # Bias values from -1.5 to 1.5 V.
    biases = [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]
    # C atoms are at z = vac/2 = 10.0 A.
    # Choose tip separation from 0.1 to 2.0 A.
    zs = [10.1, 10.25, 10.5, 11.0, 11.5, 12.0, 13.0]
    for bias in biases:
        for z in zs:
            plot_stm(ground_state_file, bias, z)

if __name__ == "__main__":
    _main()
