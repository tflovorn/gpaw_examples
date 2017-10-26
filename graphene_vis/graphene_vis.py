from __future__ import division
import numpy as np
from ase import Atoms
import ase.io
import ase.visualize
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
    graphene.center()
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
            txt="graphene_gs.txt")

    graphene.calc = calc
    graphene.get_potential_energy()

    calc.write(ground_state_file, 'all')

def _main():
    # First step: find the ground-state charge density.
    ground_state_file = "graphene_gs.gpw"
    run_ground_state(ground_state_file)

    # Load ground state data and visualize.
    calc = GPAW(ground_state_file,
            txt="graphene_vis.txt")

    # Output data for charge density corresponding to pseudo-wavefunctions
    # and all-electron wavefunctions.
    # Use xcrysden 'xsf' format.
    # Some documentation on data output from GPAW at:
    # https://wiki.fysik.dtu.dk/gpaw/tutorials/plotting/plot_wave_functions.html
    # https://wiki.fysik.dtu.dk/gpaw/tutorials/all-electron/all_electron_density.html
    # https://wiki.fysik.dtu.dk/gpaw/tutorials/ps2ae/ps2ae.html
    ase.io.write("graphene_pseudo_density.xsf", calc.get_atoms(), data=calc.get_pseudo_density())
    ase.io.write("graphene_ae_density.xsf", calc.get_atoms(), data=calc.get_all_electron_density())

if __name__ == "__main__":
    _main()
