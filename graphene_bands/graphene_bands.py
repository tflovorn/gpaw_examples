from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.dft.kpoints import bandpath
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

def _main():
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
    E_Fermi = calc.get_fermi_level()

    ground_state_file = "graphene_gs.gpw"
    calc.write(ground_state_file)

    # Second step: calculate band structure along high-symmetry lines.
    # High-symmetry path: Gamma -> K -> M -> Gamma.
    Gamma = [0.0, 0.0, 0.0]
    K = [1/3, 1/3, 0.0]
    M = [1/2, 0.0, 0.0]
    band_ks, band_xs, band_special_xs = bandpath([Gamma, K, M, Gamma], graphene.cell, 100)

    calc = GPAW(ground_state_file,
            # Calculation does not include spin, so with nbands=10 we effectively
            # include 20 states.
            # 8 valence electrons are present (2 s + 2 p electrons from each C).
            nbands=10,
            fixdensity=True,
            symmetry='off',
            kpts=band_ks,
            convergence={'bands': 5},
            txt="graphene_bands.txt")

    calc.get_potential_energy()

    # Collect band energies E[k_index][band_index]
    E_kn = np.array([calc.get_eigenvalues(k_index) for k_index in range(len(band_ks))])
    # Shift band energies so that E = 0 is at the Fermi level.
    E_kn -= E_Fermi

    # Plot band structure.
    plt.xticks(band_special_xs, ["$\\Gamma$", "$K$", "$M$", "$\\Gamma$"])
    plt.xlim(0.0, band_xs[-1])

    plt.axhline(0.0, color='k')
    for x in band_special_xs:
        plt.axvline(x, color='k')

    E_nk = E_kn.T # E_nk = E[band_index][k_index]
    for E_n in E_nk:
        plt.plot(band_xs, E_n, 'b-')

    plt.savefig("graphene_bands.png", bbox_inches='tight', dpi=500)

if __name__ == "__main__":
    _main()
