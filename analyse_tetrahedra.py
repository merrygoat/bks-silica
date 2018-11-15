from atooms import postprocessing as pp
from atooms.trajectory import TrajectoryLAMMPS
from atooms import trajectory
import numba
import numpy as np
import math
import itertools
import pickle


@numba.jit(nopython=True)
def square_to_condensed(i, j, n):
    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    return n * j - j * (j + 1) / 2 + i - 1 - j


@numba.jit(nopython=True)
def calc_row_idx(k, n):
    return int(math.ceil((1 / 2.) * (- (-8 * k + 4 * n ** 2 - 4 * n - 7) ** 0.5 + 2 * n - 1) - 1))


@numba.jit(nopython=True)
def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i * (i + 1)) / 2


@numba.jit(nopython=True)
def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)


# @numba.jit(nopython=True):
def condensed_to_square(k, n):
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j


@numba.jit(nopython=True)
def neighbours(pos=np.zeros(0), side=np.zeros(3), N=0, rcut=0.0, max_neigh=20):
    dists = np.zeros(int(N * (N - 1) / 2))
    rcutsq = rcut ** 2
    neigh_table = -np.ones((N, max_neigh))
    num_neigh = np.zeros(N)
    dists_neigh = np.ones((N, max_neigh)) * 1e25
    for i in range(N - 1):
        for j in range(i + 1, N):
            rsq = 0
            for k in range(3):
                d = pos[i][k] - pos[j][k]
                if d > side[k] / 2.:
                    d -= side[k]
                elif d <= -side[k]:
                    d += side[k]
                rsq += d ** 2
            if rsq < rcutsq:
                ni = int(num_neigh[i])
                nj = int(num_neigh[j])
                neigh_table[i, ni] = j
                neigh_table[j, nj] = i
                num_neigh[i] += 1
                num_neigh[j] += 1
                assert num_neigh[i] < max_neigh, "Found too many neigbours too large. Reduce the cutoff."
                dists_neigh[i, ni] = rsq
                dists_neigh[j, nj] = rsq

            dists[square_to_condensed(i, j, N)] = rsq
    # sort neighs
    # order = dists_neigh.argsort(axis=1)
    return dists, neigh_table, dists_neigh


class Tetrahedra(pp.correlation.Correlation):
    def __init__(self, trajectory, grid, cutoff=1.0, max_neigh=2.0, centre_species='1'):
        pp.correlation.Correlation.__init__(self, trajectory, grid, phasespace=["pos", "species"])
        self.cutoff = cutoff
        self.max_neigh = max_neigh
        self.centre_species = centre_species

    def _compute(self):
        ncfg = len(self.trajectory)
        self.tetrahedra = {}
        self.edges = {}
        for cfg in range(0, ncfg):
            current_system = self.trajectory.read(cfg)
            self.side = current_system.cell.side
            N = self._pos[cfg].shape[0]
            dists, neighs, dist_neighs = neighbours(self._pos[cfg], self.side, N, self.cutoff, self.max_neigh)
            # sort neighbour distances

            self.neighs = neighs
            for p in range(N):
                self.neighs[p] = (neighs[p][dist_neighs[p].argsort()])

            self.neighs = self.neighs.astype(int)
            tetrahedra = []
            for i in range(N):
                if current_system.particle[i].species == self.centre_species:
                    assert current_system.particle[i].species == "1", "big error"
                    n = 0
                    tetrahedron = []
                    while self.neighs[i, n] is not -1:
                        neigh_type = current_system.particle[self.neighs[i, n]].species
                        if neigh_type != self.centre_species:
                            assert neigh_type == '2', "%s %s" % (neigh_type, self.centre_species)
                            tetrahedron.append((self.neighs[i, n], self._pos[cfg][i]))
                        if len(tetrahedron) == 4:
                            break
                        n += 1
                    tetrahedra.append(tetrahedron)
            self.tetrahedra[cfg] = tetrahedra
            # write all edges
            edges = []
            for t in tetrahedra:
                for i, j in itertools.combinations(range(4), 2):
                    edges.append((t[i][0], t[j][0]))
            self.edges[cfg] = edges

    def dump(self, cfg, filename, numtypes=2):
        current_system = self.trajectory.read(cfg)
        bonds = self.edges[cfg]
        natoms = len(current_system.particle)
        nbonds = len(bonds)

        with open(filename, 'w') as fw:
            fw.write("LAMMPS Description\n\n")
            block = """{} atoms
        {} bonds
        0 angles
        0 dihedrals 
        0 impropers
        
        """.format(natoms, nbonds)
            fw.write(block)
            block = """{} atom types
        1 bond types
        
        """.format(numtypes)
            fw.write(block)

            los = current_system.cell.origin - current_system.cell.side / 2.
            his = los + current_system.cell.side

            block = """{} {} xlo xhi
        {} {} ylo yhi
        {} {} zlo zhi
        
        """.format(los[0], his[0], los[1], his[1], los[2], his[2])

            fw.write(block)

            fw.write("Masses\n\n 1 1\n2 1\n\n")

            fw.write("Atoms\n\n")
            for i, p in enumerate(current_system.particle):
                fw.write("{} {} {} {} {}\n".format(i, p.species, p.position[0], p.position[1], p.position[2]))

            fw.write("\nBonds\n\n")
            for i, b in enumerate(bonds):
                print(i, b)
                fw.write("{} 1 {} {}\n".format(i, bonds[i][0], bonds[i][1]))

    def dump_vtk(self, cfg, filename):
        """Incomplete method to output vtk-formatted data"""

        system = self.trajectory.read(cfg)
        pos = self._pos[cfg]
        with open(filename, 'w') as fw:
            fw.write("# vtk DataFile Version 2.0\n")
            fw.write("Tetrahedra from configuration %d\n" % cfg)
            fw.write("ASCII\n")
            fw.write("DATASET POLYDATA\n")

            fw.write("POINTS %d double\n" % pos.shape[0])
            for p in pos:
                fw.write("%g %g %g\n" % (p[0], p[1], p[2]))

    def store(self, filename):
        results = {"edges": self.edges, "tetrahedra": self.tetrahedra}
        pickle.dump(results, open(filename, 'wb'))

    def plot(self, cfg):
        import pylab as pl
        import mpl_toolkits.mplot3d as a3
        current_system = self.trajectory.read(cfg)
        side = current_system.cell.side
        axes = a3.Axes3D(pl.figure())
        xyz = self._pos[cfg]
        for t in self.tetrahedra[cfg]:
            corners = np.array([i[0] for i in t])
            tri = np.array(list(itertools.combinations(corners, 3)))
            vts = xyz[tri, :]
            com = vts.mean(axis=1)
            d = [np.linalg.norm(vts[i, :] - com) for i in range(vts.shape[0])]
            if np.any(d > self.cutoff):
                # pick one corner as reference
                for p in range(1, 4):
                    for k in range(3):
                        r = vts[p, k] - vts[0, k]
                        if r > side[k] / 2.:
                            vts[p, k] -= side[k]
                        elif r < -side[k] / 2.:
                            vts[p, k] += side[k]

            tri = a3.art3d.Poly3DCollection(vts)
            tri.set_alpha(0.2)
            tri.set_color('grey')
            axes.add_collection3d(tri)
            axes.set_axis_off()
            axes.set_aspect('equal')





tj = TrajectoryLAMMPS("../DUMPS/selection/p0.0_T4200.0.dump")
ts = trajectory.Sliced(tj,slice(-1,len(tj)))
tetra = Tetrahedra(ts,None,cutoff=4.0,max_neigh=50)
tetra.compute()
#p = tetra._pos[0]
#N = p.shape[0]
#dists,nt,dn = neighbours(p,tetra.side,N,rcut=2., max_neigh=50)

