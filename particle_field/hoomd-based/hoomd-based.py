
import hoomd
import hoomd.md
hoomd.context.initialize("");

hoomd.init.create_lattice(unitcell=hoomd.lattice.sc(a=2.0), n=5)