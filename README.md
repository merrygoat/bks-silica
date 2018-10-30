* Simulations of BKS silica in lammps
 
Silica can be simulated using the BKS model. The problem with this is that the Coulombic interactions are very long ranged. LAMMPS uses reciprocal space fixes to compute the long range interactions, but even then the simulations are still very slow.
 
As an alternative the Wolf summation method can be used via the coul/wolf style. This is a means of getting effective long-range interactions with a short-range potential by smoothly truncating the potential. This is done by Berthier and Kob in https://arxiv.org/abs/0707.0319 