import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'serif',"font.serif" : "Times New Roman", "text.usetex": True})

### Arguments ###

parser = argparse.ArgumentParser()
parser.add_argument('top', type=str, help='tpr file')
parser.add_argument('traj', type=str, help='trajectory file')
parser.add_argument('first', type=int, help='First frame starts at 0')
parser.add_argument('last', type=int, help='Last frame inclusive')
parser.add_argument('step', type=int, help='main selection')
parser.add_argument('first_res', type=int, help='main selection')
parser.add_argument('last_res', type=int, help='main selection')
parser.add_argument('out', type=str, help='output file')
args = parser.parse_args()

def names(selection,num_atoms):
	'''
	Function to get the residue names and store them in lists for a given selection
	'''
	sel_names = []
	for i in range(num_atoms):
		sel_names.append(selection.atoms[i].resname)
	return(sel_names)

def animation(min_frame,max_frame,phi_traj,psi_traj,step,start):
	num_res = np.shape(phi_traj)[1]
	cmap = plt.cm.rainbow(np.linspace(0, 1,num_res))
	for frame in tqdm(range(min_frame,max_frame)):
		fig = plt.figure(figsize=(6,6))
		for i in range(num_res):
			plt.scatter(phi_traj[:frame+1,i],psi_traj[:frame+1,i],alpha=0.5,color=cmap[i],s=10)
			plt.plot(phi_traj[:frame+1,i],psi_traj[:frame+1,i],alpha=0.1,color=cmap[i])
			plt.text(100,190,f'{(frame*step+start):8.3f}ns')
		plt.xlim(-180,180)
		plt.ylim(-180,180)
		plt.xticks(np.arange(-180,180+60,60))
		plt.yticks(np.arange(-180,180+60,60))
		plt.xlabel(r'$\phi$ ($^\circ$)')
		plt.ylabel(r'$\psi$ ($^\circ$)')
		plt.savefig(f'{args.out}_{frame:05d}.png',bbox_inches="tight")
		plt.close()

def run():
	if args.top.endswith('.tpr'):
		pass
	else:
		raise ValueError("Extension for topology must be .tpr")
	
	print(f'MDA version: {mda.__version__}')

	u = mda.Universe(args.top,args.traj) 
	dt = (u.trajectory[1].time-u.trajectory[0].time)*1e-3

	print(f'The calculated time step is:\t\t\t{dt:8.4f} ns')

	print(len(u.residues))

	sele_phi = [res.phi_selection() for res in u.residues[args.first_res:args.last_res+1]]
	sele_psi = [res.psi_selection() for res in u.residues[args.first_res:args.last_res+1]]
	
	R_phi = Dihedral(sele_phi).run(args.first,args.last+1,args.step).results.angles
	R_psi = Dihedral(sele_psi).run(args.first,args.last+1,args.step).results.angles

	frames = np.shape(R_phi)[0]
	print(frames)
	cores = multiprocessing.cpu_count()
	divide_conquer = frames//cores
	print(divide_conquer)

	pool = multiprocessing.Pool(cores)
	partial_animate=partial(animation, phi_traj=R_phi,psi_traj=R_psi,step=dt*args.step,start=args.first*dt)

	min_frames = []
	max_frames = []
	for i in range(cores):
		min_frames.append(i*divide_conquer)
		if i == cores-1:
			max_frames.append(frames)
		else:
			max_frames.append((i+1)*divide_conquer)

	pool.starmap(partial_animate,zip(min_frames,max_frames))
	#animation(R_phi.results.angles,R_psi.results.angles,dt*args.step,args.first*dt)

if __name__ == '__main__':
	run()