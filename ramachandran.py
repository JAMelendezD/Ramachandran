import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
import argparse
from numba import jit
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
parser.add_argument('--mode', default=0,required=False, type=int, help='Mode options, 0 or 1')
args = parser.parse_args()

def pre_process(frames,num_res,array_):
	for frame in range(frames):
		for i in range(num_res):
			if frame < frames-1:
				delta = np.abs(array_[frame+1:frame+2,i]-array_[frame:frame+1,i])
				if delta[0] > 200:
					if array_[frame:frame+1,i] < 0:
						array_[frame+1:frame+2,i] = -180
					else:
						array_[frame+1:frame+2,i] = 180
	return(array_)

def fix_boundary(point,cutoff):
	new_point = [point[0],0]
	extra = [0,point[1]]
	delta = abs(point[1]-point[0])
	if delta > cutoff:
		if point[0] > 0:
			new_point[1] = 180
			extra[0] = -180
		else:
			new_point[1] = -180
			extra[0] = 180
		return(new_point,extra)
	else:
		return(None,None)

def animate0(min_frame,max_frame,phi_traj,psi_traj,step,start,cutoff):
	num_res = np.shape(phi_traj)[1]
	cmap = plt.cm.rainbow(np.linspace(0, 1,num_res))
	fig = plt.figure(figsize=(6,6))
	for frame in tqdm(range(min_frame,max_frame),colour='green',desc='Frames'):
		for i in range(num_res):
			point_x = [phi_traj[frame:frame+1,i][0],phi_traj[frame+1:frame+2,i][0]]
			point_y = [psi_traj[frame:frame+1,i][0],psi_traj[frame+1:frame+2,i][0]]

			new_point_x,extra_x = fix_boundary(point_x,cutoff)
			new_point_y,extra_y = fix_boundary(point_y,cutoff)

			if new_point_x == None and new_point_y == None:
				plt.plot(point_x,point_y,alpha=0.4,color=cmap[i],zorder=-1)
			elif new_point_x == None and new_point_y != None:
				plt.plot(point_x,new_point_y,alpha=0.4,color=cmap[i],zorder=-1)
				plt.plot(point_x,extra_y,alpha=0.4,color=cmap[i],zorder=-1)
			elif new_point_x != None and new_point_y == None:
				plt.plot(new_point_x,point_y,alpha=0.4,color=cmap[i],zorder=-1)
				plt.plot(extra_x,point_y,alpha=0.4,color=cmap[i],zorder=-1)
			else:
				plt.plot(new_point_x,new_point_y,alpha=0.4,color=cmap[i],zorder=-1)
				plt.plot(extra_x,extra_y,alpha=0.4,color=cmap[i],zorder=-1)			
			plt.scatter(point_x,point_y,alpha=0.4,color=cmap[i],s=10,zorder=3)
			plt.title(f'{(frame*step+start):8.3f}ns')
		plt.xlim(-180,180)
		plt.ylim(-180,180)
		plt.xticks(np.arange(-180,180+60,60))
		plt.yticks(np.arange(-180,180+60,60))
		plt.xlabel(r'$\phi$ ($^\circ$)')
		plt.ylabel(r'$\psi$ ($^\circ$)')
		plt.savefig(f'{args.out}_{frame:05d}.png',bbox_inches="tight")
		plt.title('')

def animate1(min_frame,max_frame,phi_traj,psi_traj,phi_traj_p,psi_traj_p,step,start):
	num_res = np.shape(phi_traj)[1]
	cmap = plt.cm.rainbow(np.linspace(0, 1,num_res))
	for frame in tqdm(range(min_frame,max_frame),colour='green',desc='Frames'):
		fig = plt.figure(figsize=(6,6))
		for i in range(num_res):
			plt.plot(phi_traj_p[:frame+1,i],psi_traj_p[:frame+1,i],alpha=0.4,color=cmap[i],zorder=-1)
			plt.scatter(phi_traj[:frame+1,i],psi_traj[:frame+1,i],alpha=0.4,color=cmap[i],s=10,zorder=3)
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
	print(f'MDA version: {mda.__version__}')

	u = mda.Universe(args.top,args.traj) 
	dt = (u.trajectory[1].time-u.trajectory[0].time)*1e-3

	print(f'The calculated time step is:\t\t\t\t{dt:8.4f} ns')

	sele_phi = [res.phi_selection() for res in u.residues[args.first_res:args.last_res+1]]
	sele_psi = [res.psi_selection() for res in u.residues[args.first_res:args.last_res+1]]
	
	print('Calculating Psi and Phi dihedrals')

	R_phi = Dihedral(sele_phi).run(args.first,args.last+1,args.step).results.angles
	R_psi = Dihedral(sele_psi).run(args.first,args.last+1,args.step).results.angles

	print('Finished Psi and Phi dihedrals calculation')

	frames = np.shape(R_phi)[0]
	num_res = np.shape(R_phi)[1]
	print(f'The number of frames are:\t\t\t\t{frames:8d}')
	print(f'The number of residues are:\t\t\t\t{num_res:8d}')
	print(f'The first and last resnames:\t\t\t\t{sele_psi[0].resnames[0]}  {sele_psi[-1].resnames[-1]}')
	cores = multiprocessing.cpu_count()
	print(f'Will use all available cores to generate images:\t{cores:8d}')
	divide_conquer = int(np.ceil(frames/cores))
		
	if args.mode == 0:
		partial_animate=partial(animate0, phi_traj=R_phi,psi_traj=R_psi,step=dt*args.step,start=args.first*dt,cutoff=280)
		partial_animate(0,frames)
	elif args.mode == 1:
		R_phi_p = pre_process(frames,num_res,np.copy(R_phi))
		R_psi_p = pre_process(frames,num_res,np.copy(R_psi))
		partial_animate=partial(animate1, phi_traj=R_phi,psi_traj=R_psi,phi_traj_p=R_phi_p,psi_traj_p=R_psi_p,step=dt*args.step,start=args.first*dt)
		pool = multiprocessing.Pool(cores)

		min_frames = []
		max_frames = []
		for i in range(cores):
			min_frames.append(i*divide_conquer)
			if i == cores-1:
				max_frames.append(frames)
			else:
				max_frames.append((i+1)*divide_conquer)

		pool.starmap(partial_animate,zip(min_frames,max_frames))

if __name__ == '__main__':
	run()