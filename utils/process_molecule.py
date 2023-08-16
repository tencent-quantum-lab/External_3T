import pymol
from pymol import cmd
import sys, os
import rdkit
from rdkit import Chem
from rdkit.Chem.Lipinski import RotatableBondSmarts
import glob, subprocess
import os, time
import ase
import ase.io as sio
from swiss_util import swiss_func
from charmm2gromacs_util import chm2gmx_func
from GL_data import data
import json, pickle, random
import numpy as np

def prelig4swiss(infile, outfile):
    outfile_0 = infile.replace('mol2', 'mol2')
    cmd.load(infile, 'MOL')
    cmd.remove('hydrogens')
    cmd.h_add('MOL')
    cmd.save(outfile_0, 'MOL', format='mol2')
    
    MOL_list = [x for x in open(outfile_0,'r')]
    idx = [i for i,x in enumerate(MOL_list) if x.startswith('@')]
    block = MOL_list[idx[1]+1:idx[2]]
    block = [x.split('\t') for x in block]

    block_new = []
    atom_count = {'C':1, 'N':1, 'O':1, 'S':1, 'P':1, 'F':1, 'Br':1, 'Cl':1, 'I': 1,   
                  'Li':1, 'Na':1, 'K':1, 'Mg':1, 'Al':1, 'Si':1,
                  'Ca':1, 'Cr':1, 'Mn':1, 'Fe':1, 'Co':1, 'Cu':1}
    for i in block:
        at = i[1].strip()
        if 'H' not in at:
            count = atom_count[at]
            atom_count[at]+=1
            at_new = at+str(count)
            at_new = at_new.rjust(4)
            block_new.append([i[0], at_new]+i[2:])
        else:
            block_new.append(i)

    block_new = ['\t'.join(x) for x in block_new]
    MOL_list_new = MOL_list[:idx[1]+1]+block_new+MOL_list[idx[2]:]
    f = open(outfile,'w')
    for i in MOL_list_new:
        f.write(i)
    f.close()
  
def get_rotatable_bond(infile, outfile):
    try:
        mol = Chem.rdmolfiles.MolFromMol2File(infile, removeHs=False)
        rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
        rot_atom_pairs = [(x[0]+1, x[1]+1) for x in rot_atom_pairs]
        f = open(outfile,'w')
        f.write('id,atom1,atom2,type\n')
        for i,(j,k) in enumerate(rot_atom_pairs):
            f.write('%d,%d,%d,1\n'%(i+1,j,k))
        f.close()
    except AttributeError:
        print(infile)
    return

def extract_cgenff(lig_mol2, dl_txt, lig_ori, lig_zip, reply_txt):
    os.system(' '.join(['cp',lig_mol2,lig_ori]))
    dl_link = swiss_func(lig_mol2)
    time.sleep(5)
    start_time = time.time()
    while True:
        out = os.system(' '.join(['wget --no-check-certificate',dl_link,'2>',reply_txt]))
        if out == 0: break
        time.sleep(5)
        current_time = time.time()
        if current_time - start_time > 600:
            # 10 minutes without server success, we should terminate this molecule
            raise Exception('SwissParam server mol2 conversion fail')
    with open(reply_txt,'w') as f:
        f.write('y\n')
    os.system(' '.join(['unzip',lig_zip,'<',reply_txt]))
    return

def charmm2gmx(lig_itp, lig_par, lig_prm, lig_ff_folder, lig_bonded):
    chm2gmx_func(lig_itp,lig_par,lig_ff_folder)
    #os.system(' '.join(['python py3charmm2gromacs-pvm.py',lig_itp,lig_par,lig_ff_folder]))
    out_lines = []
    with open(os.path.join(lig_ff_folder,'forcefield.itp'),'r') as f:
        line = f.readline()
        while line:
            words = line.strip().split()
            if len(words)==0:
                out_lines.append(line)
            elif not words[0]=='#include':
                out_lines.append(line)
            line = f.readline()
        out_lines.append('#include '+lig_bonded+'\n')
    with open(lig_prm,'w') as f:
        for line in out_lines:
            f.write(line)
    os.system(' '.join(['mv', os.path.join(lig_ff_folder,'ffbonded.itp'), lig_bonded]))
    return

def build_gro(lig_pdb, lig_gro):
    os.system(' '.join(['gmx editconf -f',lig_pdb,'-o',lig_gro]))
    return

def build_new_rotbond(lig_ori, lig_rotbond, converted_ligand_input, converted_ligand_data, converted_ligand_rotbond):
    lig_data = data(converted_ligand_input, converted_ligand_data)
    new_lig_pos = lig_data.atom_pos
    old_lig_pos = []
    with open(lig_ori, 'r') as f:
        line = f.readline()
        while line:
            words = line.strip().split()
            if len(words)==9:
                old_lig_pos.append([float(i) for i in words[2:5]])
            line = f.readline()
        old_lig_pos = np.array(old_lig_pos)
        if not old_lig_pos.shape[0] == new_lig_pos.shape[0]:
            raise Exception('Unmatched ligand atom count for rotatable bond rearrangement')
    n_atoms = old_lig_pos.shape[0]
    all_dist = np.linalg.norm(np.repeat(old_lig_pos[np.newaxis,:,:], n_atoms, axis=0) -\
                              np.repeat(new_lig_pos[:,np.newaxis,:], n_atoms, axis=1), axis=2)
    old_to_new = np.argmin(all_dist, axis=0)
    content = []
    with open(lig_rotbond,'r') as f:
        content.append(f.readline())
        line = f.readline()
        while line:
            words = line.split(',')
            words[1] = str(old_to_new[ int(words[1])-1 ] + 1)
            words[2] = str(old_to_new[ int(words[2])-1 ] + 1)
            content.append( ','.join(words) )
            line = f.readline()
    with open(converted_ligand_rotbond,'w') as f:
        for line in content:
            f.write(line)
    return

def convert_gromacs_lammps_ligand(lig_itp, lig_prm, lig_gro, lig_top):
    with open(lig_itp,'r') as f:
        content = f.read()
    with open(lig_top,'w') as f:
        f.write('; Include ligand parameters' + '\n' +\
                   '#include "' + lig_prm + '"\n\n')
        f.write(content)
        f.write('\n' +\
                   '[ molecules ]' +'\n' +\
                   '; Compound        #mols' + '\n' +\
                   'LIG                       1' + '\n')
    cwd = os.getcwd()
    lig_gro_full = os.path.join(cwd, lig_gro)
    lig_top_full = os.path.join(cwd, lig_top)
    os.chdir('../utils/Convert_Gromacs_LAMMPS')
    os.system(' '.join(['python Convert_Gromacs_LAMMPS.py',
                        lig_gro_full, lig_top_full, cwd]))
    os.chdir(cwd)
    return

def cleanup_workspace():
    lig_ff = 'LIG_ff'
    if os.path.isdir(lig_ff):
        os.system('rm -r '+lig_ff)
    files = os.listdir()
    for file in files:
        os.system('rm '+file)
    return

def check_cache(mol_xyz):
    if not os.path.isdir('cache'): os.mkdir('cache')
    cache_index = 'cache/index.json'
    if not os.path.isfile(cache_index):
        return None
    index = json.load(open(cache_index,'r'))
    if mol_xyz in index['xyz_to_pkl']:
        mol_pkl = os.path.join('cache', index['xyz_to_pkl'][mol_xyz])
        pkl_out = pickle.load(open(mol_pkl, 'rb'))
        return pkl_out
    else:
        return None

def store_cache(mol_xyz, mol_data):
    cache_index = 'cache/index.json'
    if not os.path.isfile(cache_index):
        index = {'xyz_to_pkl':{}, 'pkl_ids':[]}
    else:
        index = json.load(open(cache_index,'r'))
    pkl_ids = index['pkl_ids']
    xyz_to_pkl = index['xyz_to_pkl']
    if mol_xyz in xyz_to_pkl:
        # update the cache, although this will not be why we usually call this function
        old_pkl = os.path.join('cache', xyz_to_pkl[mol_xyz])
        pkl_ids.remove(xyz_to_pkl[mol_xyz])
        os.system('rm '+old_pkl)
        del xyz_to_pkl[mol_xyz]
    duplicate = True
    while duplicate:
        pkl_id = str(random.randint(0, sys.maxsize))+'.pkl'
        if not pkl_id in pkl_ids:
            duplicate = False
    pkl_ids.append(pkl_id)
    xyz_to_pkl[mol_xyz] = pkl_id
    pickle.dump(mol_data, open( os.path.join('cache', pkl_id), 'wb' ))
    json.dump(index, open(cache_index,'w'))
    return    

def convert_molecule(mol_xyz, override=None):
    mol_data = check_cache(mol_xyz)

    if mol_data is None:
        temp_dir = 'workspace'
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
        cwd = os.getcwd()
        os.system('cp '+mol_xyz+' '+temp_dir)
        os.chdir(temp_dir)

        full_mol_xyz_path = mol_xyz
        mol_xyz = os.path.split(mol_xyz)[-1]
        mol = sio.read(mol_xyz, format='xyz')
        temp_mol2, lig_mol2 = 'temp.mol2', 'LIG.mol2'
        os.system('obabel '+mol_xyz+' -O '+temp_mol2)
        prelig4swiss(temp_mol2, lig_mol2)
        get_rotatable_bond(lig_mol2, 'LIG.rotbond')

        extract_cgenff(lig_mol2, 'download.txt', 'LIG_ori.mol2', 'LIG.zip', 'reply.txt')

        charmm2gmx('LIG.itp', 'LIG.par', 'LIG.prm', 'LIG_ff', 'LIG_bonded.itp')

        build_gro('LIG.pdb', 'LIG.gro')

        convert_gromacs_lammps_ligand('LIG.itp', 'LIG.prm', 'LIG.gro', 'LIG.top')

        build_new_rotbond('LIG_ori.mol2', 'LIG.rotbond', 'LIG_converted.input', 'LIG_converted.lmp', 'LIG_converted.rotbond')

        mol_data = data('LIG_converted.input', 'LIG_converted.lmp', 'LIG_converted.rotbond')

        cleanup_workspace()

        os.chdir(cwd)
        store_cache(full_mol_xyz_path, mol_data)
        
    return mol_data
    
