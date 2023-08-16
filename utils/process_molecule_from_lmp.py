import sys, os
import rdkit
from rdkit import Chem
from rdkit.Chem.Lipinski import RotatableBondSmarts
import glob, subprocess
import os, time
from GL_data import data
import json, pickle, random
import numpy as np
from xyz2mol import xyz2mol, read_xyz_file

def get_rotatable_bond(xyzfile, mol2file, outfile):
    try:
        atoms, charge, coordinates = read_xyz_file(xyzfile, look_for_charge=True)
        mol = xyz2mol(atoms, coordinates, charge=charge)
        assert len(mol) == 1
        mol = mol[0]
        Chem.rdmolfiles.MolToPDBFile(mol, 'LIG.pdb')
        os.system('obabel -ipdb LIG.pdb -O ' + mol2file)
        rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
        rot_atom_pairs = [(x[0]+1, x[1]+1) for x in rot_atom_pairs]
        f = open(outfile,'w')
        f.write('id,atom1,atom2,type\n')
        for i,(j,k) in enumerate(rot_atom_pairs):
            f.write('%d,%d,%d,1\n'%(i+1,j,k))
        f.close()
    except AttributeError:
        print(xyzfile)
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

def split_lmp_data_file(data_file, converted_ligand_input, converted_ligand_data, converted_ligand_rotbond):
    # we need to split ParLigGen's LAMMPS *.lmp data format file into input file and data file which is
    # compatible with the rest of our code (originally written for SwissParam Gromacs-LAMMPS conversion

    def _extract_section(fstream):
        line = fstream.readline()
        line = fstream.readline()
        content = []
        while line:
            words = line.strip().split()
            if len(words) == 0:
                break
            content.append(words)
            line = fstream.readline()
        return content

    def _dump_input_file(input_content, input_filename):
        assert len(input_content) == 1
        assert input_content[0][0] == 'Pair Coeffs'
        pair_coeffs_content = input_content[0][1]
        input_lines = []
        for words in pair_coeffs_content:
            words = ['pair_coeff', words[0], words[0], words[1], words[2]]
            input_lines.append( ' '.join(words) )
        with open(input_filename,'w') as f:
            f.write( '\n'.join(input_lines) )
        return

    def _dump_data_file(title, headers, data_content, data_filename):
        data_lines = []
        data_lines.append( title )
        data_lines += headers
        for i in range(len(data_content)):
            section = data_content[i][0]
            data_lines.append( '' )
            data_lines.append( section )
            data_lines.append( '' )
            n_lines = len(data_content[i][1])
            for j in range(len(data_content[i][1])):
                words = data_content[i][1][j]
                if section in ['Bond Coeffs', 'Angle Coeffs']:
                    words = [words[0], 'harmonic', words[1], words[2]]
                elif section == 'Dihedral Coeffs':
                    opls_coeff = [float(word) for word in words[1:5]]
                    multiharm_coeff = np.array( [0,0,0,0,0], dtype=float )
                    multiharm_coeff += 0.5 * opls_coeff[0] * np.array( [1,1,0,0,0] ) #(1 + cosx)
                    multiharm_coeff += 0.5 * opls_coeff[1] * np.array( [2,0,2,0,0] ) #(1 - cos2x)
                    multiharm_coeff += 0.5 * opls_coeff[2] * np.array( [1,-3,0,4,0] ) #(1 + cos3x)
                    multiharm_coeff += 0.5 * opls_coeff[3] * np.array( [0,0,8,0,-8] ) #(1 - cos4x)
                    multiharm_coeff = [str(coeff) for coeff in multiharm_coeff]
                    words = [words[0], 'multi/harmonic'] + multiharm_coeff
                elif section == 'Improper Coeffs':
                    cvff_coeff = [float(word) for word in words[1:4]]
                    assert (cvff_coeff[1] == -1) and (cvff_coeff[2] == 2)
                    # all cvff so far is of the form (sin(theta))^2, we will approximate it as theta^2 for small theta
                    harm_coeff = np.array( [2*cvff_coeff[0],0] )
                    harm_coeff = [str(coeff) for coeff in harm_coeff]
                    words = [words[0], 'harmonic'] + harm_coeff
                else:
                    pass
                data_lines.append( ' '.join(words) )
        with open(data_filename,'w') as f:
            f.write( '\n'.join(data_lines) )
        return

    def _dump_rotbond_file(data_content, rotbond_filename):
        type_elem_dict = {}
        mass_dict = {1:'H', 7:'Li', 12:'C', 14:'N', 16:'O', 19:'F', 31:'P', 127:'I', 207:'Pb'}
        for i in range(len(data_content)):
            if data_content[i][0] == 'Masses':
                masses_content = data_content[i][1]
                for j in range(len(masses_content)):
                    mass = round( float(masses_content[j][1]) )
                    type_elem_dict[ masses_content[j][0] ] = mass_dict[ mass ]
        xyz_lines = []
        for i in range(len(data_content)):
            if data_content[i][0] == 'Atoms':
                atoms_content = data_content[i][1]
                n_atoms = len(atoms_content)
                xyz_lines.append( str(n_atoms) )
                charge = 0
                name = data_file.split('_')
                if ('plus' in name[-1]) or ('minus' in name[-1]):
                    name = name[-1]
                    assert name.endswith('.lmp')
                    name = name[:-4]
                    if name == 'plus': charge = 1
                    elif name == 'minus': charge = -1
                    elif 'plus' in name: charge = int(name[4:])
                    elif 'minus' in name: charge = -int(name[5:])
                    else: raise Exception('unallowed *.lmp filename : '+data_file)
                xyz_lines.append( 'charge=' + str(charge) )
                for j in range(n_atoms):
                    atom_type = atoms_content[j][2]
                    elem = type_elem_dict[ atom_type ]
                    words = [elem, atoms_content[j][4], atoms_content[j][5], atoms_content[j][6]]
                    xyz_lines.append( ' '.join(words) )
        xyz_file = 'LIG.xyz'
        with open(xyz_file,'w') as f:
            f.write( '\n'.join(xyz_lines) )
        
        get_rotatable_bond('LIG.xyz', 'LIG.mol2', 'LIG.rotbond')
        build_new_rotbond('LIG.mol2', 'LIG.rotbond', 'LIG_converted.input', 'LIG_converted.lmp', 'LIG_converted.rotbond')
        return
                    
    
    with open(data_file, 'r') as f:
        line = f.readline()
        title = line.strip()
        line = f.readline()
        headers = []
        data_content = []
        input_content = []
        while line:
            words = line.strip().split()
            if len(words) == 0:
                pass
            else:
                full_words = ' '.join(words)
                if full_words in ['Masses','Atoms','Bonds','Angles','Dihedrals','Impropers','Velocities',
                                  'Bond Coeffs','Angle Coeffs','Dihedral Coeffs','Improper Coeffs']:
                    content = _extract_section(f)
                    #print('Found', full_words, ':', len(content))
                    data_content.append( [full_words,content] )
                elif full_words in ['Pair Coeffs']:
                    content = _extract_section(f)
                    input_content.append( [full_words,content] )
                else:
                    headers.append(full_words)
            line = f.readline()

    _dump_input_file(input_content, converted_ligand_input)
    _dump_data_file(title, headers, data_content, converted_ligand_data)
    _dump_rotbond_file(data_content, converted_ligand_rotbond)
    
    return

def convert_molecule(mol_lmp, override=None):
    # Caution, this code is exclusively meant for *.lmp format generated by LigParGen
    # bond_style and angle_style = harmonic, dihedral style = opls, and improper_style = cvff
    # In addition to that, the name of the *.lmp file should specify the charge state. Here are examples:
    # ammonium_plus.lmp
    # CO3_minus2.lmp
    # CH3COO_minus.lmp
    # benzene-1,2-diaminium_plus2.lmp
    mol_data = check_cache(mol_lmp)

    if mol_data is None:
        temp_dir = 'workspace'
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
        cwd = os.getcwd()
        os.system('cp '+mol_lmp+' '+temp_dir)
        os.chdir(temp_dir)

        full_mol_lmp_path = mol_lmp
        mol_lmp = os.path.split(mol_lmp)[-1]

        split_lmp_data_file(mol_lmp, 'LIG_converted.input', 'LIG_converted.lmp', 'LIG_converted.rotbond')
        mol_data = data('LIG_converted.input', 'LIG_converted.lmp', 'LIG_converted.rotbond')

        cleanup_workspace()

        os.chdir(cwd)
        store_cache(full_mol_lmp_path, mol_data)
        
    return mol_data
    
#convert_molecule('input/CO3_minus2.lmp')
