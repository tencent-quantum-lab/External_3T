import sys
sys.path.append('./utils/')
sys.path.append('./utils/Convert_Gromacs_LAMMPS/')
from run_utils import parse_config
from run_utils import create_model
from run_utils import create_optimizers
from run_utils import run_model
import torch
import random
import os
import json

def main(config_json):
    blocks = parse_config(config_json)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = None
    for block in blocks:
        model = create_model(block, base_model=model).to(device)
        optimizers = create_optimizers(model, block)
        run_model(model, optimizers, block)

for repeat in range(10):
    # Read template
    tag = 'BigCube_MoveLi_EC_DMC_VC_PF6'
    template = open('configs/'+tag+'_template.json','r').read()

    # Prepare random config file
    n_Li = str(random.randrange(2,9))
    n_VC = str(random.randrange(1,4))
    n_PF6 = str(random.randrange(1,3))
    config_str = template.replace('$$Li$$', n_Li)
    config_str = config_str.replace('$$VC$$', n_VC)
    config_str = config_str.replace('$$PF6$$', n_PF6)
    config_dir = 'configs/'+tag
    if not os.path.isdir(config_dir):
        os.mkdir(config_dir)
    duplicate = True
    while duplicate:
        config_json = config_dir + '/' + tag + '_' + hex(random.randrange(0, sys.maxsize)) + '.json'
        if not os.path.isfile(config_json):
            duplicate = False
    with open(config_json,'w') as f:
        f.write(config_str)

    # Execute
    main(config_json)

    # Store results
    result_dir = 'results/' + tag + '/'
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    result_dir = result_dir + config_json[:-5].split('_')[-1] + '/'
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    os.system('mv FF_step* ' + result_dir)
    os.system('mv VASP_step* ' + result_dir)
    os.system('mv default.log ' + result_dir)
    os.system('mv VASP_files ' + result_dir)
    os.system('cp ' + config_json + ' ' + result_dir)
    json_str = json.loads(config_str)
    poscar_file = json_str[0]['lattice_poscar']['file']
    os.system('cp ' + poscar_file + ' ' + result_dir)
