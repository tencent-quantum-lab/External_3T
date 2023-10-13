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
import traceback
from datetime import datetime

def main(config_json):
    blocks = parse_config(config_json)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = None
    for block in blocks:
        model = create_model(block, base_model=model).to(device)
        optimizers = create_optimizers(model, block)
        run_model(model, optimizers, block)
    return

def multiple_runs(run_num, tag, config_modify_func):
    for repeat in range(run_num):
        # Read template
        root_path = os.getcwd()
        template = open('configs/'+tag+'_template.json','r').read()

        # Prepare config file with desired number and type of input molecules
        config_json, config_str = config_modify_func(template)
        for iii in range(20):
            print("Try %d times..."%iii)
            os.chdir(root_path)
            try:
                # Execute
                main(config_json)
                now = datetime.now()
                date_time = now.strftime("%Y-%m-%d, %H:%M:%S")
                with open(os.path.join(root_path,'3TVASP_running.log'), 'a+') as f:
                    f.write("==========================================\n")
                    f.write(config_json+" NO.%d \n"%iii)
                    f.write(date_time + " Done!\n")
                break
            except Exception as error:
                now = datetime.now()
                date_time = now.strftime("%Y-%m-%d, %H:%M:%S")
                with open(os.path.join(root_path,'3TVASP_running.log'), 'a+') as f:
                    f.write("==========================================\n")
                    f.write(config_json+" NO.%d \n"%iii)
                    f.write(traceback.format_exc())
                    f.write("\n")
                    f.write(date_time + " An exception occurred: " + repr(error) + "\n")
                os.system('rm -rf FF_step* VASP_step* VASP_files/ default.log workspace/')
                continue

        # Store results
        result_dir = 'results/' + tag + '/'
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        result_dir = result_dir + config_json[:-5].split('_')[-1] + '/'
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        os.system('mv FF_step* ' + result_dir)
        os.system('mv VASP_step* ' + result_dir)
        os.system('mv default.log ' + result_dir)
        os.system('mv VASP_files ' + result_dir)
        os.system('cp ' + config_json + ' ' + result_dir)
        json_str = json.loads(config_str)
        poscar_file = json_str[0]['lattice_poscar']['file']
        os.system('cp ' + poscar_file + ' ' + result_dir)
    return
