import os, sys
os.chdir('/root/MaskerLi/3TVASP/External_3T/')
sys.path.append('./utils')
from utils.process_molecule import convert_molecule

file_list = ['input/oEC2_minus2.lmp']
for fn in file_list:
    convert_molecule(fn)
