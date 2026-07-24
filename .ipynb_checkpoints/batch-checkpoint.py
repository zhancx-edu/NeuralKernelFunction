import sys
import numpy as np
import subprocess
import os




########################## run NKF model ##########################

# for seed in ["1", "2", "3"]:

#     for datasets_id in ["SCEDC_20", "SCEDC_25", "SCEDC_30", "ComCat", "SaltonSea", "SanJac", "WHITE"]:
        
#         subprocess.run(["python", "run_models.py", seed, "1", datasets_id, "neural", "neural", "neural"])

########################## run NKF model ##########################


########################## run ETAS model ##########################

for seed in ["1", "2", "3"]:

    for datasets_id in ["SCEDC_20", "SCEDC_25", "SCEDC_30", "ComCat", "SaltonSea", "SanJac", "WHITE"]:
        
        subprocess.run(["python", "run_models.py", seed, "1", datasets_id, "empirical", "empirical", "empirical"])

########################## run ETAS model ##########################