# import numpy as np
# from pathlib import Path
# import os

# folders=["wine+quality"]

# for folder in folders:
#     for file in os.listdir(folder):
#         filepath=os.path.join(folder,file)
#         if(filepath.endswith(".csv")==False):
#             continue

#         f=open(filepath)
#         content=f.read()
#         print(content)
#         f.close()