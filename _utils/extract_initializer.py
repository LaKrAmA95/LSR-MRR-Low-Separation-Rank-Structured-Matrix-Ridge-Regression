import os
import pickle
import platform
import re

def extract_initializer(seed,path):
    
    #setting up the path 
    folder_path = path
 
    # Get all .pkl files in the folder
    pkl_files = [file for file in os.listdir(folder_path) if file.endswith('.pkl')]

    #seed to explore
    seed = seed

    # Find the file that matches the specified seed
    selected_file = None
    for file_name in pkl_files:
        # Use regular expression to search for Seed_<seed> in the filename
        if re.search(f"Seed_{seed}_", file_name):
            selected_file = file_name
            break  # Stop once the file is found
        
    file_path = os.path.join(path,selected_file)
    with open(file_path,'rb') as f:
        data = pickle.load(f)
        initializer = data[-1]
    
    print(f'Initialized with:{selected_file}')
    
    return initializer

        
    
