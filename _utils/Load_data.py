import pandas as pd
import numpy as np
import scipy


import platform
import os

def load_data(seed,data_directory):
    
    #importing the resting state fmri data.

    if platform.system() == "Windows":
        data_file_path = os.path.join(data_directory,"fmri_rs.npy")
    elif platform.system() == "Darwin":
        data_file_path = os.path.join(data_directory,"fmri_rs.npy")
    elif platform.system() == "Linux":
        data_file_path = os.path.join(data_directory,"fmri_rs.npy")
    else:
        raise RuntimeError("Unsupported operating system")

    #loading the data file 
    with open(data_file_path, "rb") as f:
        fmri_rs = np.load(f)


    #importing the matfile with the seed data

    # Define the file path based on the operating system
    if platform.system() == "Windows":
        mat_file_path = os.path.join(data_directory, "MMP_HCP_60_splits.mat")
    elif platform.system() == "Darwin":
        mat_file_path = os.path.join(data_directory, "MMP_HCP_60_splits.mat")
    elif platform.system() == "Linux":
        mat_file_path = os.path.join(data_directory, "MMP_HCP_60_splits.mat")
    else:
        raise ValueError("Unsupported platform")

    # Load the .mat file
    mat_file = scipy.io.loadmat(mat_file_path)


    #getting the subject ids 

    # Define the file path based on the operating system
    if platform.system() == "Windows":
        file_path = os.path.join(data_directory,"MMP_HCP_753_subs.txt")
    elif platform.system() == "Darwin":
        file_path = os.path.join(data_directory,"MMP_HCP_753_subs.txt")
    elif platform.system() == "Linux":
        file_path = os.path.join(data_directory,"MMP_HCP_753_subs.txt")
    else:
        raise ValueError("Unsupported platform")


    # Load the file if the path is set
    if file_path:
        try:
            with open(file_path, 'r') as file:
                HCP_753_Subjects = [int(line.strip()) for line in file.readlines()]
        except Exception as e:
            print(f"An error occurred: {e}")
            

    # Loading the response variables
    if platform.system() == "Windows":
        csv_file_path = os.path.join(data_directory, "MMP_HCP_componentscores.csv")
    elif platform.system() == "Darwin":
        csv_file_path = os.path.join(data_directory, "MMP_HCP_componentscores.csv")
    elif platform.system() == "Linux":
        csv_file_path = os.path.join(data_directory, "MMP_HCP_componentscores.csv")
    else:
        raise ValueError("Unsupported platform")


    # Extract subject lists from the loaded file
    seed_1 = mat_file['folds'][f'seed_{seed}'][0, 0]
    subject_lists = seed_1['sub_fold'][0, 0]['subject_list']
    test_subjects = [int(item[0]) for item in subject_lists[0, 0].flatten()]


    # Transposing the data file so that each sample would be a row.
    fmri_rs = fmri_rs.T

    #extracting the response variables 

    # reading the csv    
    df = pd.read_csv(csv_file_path)
    #converts the subject column from the datasframe into numbers if there is an issue in conversion error is safely handled by coerce
    df['Subject'] = pd.to_numeric(df['Subject'], errors='coerce')
    #selecting the rows of the subject which are only in the list 
    df = df[df['Subject'].isin(HCP_753_Subjects)].reset_index(drop = True)
    #Split all our data into a Train and Test Set
    df_train, df_test = df[~df['Subject'].isin(test_subjects)], df[df['Subject'].isin(test_subjects)]
            

    return fmri_rs,df_train,df_test


def vectomat_matlab(vector, outputdim):
    '''
    This code is to regenerate the symmetric functional connectivity matrix from the given vectorized upper triangular portion.
    This code accounts for the mismatch between MATLAB and Python indexing.

    vector: ndarray - the sample vector as a column vector (:,1)
    outputdim: scalar - the dimension of the symmetric matrix
    '''

    # Checking whether the vector dimension and the desired output dimensions match
    vector_length = vector.shape[0]
    desired_length = outputdim * (outputdim - 1) / 2

    # Check if lengths match and raise an error if not
    if vector_length != desired_length:
        raise ValueError("Vector length is insufficient to construct the symmetric matrix.")
    
    # Create a symmetric matrix with zeros
    matrix = np.zeros((outputdim, outputdim))
    
    p = 0
    
    for i in range(outputdim):
        for j in range (i,outputdim):
            if i == j :
                matrix[i,j] = 0
            else:
                matrix[i,j] = vector[p]
                matrix[j,i] = matrix[i,j]
                p = p+1
                
    return matrix

def samplestomat(dataset,outputdim):
    
    from Load_data import vectomat_matlab
    
    '''
    This code is developed to convert the vectorized data matrix in to a 3D data tensor.
    
    dataset : nd:array - (samples*features)
    outputdim : scalar

    '''

    #number of samples
    n_samples = dataset.shape[0]
    #3D matrix to hold the output
    out_dataset = np.zeros((n_samples,outputdim,outputdim))

    for p in range(n_samples):
        
        sample = dataset[p]
        sample = vectomat_matlab(sample,outputdim)
        out_dataset[p] = sample

    #random_index = np.random.randint(0, n_samples)
    #random_sample = out_dataset[random_index]

    # Plot the heatmap
    #plt.figure(figsize=(10, 8))
    #sns.heatmap(random_sample, cmap='viridis', cbar=True)
    #plt.title(f'Heatmap of Random Sample {random_index}')
    #plt.show()

    return out_dataset


def normalize_by_frobenius_norm(samples):
    """
    Normalizes each sample (2D matrix) in the array by its Frobenius norm.

    Parameters:
    samples (numpy.ndarray): A 3D numpy array with dimensions [samples, rows, columns].

    Returns:
    numpy.ndarray: A 3D numpy array with each sample normalized by its Frobenius norm.
    """
    # Calculate the Frobenius norm for each sample
    frobenius_norms = np.linalg.norm(samples, axis=(1, 2))
    
    # Reshape the norms to broadcast correctly for division
    frobenius_norms = frobenius_norms[:, np.newaxis, np.newaxis]
    
    # Normalize each sample by its Frobenius norm
    normalized_samples = samples / frobenius_norms
    
    return normalized_samples


        
