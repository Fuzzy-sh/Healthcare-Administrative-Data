import os
from tqdm import tqdm
import sys


# this class is for reading and writing the data
# import config.config_param as config_param
# from config.config_param import config_param     




from utils import PreprocessData 
from config.config_param import config_param




#preprocess_data, split_data, read_data, save_data
# from utils.utils_cltr import ConvertDatasetsToClusters
tqdm.pandas()
import argparse


def preprocess_file(file_path, file_name, output_path):
    """
    Preprocesses a specified file by reading, cleaning, and saving the processed data.
    Args:
        file_path (str): The directory path where the input file is located.
        file_name (str): The name of the file to preprocess.
        output_path (str): The directory path where the preprocessed file will be saved.
    Description:
        - Loads configuration parameters.
        - Reads the raw data file.
        - Applies preprocessing steps using the PreprocessData class.
        - Saves the preprocessed data to the specified output path.
    Prints:
        - Status messages indicating the start of preprocessing, file paths, and the number of records processed.

    """
   # call class preprocess_data form utils

    config_params = config_param()
    # read the data from config
    param_dic = config_params.get_params()

    preprocess = PreprocessData(param_dic)
   
    print(f"Starting preprocessing for: {file_name}")
    print(f"File path: {file_path}")
    # Add your preprocessing logic here
    # For example, reading the file, cleaning, etc.


    file_path = os.path.join(file_path , file_name)
    # df is the raw data 
    df = preprocess.read_data(file_path)
    
    preprocessed_df = preprocess.preprocess(df, file_name)
    # # print (preprocessed_data)
    # preprocessed_file_path = param_dic['data_path_out']

    preprocessed_file_path = os.path.join(output_path , file_name)
    print(f"Number of records for aggregated {preprocessed_file_path} : {preprocessed_df.shape[0]}")
    preprocess.save_data(preprocessed_df, preprocessed_file_path)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess a file.")
    parser.add_argument("--file_path", required=True, help="Full path to the file.")
    parser.add_argument("--file_name", required=True, help="Name of the file.")
    parser.add_argument("--output_path", required=True, help="Output path for the preprocessed file.", default='preprocessed_data')
    args = parser.parse_args()

    # Call the preprocessing function
    preprocess_file(args.file_path, args.file_name, args.output_path)