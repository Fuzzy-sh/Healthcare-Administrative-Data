import os
import pandas as pd
from tqdm import tqdm
import os
# this class is for reading and writing the data
import config.config_param as config_param         
from config.config_param import config_param
from utils import  UpdateData, PreprocessData, SplitData


tqdm.pandas()



def merge_files(param_dic):
    """
    Merges multiple preprocessed data files into a single dataset, applies column updates, normalization, and one-hot encoding, and saves the merged data.
    Args:
        param_dic (dict): Dictionary containing parameters for preprocessing, including:
            - 'data_path_out': Output directory for preprocessed data.
            - 'data_path': Input data directory.
            - 'data_type': File extension/type to filter files (e.g., '.h5').
            - 'data_name': Substring to identify relevant files.
    Returns:
        pd.DataFrame: The merged and fully preprocessed dataset.
    Workflow:
        1. Checks if the merged data file already exists; if so, loads and returns it.
        2. Otherwise, finds all relevant files in the output directory, reads, and concatenates them.
        3. Applies column changes, normalization, and one-hot encoding using the UpdateData class.
        4. Saves the final merged dataset for future use.
    """

   # call class preprocess_data from utils


    preprocess = PreprocessData(param_dic)
    sub_homeless_file = pd.DataFrame()
    preprocessed_data_path = param_dic['data_path_out']
    data_path = param_dic['data_path']
    data_type = param_dic['data_type']
    data_name = param_dic['data_name']
    file_path_merged = os.path.join(preprocessed_data_path , 'merged_data.h5')
    if os.path.exists(file_path_merged):
        print('File exists and it is being returned to the next step')
        return preprocess.read_data(file_path_merged)
        

    # create an object from the class preprocess_SC
    file_list = [file for file in os.listdir(preprocessed_data_path) if file.endswith(data_type) and data_name in file]
    file_list.sort()
    # merge the files 
    for file in tqdm(file_list):
        print('file:', file)

        print('Processing folder:', preprocessed_data_path )
        file_path = os.path.join(preprocessed_data_path , file)

        # df is the raw data 
        df = preprocess.read_data(file_path)
        # preprocessed_df = preprocess.preprocess(df)
        sub_homeless_file = pd.concat([sub_homeless_file, df], axis=0)
        # # print (preprocessed_data)
        # save_data(preprocessed_df, file_path )
        
        

    updatedata = UpdateData(sub_homeless_file, param_dic)
    
    preprocessed_data = updatedata.column_change()
    # print(preprocessed_data.columns)
    updated_data_ = updatedata.normalize_data(preprocessed_data)
    print(updated_data_.columns)
    updated_data = updatedata.one_hot_encoded(updated_data_)

    
    preprocess.save_data(updated_data, file_path_merged )
    
    return updated_data

def main():
    """
    Main function to execute the data preprocessing pipeline.

    This function performs the following steps:
    1. Initializes configuration parameters.
    2. Reads parameters from the configuration.
    3. Merges data files based on the provided parameters.
    4. Retrieves updated parameters after merging.
    5. Splits the merged data by subject using the updated parameters.
    6. Saves the resulting split data to disk.
    """
    config_params = config_param()
    param_dic = config_params.get_params()
    merged_data = merge_files(param_dic)
    sd = SplitData(merged_data, config_params.get_params_updated())
    sd.make_split_by_subj()
    sd.save_split_data()


if __name__ == "__main__":
    main()
