import os
import pandas as pd
import glob

# 📂 Define the directory where your H5 files are stored
input_folder = "/work/messier_lab/fuzzy/train/"  # Change this to your folder path
# input_folder = "results_knn/"
# 🔍 Find all HDF5 files matching the pattern `obs_cluster_*.h5`
h5_files = sorted(glob.glob(os.path.join(input_folder, "obs_cluster_*.h5")))

# 🚀 Initialize an empty list to store DataFrames
dataframes = []

# 📌 Read and concatenate all HDF5 files
for file in h5_files:
    print(f"Processing: {file}")
    
    # Read the HDF5 file into a DataFrame
    try:
        with pd.re(file, "r") as store:
            key = store.keys()[0]  # Get the key of the stored dataframe
            df = store[key]  # Load the dataframe

            # 🔹 Add a new column to track the original file index
            df["source_file"] = os.path.basename(file)
            dataframes.append(df)
    except:
        print(f"❌ Error reading file: {file}")
        continue
    

# 📌 Concatenate all DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)
output_folder = "train_val_test_data/kmeans_latend_dim_64/train/"
# 🛠 Save the merged DataFrame to a new HDF5 file
output_file = os.path.join(output_folder, "merged_obs_cluster.h5")

df.to_hdf(output_file, key="merged_data", mode="w")
# with pd.HDFStore(output_file, "w") as store:
#     store.put("merged_data", merged_df)

print(f"✅ Merging complete! Saved as: {output_file}")
