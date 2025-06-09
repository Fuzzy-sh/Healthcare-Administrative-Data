import os
import pandas as pd
import numpy as np
import tqdm
from tqdm import tqdm
from itertools import product

# class config:

#     def __init__(self, path_to_read , outcome):
    #     self.data_path = path_to_read
    #     self.outcome = outcome
    #     print(f"Reading data from {self.data_path} for {self.outcome}")
    #     # self.outcome = 'mortality'
    #     # self.outcome_path = os.path.join(self.data_path, self.outcome)
    #     # self.model_path = 'models'
    #     # self.model_name = 'model.pkl'
    #     # self.model_path = os.path.join(self.model_path, self.model_name)
    #     # self.train_data_path = os.path.join(self.data_path, 'train')
    #     # self.test_data_path = os.path.join(self.data_path, 'test')
    #     # self.val_data_path = os.path.join(self.data_path, 'val')
    #     # self.train_data = read_data(self.train_data_path, self.outcome)
    #     # self.val_data = read_data(self.val_data_path, self.outcome)
    #     # self.test_data = read_data(self.test_data_path, self.outcome)
    #     # self.features = ['age

    # def __str__(self):
    #     return f"Reading data from {self.data_path} for {self.outcome}"
    # def __repr__(self):

    # def test(self):
    #     print("Hello")
    #     return "Hello"

    # def read_data(self):
    #     # only the files with .h5 extension are read and has the outcome in the file name
    #     file_list = [file for file in os.listdir(self.data_path ) if file.endswith('.h5') and self.outcome in file]
    #     print('Processing folder:', self.data_path )
    #     dfs = []

    #     for file in tqdm(file_list):
    #         file_path = os.path.join(self.data_path , file)
    #         df=pd.read_hdf(file_path)
    #         dfs.append(df)
    #         break
     

    #     combined_all_df= pd.concat(dfs).reset_index(drop=True)
    #     print(f"Number of all individuals for {self.outcome}: {combined_all_df['subject_id'].nunique()}")
    #     return combined_all_df
    


class config_param:
    def __init__(self):
        # Initialize the class with the given parameters

        # Define the parameters
        age_labels = ['18-29', '30-39', '40-49', '50-59', '60+']
        self.col_age_cat = ['subject_age_'+label for label in age_labels]
        self.col_age_cat_updated = ['o:'+col for col in self.col_age_cat]
        self.group_colunms = ['subject_id', 'subject_sex','start_date',
       'end_date', 'substance_time_diff_days', 'mood_time_diff_days',
       'anxiety_time_diff_days', 'psychotic_time_diff_days',
       'cognitive_time_diff_days', 'otherpsych_time_diff_days',
       'Ex_MS_time_diff_days', 'Ex_Dyslipid_time_diff_days']
        
        self.group_colunms_updated = ['m:'+col for col in self.group_colunms]
        
        self.dummies_columns = ['visit', 'database']
        self.dummies_columns_updated = ['o:'+col for col in self.dummies_columns]
        self.sex_col = ['subject_sex']
        self.sex_col_updated = ['m:'+col for col in self.sex_col]
        self.colid = 'subject_id'
        self.colid_updated = 'm:subject_id'
        self.end_date = ['end_date' ] 
        #################   Convert training data and compute conversion factors    ######################

        # all 47 columns of interest + additional meta columns to easier associate trajectories with other patient auxiliary info (eg. notes)
        self.colmeta = ['subject_id', 'start_date'] #, 'charttime', 'icustayid']  # Meta-data around patient visits
        self.colmeta_updated = ['m:'+col for col in self.colmeta]
        # binary features
        self.colbin = ['subject_sex_M'] #, 'mechvent', 'max_dose_vaso', 're_admission'] 
        self.colbin_updated = ['m:'+col for col in self.colbin]
        # self.group_colunms = self.colmeta + self.sex_col + self.end_date 
        # Patient features that will be z-normalize
        self.age_index = ['subject_age_index'] #'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C', 'FiO2_1', 
                # 'Potassium', 'Sodium', 'Chloride', 'Glucose', 'Magnesium', 'Calcium', 'Hb', 
                # 'WBC_count', 'Platelets_count', 'PTT', 'PT', 'Arterial_pH', 'paO2', 'paCO2',
                # 'Arterial_BE', 'HCO3', 'Arterial_lactate', 'SOFA', 'SIRS', 'Shock_Index',
                # 'PaO2_FiO2', 'cumulated_balance']

        self.colstatic = self.sex_col + self.col_age_cat
        self.colstatic_updated = ['m:'+col for col in self.colstatic]

        self.age_cat = 'subject_age'
        self.colamhdiagn = ['substance_claim','mood_claim', 'anxiety_claim', 'psychotic_claim', 'cognitive_claim', 'otherpsych_claim', 
            'substance_hos', 'mood_hos', 'anxiety_hos', 'cognitive_medication', 
            'psychotic_hos', 'cognitive_hos', 'otherpsych_hos', 'selfharm_hos']
        self.colamhdiagn_updated = ['o:'+col for col in self.colamhdiagn]

        self.colelxdiag = ['EX_CHF', 'EX_Arrhy',
            'EX_VD', 'EX_PCD', 'EX_PVD', 'EX_HPTN_UC', 'EX_HPTN_C', 'EX_Para',
            'Ex_OthND', 'Ex_COPD', 'Ex_Diab_UC', 'Ex_Diab_C', 'Ex_Hptothy', 'Ex_RF',
            'Ex_LD', 'Ex_PUD_NB', 'Ex_HIV', 'Ex_Lymp', 'Ex_METS', 'Ex_Tumor',
            'Ex_Rheum_A', 'Ex_Coag', 'Ex_Obesity', 'Ex_WL', 'Ex_Fluid', 'Ex_BLA',
            'Ex_DA', 'Ex_Alcohol', 'Ex_Drug', 'Ex_Psycho', 'Ex_Dep', 'Ex_Stroke',
            'Ex_Dyslipid', 'Ex_Sleep', 'Ex_IHD', 'EX_Fall', 'EX_Urinary',
            'EX_Visual', 'EX_Hearing', 'EX_Tobacco', 'EX_Delirium', 'Ex_MS',
            'EX_parkinsons']
        self.colelxdiag_updated = ['o:'+col for col in self.colelxdiag]

        self.colelxdiag_physical = ['EX_CHF', 'EX_Arrhy',
            'EX_VD', 'EX_PCD', 'EX_PVD', 'EX_HPTN_UC', 'EX_HPTN_C', 'EX_Para',
            'Ex_OthND', 'Ex_COPD', 'Ex_Diab_UC', 'Ex_Diab_C', 'Ex_Hptothy', 'Ex_RF',
            'Ex_LD', 'Ex_PUD_NB', 'Ex_HIV', 'Ex_Lymp', 'Ex_METS', 'Ex_Tumor',
            'Ex_Rheum_A', 'Ex_Coag', 'Ex_Obesity', 'Ex_WL', 'Ex_Fluid', 'Ex_BLA',
            'Ex_DA',  'Ex_Stroke', 'Ex_Dyslipid', 'Ex_Sleep', 'Ex_IHD', 'EX_Fall', 'EX_Urinary',
            'EX_Visual', 'EX_Hearing', 'EX_Delirium', 'Ex_MS']
        self.colelxdiag_physical_updated = ['o:'+col for col in self.colelxdiag_physical]

        self.colelxdiag_mental = ['Ex_Alcohol', 'Ex_Drug', 'Ex_Psycho', 'Ex_Dep', 'EX_Tobacco', 'EX_parkinsons']
        self.colelxdiag_mental_updated = ['o:'+col for col in self.colelxdiag_mental]



        self.coltreatment = [ 'no_treatment','antidepressant_medication', 'antipsychotic_medication', 'benzo_medication', 'mood_medication', 'rehabilitation', 'addictions_counselling', 'unspecified_counselling', 'mental_health_therapy', 'occupational_therapy']
        self.coltretment_medication = ['antidepressant_medication', 'antipsychotic_medication', 'benzo_medication', 'mood_medication']
        self.coltreatment_counseling = ['addictions_counselling', 'unspecified_counselling', 'rehabilitation']
        self.coltreatment_therapy  = ['mental_health_therapy', 'occupational_therapy']
        self.coltreatment_no_treatment = ['no_treatment']
        self.colgroup_treatment = ['no_treatment','treatment_medication', 'treatment_therapy']
        self.colgroup_treatment_updated = ['a:'+col for col in self.colgroup_treatment]
        self.coltreatment_mdications_updated = ['a:'+col for col in self.coltretment_medication]
        self.coltreatment_counseling_updated = ['a:'+col for col in self.coltreatment_counseling]
        self.coltreatment_therapy_updated = ['a:'+col for col in self.coltreatment_therapy]
        self.coltreatment_no_treatment_updated = ['a:'+col for col in self.coltreatment_no_treatment]
        self.coltreatment_updated = ['a:'+col for col in self.coltreatment]

        self.time_buffer = pd.Timedelta(days=45)

        # self.action_columns = self.coltreatment_no_treatment_updated+ self.coltreatment_mdications_updated
        self.action_columns = ['treatment_medication',  'treatment_therapy', 'treatment_counseling']
        self.action_columns = ['a:'+col for col in self.action_columns]
        one_hot_combinations = list(product([0, 1], repeat=len(self.action_columns)))
        self.actions_dict= dict([(key,list(value)) for key, value in zip(range(0,len(one_hot_combinations)), one_hot_combinations)])
        self.group = True


        self.action_levels = 3
        self.action_columns_level_1 = ['a1:treatment']
        one_hot_combinations_level_1 = list(product([0, 1], repeat=len(self.action_columns_level_1)))
        self.train_actions_level_1_dict= dict([(key,list(value)) for key, value in zip(range(0,len(one_hot_combinations_level_1)), one_hot_combinations_level_1)])
        self.action_columns_level_2 = ['a2:medication', 'a2:therapy', 'a2:counseling']
        one_hot_combinations_level_2 = list(product([0, 1], repeat=len(self.action_columns_level_2)))
        # Remove entries where all values are 0
        # filtered_combinations = [combo for combo in one_hot_combinations_level_2 if sum(combo) > 0]

        # Create the dictionary without `000`
        # self.train_actions_level_2_dict = {key: list(value) for key, value in enumerate(filtered_combinations)}

        self.train_actions_level_2_dict= dict([(key,list(value)) for key, value in zip(range(0,len(one_hot_combinations_level_2)), one_hot_combinations_level_2)])

        self.action_columns_level_3 = ['a3:antidepressant_medication', 'a3:antipsychotic_medication', 'a3:benzo_medication', 'a3:mood_medication']
        one_hot_combinations_level_3 = list(product([0, 1], repeat=len(self.action_columns_level_3)))
        
        # filtered_combinations = [combo for combo in one_hot_combinations_level_3 if sum(combo) > 0]
        # self.train_actions_level_3_dict = {key: list(value) for key, value in enumerate(filtered_combinations)}

        self.train_actions_level_3_dict= dict([(key,list(value)) for key, value in zip(range(0,len(one_hot_combinations_level_3)), one_hot_combinations_level_3)])







        self.outcome = 'homeless'
        self.outcome_updated = 'r:'+self.outcome
        self.outcomes = ['police_interaction', 'homeless']
        self.outcomes_updated = ['r:'+col for col in self.outcomes]

        self.colvisits = ['visit_emr_MH_elect', 'visit_emr_MH_non_elect',
            'visit_emr_NonMH', 'visit_emr_visit', 'visit_family_gp',
            'visit_hospitalized_MH', 'visit_hospitalized_NonMH', 'visit_im',
            'visit_neurology', 'visit_other', 
            'visit_psychiatry', 'police_interaction']
        self.colvisits_updated = ['o:'+col for col in self.colvisits]
        self.colhsu = ['visit_emr_MH_elect', 'visit_emr_MH_non_elect',
            'visit_emr_NonMH', 'visit_emr_visit', 'visit_family_gp',
            'visit_hospitalized_MH', 'visit_hospitalized_NonMH', 'visit_im',
            'visit_neurology', 'visit_other', 
            'visit_psychiatry']
        self.colhsu_updated = ['o:'+col for col in self.colhsu]
        self.colhomeless = 'homeless'
        self.colpolice = 'police_interaction'
        self.colhomeless_updated = ['homeless']
        self.colpolice_updated = ['police_interaction']
        self.coldbs = ['database_claim', 'database_dad', 'database_form10', 'database_nacrs', 'database_pin']


        self.collog= self.colamhdiagn + self.colelxdiag + self.colvisits
        self.collog_updated = self.colamhdiagn_updated + self.colelxdiag_updated + self.colvisits_updated

        # Define treatment groups
        self.medications = [
          'antidepressant_medication', 'antipsychotic_medication', 'benzo_medication', 'mood_medication'
        ]
        self.medications_updated = ['a:'+col for col in self.medications]

        self.therapies_counseling = [
            'mental_health_therapy', 'occupational_therapy',
            'addictions_counselling', 'unspecified_counselling'
        ]
        self.rehabilitation = ['rehabilitation']
        
        self.data_path = './data/AHS_data/'
        self.data_path_out = './data/preprocessed_data/'
        self.cluster_path = 'cluster_restuls/'
        self.cluster_encoder_path = 'cluster_restuls_encoder/'
        self.split_data_path = './data/train_val_test_data/'
        self.models_path = 'models/'
        self.action_dic_filename = 'actions_dict.pkl'
        self.action_dic_cluster_filename = 'actions_dict_cluster.pkl'
        # self.data_path_preprocessed_in = 'preprocessed_data/'
        # self.data_path_preprocessed_out = 'aggregated_data/'
        self.file_name_train = 'train.h5'
        self.file_name_val = 'val.h5'
        self.file_name_test = 'test.h5'
        self.merge_file_name = 'merged_data.h5'
        self.time_interval = 365 * 2
        self.time_interval_following = 365 * 2
        self.data_type ='.h5'
        self.data_name = 'homeless'
        self.dates = ['index_date', 'end_date']
          
        self.rewards = ['rewards','rewards_h', 'rewards_p', 'rewards_h_date', 'rewards_p_date']
        self.rewards_updated = ['r:rewards']
        self.train_frac = 0.8
        self.val_frac = 0.1
        self.test_frac = 0.1


            # Define the folder and file paths
        
        # os.makedirs(self.split_data_path, exist_ok=True)  # Create the folder if it doesn't exist

        self.split_files = {
            "train": os.path.join(self.data_path_out, "train.h5"),
            "test": os.path.join(self.data_path_out, "test.h5"),
            "val": os.path.join(self.data_path_out, "val.h5"),
        }
        self.split_with_cluster_files = {
            "train": os.path.join(self.split_data_path, "train.h5"),
            "test": os.path.join(self.split_data_path, "test.h5"),
            "val": os.path.join(self.split_data_path, "val.h5"),
        }
        self.cluster_files = {
            "train": os.path.join(self.split_data_path, "train_cluster.h5"),
            "test": os.path.join(self.split_data_path, "test_cluster.h5"),
            "val": os.path.join(self.split_data_path, "val_cluster.h5"),
        }
        self.episode_raw_files = {
            "train": os.path.join(self.split_data_path, "train_episodes_raw.db"),
            "test": os.path.join(self.split_data_path, "test_episodes_raw.db"),
            "val": os.path.join(self.split_data_path, "val_episodes_raw.db"),
        }
        self.episode_cluster_files = {
            "train": os.path.join(self.split_data_path, "train_episodes_cluster.db"),
            "test": os.path.join(self.split_data_path, "test_episodes_cluster.db"),
            "val": os.path.join(self.split_data_path, "val_episodes_cluster.db"),
        }
        self.best_obs_cluster_n = {
            "train" : 125,
            "test" : 125,
            "val" : 125}
        self.best_act_cluster_n = {
            "train" : 43,
            "test" : 61,
            "val" : 40}
        # with the sensitivity of 11
        self.best_obs_encoder_cluster_n = {
            "train" : 125,
            "test" : 125,
            "val" : 125}
        # with the sensitivitiy of 50
        self.best_act_encoder_cluster_n = {
            "train" : 50,
            "test" : 53,
            "val" : 54}
        self.action_dic_filename_encoder = {
            "train": os.path.join(self.split_data_path, "train_encoder_cluster.pkl"),
            "test": os.path.join(self.split_data_path, "test_encoder_cluster.pkl"),
            "val": os.path.join(self.split_data_path, "val_encoder_cluster.pkl"),
        }
        self.action_dic_filename_cluster = {
            "train": os.path.join(self.split_data_path, "train_cluster.pkl"),
            "test": os.path.join(self.split_data_path, "test_cluster.pkl"),
            "val": os.path.join(self.split_data_path, "val_cluster.pkl"),
        }
        self.action_dic_filename_treatment_group = {
            "train": os.path.join(self.split_data_path, "train_treatment_group.pkl"),
            "test": os.path.join(self.split_data_path, "test_treatment_group.pkl"),
            "val": os.path.join(self.split_data_path, "val_treatment_group.pkl"),
        }


    # Method to return parameters as a dictionary or as individual values
    def get_params(self):
        return {
            'group_colunms': self.group_colunms,
            'dummies_columns': self.dummies_columns,
            'sex_col': self.sex_col,
            'colid': self.colid,
            'colmeta': self.colmeta,
            'colbin': self.colbin,
            'age_index': self.age_index,
            'colamhdiagn': self.colamhdiagn,
            'colelxdiag': self.colelxdiag,
            'colelxdiag_mental': self.colelxdiag_mental,
            'colelxdiag_physical': self.colelxdiag_physical,
            'coltreatment': self.coltreatment,
            'outcome': self.outcome,
            'outcomes': self.outcomes,
            'colvisits': self.colvisits,
            'coldbs': self.coldbs,
            'collog': self.collog,
            'medications': self.medications,
            'therapies_counseling': self.therapies_counseling,
            'rehabilitation': self.rehabilitation,
            'data_path': self.data_path,
            'data_path_out': self.data_path_out,
            # 'data_path_preprocessed_in': self.data_path_preprocessed_in,
            # 'data_path_preprocessed_out': self.data_path_preprocessed_out,
            'data_type': self.data_type,
            'data_name': self.data_name, 
            'colhomeless': self.colhomeless,
            'colpolice': self.colpolice,
            'colhsu': self.colhsu, 
            'colstatic': self.colstatic, 
            'time_interval': self.time_interval,
            'time_interval_following': self.time_interval_following, 
            'train_frac': self.train_frac,
            'val_frac': self.val_frac,
            'test_frac': self.test_frac, 
            'file_name_train': self.file_name_train,
            'file_name_val': self.file_name_val,
            'file_name_test': self.file_name_test, 
            'age_cat': self.age_cat, 
            'col_age_cat' : self.col_age_cat, 
            'rewards_list' : self.rewards_updated,
            'dates': self.dates,
            'rewards': self.rewards, 
            'time_buffer': self.time_buffer,



        }
    def get_params_updated(self):
        return {
            # 'colbin': self.colid_updated,
            'colid': self.colid_updated,
            'colmeta': self.colmeta_updated,
            # 'colnorm': self.colnorm,
            'colamhdiagn': self.colamhdiagn_updated,
            'colelxdiag': self.colelxdiag_updated,
            'colelxdiag_mental': self.colelxdiag_mental_updated,
            'colelxdiag_physical': self.colelxdiag_physical_updated,
            'coltreatment': self.coltreatment_updated,
            'coltreatment_list':self.coltreatment,
            'outcome': self.outcome_updated,
            'outcomes': self.outcomes,
            'colvisits': self.colvisits_updated,
            # 'coldbs': self.coldbs,
            'collog': self.collog_updated,
            'medications': self.medications_updated,
            # 'therapies_counseling': self.therapies_counseling,
            # 'rehabilitation': self.rehabilitation,
            'data_path': self.data_path,
            'data_path_out': self.data_path_out,
            # 'data_path_preprocessed_in': self.data_path_preprocessed_in,
            # 'data_path_preprocessed_out': self.data_path_preprocessed_out,
            'data_type': self.data_type,
            'data_name': self.data_name,
            'colhomeless': self.colhomeless_updated,
            'colpolice': self.colpolice_updated,
            'colhsu': self.colhsu_updated,
            'colstatic': self.colstatic_updated,
            'time_interval': self.time_interval,
            'time_interval_following': self.time_interval_following,
            'train_frac': self.train_frac,
            'val_frac': self.val_frac,
            'test_frac': self.test_frac,
            'file_name_train': self.file_name_train,
            'file_name_val': self.file_name_val,
            'file_name_test': self.file_name_test,
            'rewards_list' : self.rewards_updated, 
            'col_age_cat' : self.col_age_cat, 
            'dates': self.dates,
            'rewards': self.rewards_updated,
            'group_colunms': self.group_colunms_updated,
            'dummies_columns': self.dummies_columns_updated, 
            'sex_col': self.sex_col_updated, 
            'colbin': self.colbin_updated,
            'age_index': self.age_index,
            'age_cat': self.age_cat,
            'observation_prefix': 'o:', 
            'action_prefix': 'a:',
            'reward_prefix': 'r:',
            'cluster_path': self.cluster_path, 
            'cluster_encoder_path': self.cluster_encoder_path,
            'merge_file_name': self.merge_file_name, 
            'split_data_path': self.split_data_path, 
            'file_name_train': self.file_name_train,
            'file_name_val': self.file_name_val,
            'file_name_test': self.file_name_test, 
            'models_path': self.models_path, 
            'coltreatment_medication': self.coltreatment_mdications_updated,
            'coltreatment_counseling': self.coltreatment_counseling_updated,
            'coltreatment_therapy': self.coltreatment_therapy_updated,
            'coltreatment_no_treatment': self.coltreatment_no_treatment_updated,
            'colgroup_treatment': self.colgroup_treatment_updated,
            'action_dic_filename': self.action_dic_filename,
            'episode_raw_files': self.episode_raw_files,
            'split_files': self.split_files,
            'episode_cluster_files': self.episode_cluster_files, 
            'action_dic_cluster_filename': self.action_dic_cluster_filename, 
            'best_obs_cluster_n': self.best_obs_cluster_n,
            'best_act_cluster_n': self.best_act_cluster_n,
            'best_obs_encoder_cluster_n': self.best_obs_encoder_cluster_n,
            'best_act_encoder_cluster_n': self.best_act_encoder_cluster_n, 
            'split_with_cluster_files': self.split_with_cluster_files, 
            'action_dic_filename_encoder': self.action_dic_filename_encoder,
            'action_dic_filename_cluster': self.action_dic_filename_cluster,
            'action_dic_filename_treatment_group': self.action_dic_filename_treatment_group, 
            'cluster_files': self.cluster_files,
            'action_columns': self.action_columns,
            'actions_dict': self.actions_dict,
            'group': self.group, 
            'action_levels': self.action_levels,
            'action_columns_level_1': self.action_columns_level_1,
            # 'one_hot_combinations_level_1': self.one_hot_combinations_level_1,
            'actions_level_1_dict': self.train_actions_level_1_dict,
            'action_columns_level_2': self.action_columns_level_2,
            # 'one_hot_combinations_level_2': self.one_hot_combinations_level_2,
            'actions_level_2_dict': self.train_actions_level_2_dict,
            'action_columns_level_3': self.action_columns_level_3,
            # 'one_hot_combinations_level_3': self.one_hot_combinations_level_3,
            'actions_level_3_dict': self.train_actions_level_3_dict
       




            # 'group_colunms': self.group_colunms,
            # 'dummies_columns': self.dummies_columns,
            # 'sex_col': self.sex_col,
            # 'colid': self.colid_updated,
            # 'colmeta': self.colmeta,
            
            # 'colnorm': self.colnorm,
            # 'colamhdiagn': self.colamhdiagn,
            # 'colelxdiag': self.colelxdiag,
            # 'colelxdiag_mental': self.colelxdiag_mental,
            # 'colelxdiag_physical': self.colelxdiag_physical,
            # 'coltreatment': self.coltreatment,
            # 'outcome': self.outcome,
            # 'outcomes': self.outcomes,
            # 'colvisits': self.colvisits,
            # 'coldbs': self.coldbs,
            # 'collog': self.collog,
            # 'medications': self.medications,
            # 'therapies_counseling': self.therapies_counseling,
            # 'rehabilitation': self.rehabilitation,
            # 'data_path': self.data_path,
            # 'data_path_out': self.data_path_out,
            # 'data_path_preprocessed_in': self.data_path_preprocessed_in,
            # 'data_path_preprocessed_out': self.data_path_preprocessed_out,
            # 'data_type': self.data_type,
            # 'data_name': self.data_name, 
            # 'colhomeless': self.colhomeless_updated,
            # 'colpolice': self.colpolice_updated,
            # 'colhsu': self.colhsu, 
            # 'colstatic': self.colstatic, 
            # 'time_interval': self.time_interval,
            # 'time_interval_following': self.time_interval_following,
            # 'train_frac': self.train_frac,
            # 'val_frac': self.val_frac,
            # 'test_frac': self.test_frac,
            # 'file_name_train': self.file_name_train,
            # 'file_name_val': self.file_name_val,
            # 'file_name_test': self.file_name_test

        }
    # # Alternatively, you can have individual methods to return each parameter
    # def get_param1(self):
    #     return self.param1

    # def get_param2(self):
    #     return self.param2

    # def get_param3(self):
    #     return self.param3


# # Main program
# if __name__ == "__main__":
#     # Set parameters when creating an instance of MyClass
#     obj = MyClass(param1=10, param2=20, param3=30)

#     # Access and return the parameters
#     print(obj.get_params())       # {'param1': 10, 'param2': 20, 'param3': 30}
#     print(obj.get_param1())       # 10
#     print(obj.get_param2())       # 20
#     print(obj.get_param3())       # 30
