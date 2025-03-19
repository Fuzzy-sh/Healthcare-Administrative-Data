<a href="https://www.linkedin.com/in/fuzzy-shahidi"><img src="https://img.shields.io/badge/Linkdin-Fuzzy%20Shahidi-blue.svg" alt="DOI"></a>


# Zero to advanced methods for the Healthcare administrative data.
We included all the codes regarding the work that has been done for the administrative data from data preparation, traditional statistical modeling, sliding window, and patient representations, to the advanced deep learning models. 
The order of the folders for this work is:

1- <a href="https://github.com/Fuzzy-sh/AHS/tree/main/Data_preparation"> Data preparation </a>

2- <a href="https://github.com/Fuzzy-sh/AHS/tree/main/BioStatisticMethods"> Bio-Statistic Methods </a>

3- <a href="https://github.com/Fuzzy-sh/AHS/tree/main/DataPreprocessing"> Data Preprocessing </a>

4- <a href="https://github.com/Fuzzy-sh/AHS/tree/main/EnsembleHybridModels"> Ensemble Hybrid Models </a>

5- <a href="https://github.com/Fuzzy-sh/AHS/tree/main/Reinforcement_learning"> Reinforcement learning Models </a>







# 3- Codes in the "DataPreprocessing" folder published in IEEE ACCESS. 
### Reference to paper: <a href="https://doi.org/10.1109/ACCESS.2024.3520425"><img src="https://img.shields.io/badge/DOI-10.36227/techrxiv.173143474.47669751/v1-lightblue.svg" alt="DOI"></a>
## Title: Exploring the Preprocessing of a Time-Series Administrative Healthcare Dataset on Deep Learning to Improve Prediction
### Abstract:

Preprocessing methods are important in enhancing prediction performance for time-series administrative data. This study underscores the importance of preprocessing methods by comparing two data representation techniques that can be used with sliding window techniques for time-series administrative healthcare data for use with gated recurrent unit (GRU) networks. The first method uses a sequence event representation and the second employs a sequence matrix representation. The evaluation was conducted through a retrospective administrative healthcare data case study to predict multiple outcomes. The target outcomes were the first instances of healthcare encounters indicating homelessness or police interaction that appeared in the healthcare data. Results reveal that the GRU combined with sequence matrix representation and sliding window outperformed the sequence of events with the sliding window by over 20% in area under the curve (AUC) and sensitivity for both outcomes. Given the data used in this analysis, sequence matrix representation was superior to sequence event representation while using GRU models. The results remained consistent across the evaluation in real-world clinical frameworks using two trigger methods for prediction time, such as the sliding window and clinical demand window structures.

### Citation:

<pre>
@ARTICLE{10807221,
  author={Shahidi, Faezehsadat and Macdonald, M. Ethan and Seitz, Dallas and Barry, Rebecca and Messier, Geoffrey},
  journal={IEEE Access}, 
  title={Exploring the Preprocessing of a Time-Series Healthcare Administrative Dataset on Deep Learning to Improve Prediction}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Medical services;Codes;Predictive models;Law enforcement;Sparse matrices;Indexes;Data models;Databases;Logic gates;Vectors;Gated recurrent unit networks;on clinical demand window structure;sliding window;sequence of event and sequence matrix representations;time-series healthcare administrative data},
  doi={10.1109/ACCESS.2024.3520425}}
  
  </pre>

  
# 4- Codes in the "EnsembleHybridModels" folder accepted in the International Conference on Bioinformatics and Biomedicine (BIBM). 
### Reference to the conference paper: <a href="https://doi-org.ezproxy.lib.ucalgary.ca/10.1109/BIBM62325.2024.10822734"><img src="https://img.shields.io/badge/DOI-10.36227/techrxiv.173143474.47669751/v1-lightblue.svg" alt="DOI"></a>
## Title: Enhancing Risk Prediction in Mental Health Using Ensemble Hybrid Models and Administrative Healthcare Data with Irregular Intervals
### Abstract:

Risk prediction estimates the probability of future adverse outcomes for high-risk individuals to enable early intervention. Among all machine learning models, deep learning models can enhance risk predictions with respect to mental health (MH), where administrative healthcare data (Admin-HD) contain complex, irregular temporal information that a single model may not fully capture. This study aims to develop an ensemble hybrid deep-learning model (EnH-DL) to improve performance and reduce errors on unseen data compared to available deep-learning models. The model was evaluated in a cohort of individuals diagnosed with addiction or mental illness (AMH) to predict the risk of outcomes, such as the first healthcare encounter indicating homelessness (FHE-H) and the first healthcare encounters involving police (FHE-P). A sliding window and matrix-based representation were used in the preprocessing phase. Gated recurrent units (GRUs) with time decay, a one-dimensional (1D) convolutional neural network (CNN), and an attention function were integrated to build the EnH-DL. The results showed an average improvement in the area under the curve (AUC) by 2.5% and sensitivity by over 1.5%. An error reduction of over 0.22 indicates improved model reliability for unseen data. Tested in a clinical demand window to simulate real-world settings, the model achieved an 83% AUC and 79% sensitivity for FHE-H on a highly imbalanced test set. In conclusion, EnH-DL outperformed existing models for multiple MH outcomes using Admin-HD.

### Citation:

<pre>
@inproceedings{shahidi2024enhancing,
  title={Enhancing Risk Prediction in Mental Health Using Ensemble Hybrid Models and Administrative Healthcare Data with Irregular Intervals},
  author={Shahidi, Faezehsadat and MacDonald, M Ethan and Seitz, Dallas and Barry, Rebecca and Messier, Geoffrey},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={3673--3678},
  year={2024},
  organization={IEEE}
}
  
  </pre>


