# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:36:10 2021

@author: Stefan Salbrechter, MSc
"""

# Edit the following Paths ---------------------------------------------------

file_path_1  = "C:/Users/Stefa/Documents/Uni/Projektassistenz/"
file_path_2  = "F:/RTRS_News_Data/"

raw_data_dir = file_path_2+"News_Data_1/CLEAN/"
w2v_data_src = file_path_2+"News_Data_1/CLEAN/train_w2v_data_1996-2018/"  # file_path_1+"FX/Data/train_w2v_data_1996-2019/" 

root_dir     = file_path_1+"Paper III GUIDED TOPIC CLUSTERING/Textual_Factors/"
docs_dir     = file_path_2+"News_Data_1/CLEAN/"


# preprocessing_data.py -------------------------------------------------------

y_start = 2020 
y_end   = 2022

# Generate w2v training files
filename          = 'news_data_xl_####'
min_sent_len      = 5
gen_unigrams      = False
gen_bigrams       = True
gen_trigrams      = False
bigram_min_count  = 300
bigram_threshold  = 1
trigram_min_count = 300
trigram_threshold = 25
custom_bigrams    = ['carbon_tax', 'climate_change', 'climate_risk', 'sustainability_goals', 'climate-related_risks']

# Train word2vec
w2v_params       = {'sg':1, 'hs':0, 'size':64, 'negative':10, 'window':18, 'min_count':150, 'alpha':0.03, 'min_alpha':0.005, 'workers':8}
w2c_model_name   = 'w2v_skip_gram_64_neg_10_window_18_60_epochs_bigrams_1996_2018'
gram_type        = 'bigrams'
n_epochs         = 60



# TextualFactors.py ----------------------------------------------------------

# General Variables
train_w2v        =  w2v_data_src+"bigrams/" 
w2v_file         = "models/w2v_cbow_64_neg_10_window_18_60_epochs_bigrams_1996_2018.word2vec"
word_freq_df     = "data/word_count_df_bigrams_1996-2017.csv"     # Enter None to create new word_count_df
pca_dim          = 63 
transf2unitvec   = True
add_polarity_dim = True  # Conatonate an additional dimension to the vectors that contains information about the word polarity


# Generate Topics
method                = 'proj-method'    # 'cos-sim-method' or 'proj-method'
generate_word_clouds  = True
word_cloud_size       = (24,12)
word_cloud_pixel      = 1400 
word_cloud_shape      = 'rectangle'

proj_method_params = {'max_dist':     2,        # Stop iteration if the angle between the closest vector to X and it's projection on X exceeds max_dist     
                      'update_freq':  4,        # Re-fit the proj_subspace after every x words that are added to the topic
                      'gravity'    :  0.2,      # A greater parameter value keeps the topic closer to the original seed phrases
                      'k-similar'  :  3000,     # Number of similar words per seed word obtained from similarity search (lower number -> faster search but topic quality may delcline)
                      'lambda'     :  0.5       # Regularization parameter, reasonable values range from 0.0 to 2.0 (0.0 -> no regularization)
                     }


# Factor Loadings (general)
start_year    = 1996 
end_year      = 2022
chunksize_1   = 25000
news_filename = 'news_data_xl_####_clean'



# Factor Loadings (Cosine Similarity Method)
from_file_path    = root_dir+'output/2022_02_04_09h_44m_12s/'    # Enter 'None' to calculate from actual run_dir or enter run_dir (root_dir+'output/2021_11_22_12h_06m_06s/')
barrier_sim       = 0.95    # Min. cosine similarity where two words are considered as similar
max_word_contr    = 4
max_topic_tokens  = 500     # Consider the first 'max_topic_tokens' of each news article
batch_size        = 1000
chunksize_2       = 50000
bigram_model_src  = root_dir+'models/bigram_phrase_model_1996-2017.pkl'


df_filter = {}
df_filter['country']      = ['US'] #['ID', 'KR', 'SG', 'CN', 'IN', 'TH', 'VN', 'HK', 'JP', 'MY', 'PK', 'TW'] #['AT', 'BE', 'FR', 'LV', 'NL', 'PT', 'EE', 'GR', 'IE', 'LI', 'FI', 'DE', 'IT', 'LT', 'ES']
df_filter['industry']     = []
df_filter['company']      = []
df_filter['topic']        = []
df_filter['exclude_subj'] = []



