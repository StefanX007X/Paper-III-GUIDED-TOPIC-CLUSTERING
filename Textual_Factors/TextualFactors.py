# -*- coding: utf-8 -*-
"""
Created on Fri Feb 04 2022
@author: Stefan Salbrechter, MSc
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
import json
import argparse
import torch
import threading
import nltk
import time
import faiss
from math import log10, floor
from datetime import datetime
from threading import Thread
from queue import Queue, Empty, Full
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from torch.nn import functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
from sklearn.decomposition import PCA
from scipy import optimize
from pathlib import Path
from gensim.models import KeyedVectors
from tqdm.notebook import tqdm
from pandas.tseries.holiday import USFederalHolidayCalendar as usholidaycalendar
from wordcloud import WordCloud
from numpy.linalg import norm
from pandas.tseries.holiday import USFederalHolidayCalendar as usholidaycalendar
from utilities import Dataloader, MyIterableDataset, SimilarityMeasure, Get_PCA_Embds, Unitvec, UnitColumns, FilterNews, ThreadKiller, FreeGPU_Memory
from config import *

tqdm().pandas()

 

class TextualFactors_v2:
    """
    Generate a series of textual factors based on the file 'seed_words.xlsx' 
    that is located in the root directory. 
    Output the time series of each textual factor as well as the word support 
    for each topic, including distances to the topic center vector.
    Also generate word clouds for each topic.
    
    # Faiss (Efficient similarity search)
    nlist     ... number of cells for similarity search (https://www.pinecone.io/learn/faiss-tutorial/)
    nprobe    ... number of nearby cells to search too
    """
    def __init__(self, root_dir, w2v_file, docs_dir, pca_dim, train_w2v=None, 
                 transf2unitvec=True, add_polarity_dim=True, nlist=50, nprobe=8, method=None):
        
        self.root_dir   = root_dir           # root directory
        self.w2v_file   = w2v_file           # word2vec word vectors
        self.train_w2v  = train_w2v          # directory containing the files to train word2vec
        self.docs_dir   = docs_dir           # directory that contains the cleaned news articles
        self.pca_dim    = pca_dim            # reduce the dimensionality of the word vecotrs to pca_dim   
        self.cos_angle      = SimilarityMeasure(sim_measure='cos_angle').calc_similarity
        self.cos_similarity = SimilarityMeasure(sim_measure='cos_similarity').calc_similarity
        
        # Create output directory if it doesn't exist
        Path(self.root_dir+'output/').mkdir(parents=True, exist_ok=True) # directory to write files
        
        self.embeddings = KeyedVectors.load(self.w2v_file)
        self.vocab      = set(self.embeddings.wv.key_to_index.keys())
        self.vocab_list = list(self.vocab)  
            
        self.dim_reduction_pca(transf2unitvec, add_polarity_dim) # Run PCA
        self.get_embeddings = Get_PCA_Embds(self.embds_dict)
                           
        if method == 'proj-method':
            # Efficient similarity search
            self.xb = self.get_embeddings[self.vocab_list].astype(np.float32)
            quantizer  = faiss.IndexFlatL2(self.pca_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.pca_dim, nlist)
            self.index.train(self.xb)
            self.index.add(self.xb)  
            self.index.nprobe = nprobe  
        
        print(f'vocabulary size: {len(self.vocab)}')

        
 
        
    def dim_reduction_pca(self, transf2unitvec, add_polarity_dim):
        """
        Load word embeddings and run a principal component analysis
        """
        print(f"run PCA ... {datetime.now().strftime('%H:%M:%S')}")

        vocab_list      = list(self.vocab)
        self.emb_size   = self.embeddings.wv[vocab_list[0]].shape[0]
        self.embds_dict = {}

        # Run PCA
        pca = PCA(n_components=self.pca_dim)
        principalComponents = pca.fit_transform(self.embeddings.wv[vocab_list])

        if add_polarity_dim:
            word_polarities = pd.read_csv(root_dir+'data/word_polarities.csv', index_col=0)
            word_polarities.columns = ['polarity']    
            polarity_words  = set(word_polarities.index)
            self.pca_dim += 1
            
        for i, w in enumerate(vocab_list):
            if transf2unitvec:
                if add_polarity_dim:
                    if w in polarity_words:
                        self.embds_dict[w] = Unitvec(np.append(principalComponents[i], 5*max(abs(principalComponents[i]))*word_polarities.loc[w]))
                    else:
                        self.embds_dict[w] = Unitvec(np.append(principalComponents[i], 0))
                else:
                    self.embds_dict[w] = Unitvec(principalComponents[i])
            else:
                self.embds_dict[w] = principalComponents[i]
                
     
                  

    def proj_to_subspace(self, w, X):
        b = np.linalg.inv(X.T@X) @ (X.T @ w)
        w_dach = X @ b
        return w_dach
    
    
    
    def generate_word_clouds(self, data, topic='Topic_i', nwords=100, method='cs_method', X=None):      
        if self.wc_shape == 'circle':
            x, y = np.ogrid[:self.wc_pixel, :self.wc_pixel]
            mask = (x - self.wc_pixel/2) ** 2 + (y - self.wc_pixel/2) ** 2 > (self.wc_pixel/2) ** 2
            mask = 255 * mask.astype(int)
            wordcloud = WordCloud(width=self.wc_pixel, height=self.wc_pixel, max_words=nwords, relative_scaling=1, normalize_plurals=False, background_color="white", mask=mask)            
        else:          
            wordcloud = WordCloud(width=self.wc_pixel, height=int(self.wc_pixel*9/16), max_words=nwords, relative_scaling=1, normalize_plurals=False, background_color="white")
       

        if method == 'cs_method':
            wordcloud = wordcloud.generate_from_frequencies(data)
             
            fig = plt.figure(figsize=(self.wc_size[0], self.wc_size[1]))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(self.run_dir+'Word Clouds/'+topic+'.png', dpi=150, facecolor='w', edgecolor='w', orientation='portrait')
            plt.close(fig)
 

        elif method == 'proj_method':
            topics_dict = {}
            for i, w in enumerate(self.topic):
                v = self.get_embeddings[w]
                v_dach  = self.proj_to_subspace(v, X)    
                sim_val = self.cos_angle(v_dach, v)
                topics_dict[w] = 1/(1+sim_val)
                
            wordcloud = wordcloud.generate_from_frequencies(topics_dict)
             
            fig = plt.figure(figsize=(self.wc_size[0], self.wc_size[1]))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(self.run_dir+'Word Clouds/'+topic+'.png', dpi=150, facecolor='w', edgecolor='w', orientation='portrait')
            plt.close(fig)           
                 
    

 
    
    # Topic generation with Word2Vec Embeddings (Cosine Similarity Method) ---------------------------------------------
     
    def weighted_seed_words(self, x):
        words   = x['seed_words'].split(',')
        weights = x['weight'].split(',')
        return [tuple([words[i].strip(), weights[i]]) for i in range(len(words))]
    

    def most_similar(self, seed_words, topn):
        """
        Find the most similar words for a list of seed words with cosine similarity.
        If the number of seed_words is > 1, then the weighted mean of those seed vectors is calculated and normalized.
        """
        V = Get_PCA_Embds(self.embds_dict)[list(self.vocab)]

        mean_v = []
        for seed_w in seed_words:
            weight = float(seed_w[1])
            mean_v.append(weight*Get_PCA_Embds(self.embds_dict)[seed_w[0]])   
        mean_v = Unitvec(np.array(mean_v).mean(axis=0))

        most_similar_df = pd.DataFrame(index=list(self.vocab), columns=['cos_sim'])
        most_similar_df['cos_sim'] = V @ mean_v
        most_similar_df = most_similar_df.sort_values(by='cos_sim', ascending=False).iloc[:topn]

        most_similar_dict = {}
        for w in most_similar_df.index:
            most_similar_dict[w] = most_similar_df.loc[w, 'cos_sim']

        topic_vecs = Get_PCA_Embds(self.embds_dict)[list(most_similar_df.index)]

        return most_similar_dict, most_similar_df, topic_vecs
    
       
            
    def run_generate_topics(self, generate_word_clouds=False, word_cloud_size=(24,12), word_cloud_pixel=1400, word_cloud_shape='rectangle'):
        """
        
        """
        self.wc_size  = word_cloud_size
        self.wc_shape = word_cloud_shape
        self.wc_pixel = word_cloud_pixel
        
        # Create output directory and log files
        self.run_dir = self.root_dir+'output/'+datetime.now().strftime('%Y_%m_%d_%Hh_%Mm_%Ss')+'/'
        Path(self.run_dir).mkdir(parents=True, exist_ok=True) 
        
        seed_data = pd.read_excel(self.root_dir+'seed_words.xlsx', sheet_name='weighted_seed_words')   # Load seed words from file
        seed_data.seed_words = seed_data.apply(self.weighted_seed_words, axis=1)
        seed_data = seed_data.drop(columns=['weight'])
        seed_data = seed_data.set_index('topic_idx')

        self.topic_vectors = {}
        self.topic_df = pd.DataFrame(index   = np.arange(0, seed_data.topic_size.max()), 
                                     columns = pd.MultiIndex.from_product([list(seed_data.index), ["words", "cos_sim"]]))

        print(f"Generate Topics ... {datetime.now().strftime('%Hh:%Mm:%Ss')}")
        for topic in seed_data.index:
            most_similar_dict, most_similar_df, topic_vecs = self.most_similar(seed_data.loc[topic, 'seed_words'], seed_data.loc[topic, 'topic_size'])
            self.topic_df.loc[:, (topic, 'words')]   = most_similar_df.index.values
            self.topic_df.loc[:, (topic, 'cos_sim')] = most_similar_df.cos_sim.values
            self.topic_vectors[topic] = topic_vecs.tolist()
    
            # Generate Word Clouds    
            if generate_word_clouds:
                Path(self.run_dir+'Word Clouds/').mkdir(parents=True, exist_ok=True) 
                self.generate_word_clouds(most_similar_dict, topic=topic, nwords=seed_data.loc[topic, 'topic_size'])
                
                
        # Save files             
        self.topic_df.to_csv(self.run_dir+'topics.csv', encoding='utf-8', index=False)

        with open(f'{self.run_dir}topic_vectors.json', 'w') as f:
            json.dump(self.topic_vectors, f)
            



    # Topic generation with Word2Vec Embeddings (Projection Method) ---------------------------------------------
    
    def func(self, a, W_orth, I, X, C, weights, params):
        self.X_new = X + W_orth @ np.diag(a)
        H_A   = self.X_new @ np.linalg.inv(self.X_new.T @ self.X_new + np.identity(X.shape[1]) * params['lambda'] ) @ self.X_new.T
        RSS   = np.sum((((I-H_A) @ C) @ np.diag(weights))**2)
        return RSS   
    
    
    
    def run_gen_topics_proj_method(self, 
                                   params,
                                   word_cloud_size       = (24,12), 
                                   word_cloud_pixel      = 1400, 
                                   word_cloud_max_words  = 800, 
                                   generate_word_clouds  = True, 
                                   word_cloud_shape      = 'box'):
        
        """
        Generate topics based on combinations of multiple seed words (2 or more). The embeddings of the seed words form
        a subspace (hyperplane) in the higher dimensional vector space. 
               
        Then, all word embeddings of the words contained in the seed bucket are projected onto the 
        hyperplane and the angle between the original vectors and their projection vectors is calculated. 
        
        In each iteration the closest vector to the subspace is selected and added to the topic. Then, the hyperplane
        is fitted via (penalized) least squares regression through all topic vectors. 
        
        This procedure repeats until the number of words in the topic >= topic_size or if the angle between the topic hyperplane 
        and the closest word embedding which is not part of the topic is > max_dist.
                
        All embedding vectors have to be unit vectors -> transf2unitvec = True
        """       
        print('Generate topics ...')
        self.wc_size  = word_cloud_size
        self.wc_shape = word_cloud_shape
        self.wc_pixel = word_cloud_pixel        
        self.max_words = word_cloud_max_words
        
        vocab_series       = pd.Series(self.vocab_list)
        subspace_vectors   = {}
        topic_vectors      = {}
        
        # Load seed words from file
        seed_data = pd.read_excel(root_dir+'seed_words.xlsx', sheet_name='seed_words_proj_method_v2')   # Load seed words from file
        seed_data = seed_data.set_index('topic_idx')
    
        # Create output directory and log files
        self.run_dir = self.root_dir+'output/'+datetime.now().strftime('%Y_%m_%d_%Hh_%Mm_%Ss')+'/'
        Path(self.run_dir).mkdir(parents=True, exist_ok=True) 
                        
        # Create log file        
        with open(self.run_dir+'log.txt', 'w') as f:
            f.write(f"Topic generation with projection method\
                    \nWord2Vec embeddings size:  {self.emb_size}\
                    \nPCA embeddings size:       {self.pca_dim}\
                    \nGravity:                   {params['gravity']}\
                    \nk-similar:                 {params['k-similar']}\
                    \nMax. distance:             {params['max_dist']}\
                    \nlambda:                    {params['lambda']}\
                    \n\nSeed words:\n"
            )        
            for topic_idx in seed_data.index:
                f.write(f"{topic_idx}, Seed words: {str(seed_data.loc[topic_idx,'pos_seed_words']): <5} Neg. seed words: {str(seed_data.loc[topic_idx,'neg_seed_words'])}\n")     
        
                
        for num, topic_idx in enumerate(seed_data.index):           
            temp_log_file = f"\n{topic_idx}:\n"
            print(topic_idx)
            
            run, j   = True, 0                
            pos_seed = list(eval(seed_data.loc[topic_idx, 'pos_seed_words']))
            neg_seed = list(eval(seed_data.loc[topic_idx, 'neg_seed_words']))
            
            if len(neg_seed) > 0:
                if type(neg_seed[0]) != tuple:
                    neg_seed = [tuple(neg_seed)]
                else:
                    neg_seed = list(neg_seed)               
                        
            if type(pos_seed[0]) != tuple:
                pos_seed = [tuple(pos_seed)]
            else:
                pos_seed = list(pos_seed)

            
            proj_subspace  = [pos_seed[i][0] for i,_ in enumerate(pos_seed)]
            neg_seed_words = [neg_seed[i][0] for i,_ in enumerate(neg_seed)]
            pos_weights    = np.array([pos_seed[i][1] for i,_ in enumerate(pos_seed)])        
            neg_weights    = np.array([neg_seed[i][1] for i,_ in enumerate(neg_seed)])               
                        
            self.topic = [pos_seed[i][0] for i,_ in enumerate(pos_seed) if pos_seed[i][1] > 0]
             
            # Similarity Search
            xq = self.get_embeddings[proj_subspace+neg_seed_words].astype(np.float32)   # query vectors
            _, sim_idx = self.index.search(xq, params['k-similar'])         
            bucket_idx = np.unique(sim_idx.flatten())   
           
            V_buckets  = pd.DataFrame(index = vocab_series[bucket_idx], 
                                      data  = {'vector': list(self.xb[bucket_idx,:])})            
            self.V_bucket = V_buckets
            
            
            for i, w in enumerate(proj_subspace):           
                a = V_buckets.loc[w, 'vector'].reshape(-1,1)
                A = a if i == 0 else np.hstack((A, a))
                V_buckets = V_buckets.drop([w])
            
            if len(neg_weights) >= 1:
                for i, w in enumerate(neg_seed_words):
                    b = V_buckets.loc[w, 'vector'].reshape(-1,1)
                    N = b if i == 0 else np.hstack((N, b))
                    V_buckets = V_buckets.drop([w])
         
                # Adjust A by the negative seed words
                if N.ndim == 1:
                    A = UnitColumns(A @ np.diag(pos_weights) + N * neg_weights)
                else:      
                    A = UnitColumns(A @ np.diag(pos_weights) + N @ np.diag(neg_weights) @ np.ones(shape=(len(neg_weights), len(proj_subspace)))*(1/len(neg_weights)) )
             
                
            V = np.vstack(V_buckets.vector).T    
            X, C = A.copy(), A.copy()
            C_orth, var = np.array([]), np.array([])
            I = np.identity(X.shape[0]) 
            weights = pos_weights   
            gravity = params['gravity']
            
            while run == True:
                j += 1            
                B     = np.linalg.inv(X.T@X) @ X.T @ V
                B_adj = np.diag(pos_weights) @ B                               # Scale Projection Coefficients with weights 
                sel_coeff = B_adj.sum(axis=0) > 0.5*max(pos_weights)
                V_proj_adj= X @ B_adj[:,sel_coeff] 
                V_orth    = V[:,sel_coeff] - (X @ B[:,sel_coeff])              # Calculate the non-adjusted orthogonal vectors 
                
                norm_proj = norm(V_proj_adj, axis=0)  
                norm_orth = norm(V_orth, axis=0) 
                alpha     = np.arctan(norm_orth/norm_proj)   # Angle between the word vectors and their projection onto the (hyper) plane
                min_idx   = alpha.argmin()
                true_idx  = np.where(sel_coeff==True)[0] 
                idx       = true_idx[min_idx]                # Index of the smalles value of alpha       
                alpha_min = np.min(alpha)                    # Smallest alpha
                new_word  = V_buckets.index[idx]             # New word that is added to the topic
                w = V[:, idx]                                # Vector of the new word           
                C = np.vstack([C.T, w]).T                    # Append new word(s) to the proj_subspace X and fit a new (hyper) plane through all points
                self.topic.append(new_word)                  # Append new word to topic   
                
                if ((j-1) % params['update_freq']) == 0:
                    w_orth = V_orth[:, min_idx]
                else:
                    w_orth = Unitvec(w_orth + V_orth[:, min_idx])
                    
                weights = weights * (1+gravity)              # Increase weights of existing topic words
                weights = np.append(weights, 1)              # Add the weight of the newly added word
                
                if (j % params['update_freq']) == 0:                              
                    W_orth = np.array([w_orth.T]*X.shape[1]).T           
                    result = optimize.minimize(self.func, [0]*X.shape[1], method="CG", args=(W_orth, I, X, C, weights, params))   # Optimize
                    X      = UnitColumns(self.X_new)                                                                 # Update X
                
                V_buckets = V_buckets.drop([new_word])                       
                V = np.vstack(V_buckets.vector).T    
                
                gravity = max(0, gravity - params['gravity']/seed_data.loc[topic_idx, 'topic_size'])                                           # Decay gravity to 0 
                            
                if j == 1:
                    C_orth = V_orth[:, min_idx]   
                    var = np.array(np.nan)
                else:
                    C_orth = np.vstack([C_orth.T, V_orth[:, min_idx]]).T
                    var = np.append(var, np.var(C_orth, axis=1).mean())
                      

                temp_log_file += f"{new_word: <30} word #{j:<3}; angle: {alpha_min:.3f};  Similarity col1: {self.cos_similarity(X[:,0], A[:,0]):.6f}, col2: {self.cos_similarity(X[:,1], A[:,1]):.6f}\n"
                       
                if ((alpha_min > params['max_dist']) & (j >= 10)) or (C.shape[1] >= seed_data.loc[topic_idx, 'topic_size']) or (len(true_idx) == 1):
                    run = False
                    
   
            with open(self.run_dir+'log.txt', 'a') as f:   
                f.write(temp_log_file)
                
            if num == 0:
                self.topic_df = pd.DataFrame(data={topic_idx: self.topic})
            else:
                self.topic_df = pd.concat([self.topic_df, pd.DataFrame(data={topic_idx: self.topic})], axis=1)
            
            subspace_vectors[topic_idx] = X.tolist()
            topic_vectors[topic_idx]    = C.tolist()
            
            # Generate word cloud
            if generate_word_clouds:
                Path(self.run_dir+'Word Clouds/').mkdir(parents=True, exist_ok=True) 
                self.generate_word_clouds(data=None, X=X, topic=topic_idx, method="proj_method", nwords=seed_data.loc[topic_idx, 'topic_size'])
                     
        self.topic_df.to_csv(self.run_dir+'topics.csv', encoding='utf-8', index=False)
        
        with open(f'{self.run_dir}subspace_vectors.json', 'w') as f:
            json.dump(subspace_vectors, f)

        with open(f'{self.run_dir}topic_vectors.json', 'w') as f:
            json.dump(topic_vectors, f)
    
    




    
    # Factor Loadings Cosine Similarity Method -------------------------------------------------------------------

    def threaded_batches_feeder(self, tokill, batches_queue, iterable_dataset):
        """
        Threaded worker for pre-processing input data.
        tokill is a thread_killer object that indicates whether a thread should be terminated
        """
        while tokill() == False:
            for batch, (timestamp_batch, id_batch, vector_batch) in enumerate(iterable_dataset):
                #We fill the queue with new fetched batch until we reach the max size.
                batches_queue.put((batch, (timestamp_batch, id_batch, vector_batch)), block=True)
                if tokill() == True:
                    return
          
            
    def threaded_cuda_batches(self, tokill, cuda_batches_queue, batches_queue):
        """
        Thread worker for transferring pytorch tensors into
        GPU. batches_queue is the queue that fetches numpy cpu tensors.
        cuda_batches_queue receives numpy cpu tensors and transfers them to GPU space.
        """
        while tokill() == False:
            batch, (timestamp_batch, id_batch, vector_batch) = batches_queue.get(block=True)
            vector_batch_GPU = vector_batch.to(self.device)
    
            cuda_batches_queue.put((batch, (timestamp_batch, id_batch, vector_batch_GPU)), block=True)
        
            if tokill() == True:
                return
         
      
  
    def scale(self, x):
        return x/max(x)
    
        
    # Create Topic Tensor and Weight tensor
    def create_topic_tensors(self, topic_vectors_dict, topic_df):
        topic_arrays  = []
        weight_arrays = [] 
        for i in range(0, len(topic_vectors_dict)):
            topic_vecs = np.array(topic_vectors_dict[f'Topic_{i}'])
            topic_arrays.append(torch.as_tensor(topic_vecs, dtype=torch.float16))

            # Weights (those are the cosine distances to the seed vector)
            weights = topic_df.loc[:, (f'Topic_{i}', 'cos_sim')].values
            weight_arrays.append(torch.from_numpy(weights))

        topic_tensor = torch.stack(topic_arrays)
        topic_tensor = topic_tensor.to(self.device)   

        weight_tensor = torch.stack(weight_arrays)
        weight_tensor = weight_tensor.to(self.device)  

        return weight_tensor, topic_tensor

    
       
    def calc_factor_loading_gpu(self, tensor3d, topic_tensor, weight_tensor):
        zero = torch.tensor([0.], dtype=torch.float16).to(self.device)
        factor_loadings_batch = []
        
        for topic_idx in range(0, topic_tensor.shape[0]):     
            dot_prod = torch.einsum('bij, jk-> bik', tensor3d, topic_tensor[topic_idx,:,:].T)          
            dot_prod = torch.where(dot_prod >= barrier_sim, dot_prod, zero)
            dot_prod = dot_prod * weight_tensor[topic_idx, :]
            dot_prod = torch.sum(dot_prod, dim=-1)
            dot_prod = torch.clamp(dot_prod, max=max_word_contr)
            dot_prod = torch.sum(dot_prod, dim=-1)
    
            factor_loadings_batch.append(dot_prod.detach().cpu().numpy())  
    
            del dot_prod
            torch.cuda.empty_cache()
    
        return np.vstack(factor_loadings_batch).T
    
         

    # Calculate Facor Loadings with cosine similarity method
    def calc_factor_loadings_CS_Method(self, 
                                       start_year, 
                                       end_year, 
                                       df_filter, 
                                       from_file_path, 
                                       max_topic_tokens, 
                                       filename, 
                                       batch_size, 
                                       chunksize_2, 
                                       bigram_model_src):
    
        if from_file_path != None:           
            with open(from_file_path + 'topic_vectors.json', 'r') as f:
                topic_vectors_dict = json.load(f)     
                
            topic_df  = pd.read_csv(from_file_path+'topics.csv', encoding='utf-8', header=[0,1])

        else: 
            Path(self.run_dir+'Factor Indices/').mkdir(parents=True, exist_ok=True) 
            
            # Create log file        
            with open(self.run_dir+'log_(factor_index).txt', 'w') as f:
                f.write(f"Factor Indices generated with Cosine Similarity Method\
                        \nStart year:         {start_year}\
                        \nEnd year:           {end_year}\
                        \nSelected countries: {df_filter['country']}\
                        \nIncluded subjects:  {df_filter['subjects']}\
                        \nExcluded subjects:  {df_filter['exclude_subj']}")
            
            with open(self.run_dir+'topic_vectors.json', 'r') as f:    
                topic_vectors_dict = json.load(f)          
        
            topic_df  = pd.read_csv(self.run_dir+'topics.csv', encoding='utf-8', header=[0,1])
        
        print(topic_vectors_dict.keys())
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Torch device: GPU")
        else: 
            self.device = torch.device("cpu")
            print("Torch device: CPU")
        
        
        weight_tensor, topic_tensor = self.create_topic_tensors(topic_vectors_dict, topic_df)
        
        dataloader_batches_queue    = Queue(maxsize=4)
        cuda_batches_queue          = Queue(maxsize=1)
        
        vocab           = self.vocab
        embeddings_dict = self.get_embeddings
            
        iterable_dataset = MyIterableDataset(
                                    file_dir        = self.docs_dir, 
                                    filename        = news_filename, 
                                    start_year      = start_year, 
                                    end_year        = end_year, 
                                    chunksize       = chunksize_2,
                                    batch_size      = batch_size,
                                    max_tokens      = max_topic_tokens,
                                    bigram_model    = bigram_model_src,
                                    df_filter       = df_filter,
                                    embeddings_dict = embeddings_dict,
                                    vocab           = vocab)
        
        
        # We launch 4 threads to do load && pre-process the input data
        dataloader_thread_killer = ThreadKiller()
        dataloader_thread_killer.set_tokill(False)
        preprocess_workers = 4
        
        for _ in range(preprocess_workers):
            t = Thread(target=self.threaded_batches_feeder, args=(dataloader_thread_killer, dataloader_batches_queue, iterable_dataset))
            t.start()
            
        # We launch 1 thread to transfer the data to GPU
        cuda_transfers_thread_killer = ThreadKiller()
        cuda_transfers_thread_killer.set_tokill(False)
        
        cudathread = Thread(target=self.threaded_cuda_batches, args=(cuda_transfers_thread_killer, cuda_batches_queue, dataloader_batches_queue))
        cudathread.start()
    
        #We let queue to get filled before we start the training
        time.sleep(6)
        
        #We fetch a GPU batch in 0's due to the queue mechanism
        i = 0
        while iterable_dataset.finished == False:
            _, (timestamp_batch, id_batch, vector_batch) = cuda_batches_queue.get(block=True)
              
            if i == 0:
                factor_loadings = self.calc_factor_loading_gpu(vector_batch, topic_tensor, weight_tensor)
                timestamps      = timestamp_batch
                ids             = id_batch
            else:
                factor_loadings = np.vstack([factor_loadings, self.calc_factor_loading_gpu(vector_batch, topic_tensor, weight_tensor)])
                timestamps.extend(timestamp_batch)
                ids.extend(id_batch)
            
            if i%10==0:
                print(f"Batch {i:<5} | {datetime.now().strftime('%H:%M:%S')}")
               
            i += 1
    
          
        dataloader_thread_killer.set_tokill(True)
        cuda_transfers_thread_killer.set_tokill(True)    
        for _ in range(preprocess_workers):
            try:
                #Enforcing thread shutdown
                dataloader_batches_queue.get(block=True,timeout=1)
                cuda_batches_queue.get(block=True,timeout=1)    
            except Empty:
                pass
        
        print("done")
    
        FreeGPU_Memory()
        
        # Create DataFrame from Factor Loadings
        factor_loadings_df = pd.merge(left=pd.DataFrame(data={'Timestamp':timestamps, 'ID':ids}), right=pd.DataFrame(factor_loadings.astype('float32'), columns=list(topic_vectors_dict)), left_index=True, right_index=True)
        

        if from_file_path != None:
            # Save factor_loadings_df 
            factor_loadings_df.to_csv(from_file_path+'factor_loadings_'+str(start_year)+'-'+str(end_year)+'.csv', encoding='utf-8', index=False)

        else: 
            # Save factor_loadings_df 
            factor_loadings_df.to_csv(self.run_dir+'factor_loadings_'+str(start_year)+'-'+str(end_year)+'.csv', encoding='utf-8', index=False)
    
 # ------------------------------------------------------------------------------------------------------------------------   
            
                
if __name__ == '__main__':
     
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_topics', type=str, required=False)
    parser.add_argument('--gen_index',  type=str, required=False)
    
    args = parser.parse_args()

    
    print('Initialize TextualFactors')
    tf = TextualFactors_v2(
        root_dir         = root_dir, 
        w2v_file         = w2v_file, 
        docs_dir         = docs_dir, 
        pca_dim          = pca_dim, 
        train_w2v        = train_w2v, 
        transf2unitvec   = transf2unitvec,
        add_polarity_dim = add_polarity_dim,
        method           = method
        )            
              

    # Generate Topics with the Cosine Similarity Method
    if args.gen_topics == 'cs':    
        tf.run_generate_topics(
            generate_word_clouds  = generate_word_clouds,
            word_cloud_size       = word_cloud_size, 
            word_cloud_pixel      = word_cloud_pixel, 
            word_cloud_shape      = word_cloud_shape
        )
        
        
    # Generate Topics with the Projection Method
    if args.gen_topics == 'proj':    
        tf.run_gen_topics_proj_method(
            params                = proj_method_params,
            generate_word_clouds  = generate_word_clouds,
            word_cloud_size       = word_cloud_size, 
            word_cloud_pixel      = word_cloud_pixel, 
            word_cloud_shape      = word_cloud_shape
        )
                
        
        
    if args.gen_index == 'True':
        tf.calc_factor_loadings_CS_Method(
            start_year        = start_year, 
            end_year          = end_year, 
            from_file_path    = from_file_path,          
            df_filter         = df_filter,
            max_topic_tokens  = max_topic_tokens,
            filename          = news_filename, 
            batch_size        = batch_size,
            chunksize_2       = chunksize_2,
            bigram_model_src  = bigram_model_src,
            )   
        
#%%         


        
