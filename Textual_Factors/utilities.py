# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:03:07 2021

@author: Stefan Salbrechter, MSc
"""

import os
import numpy as np
import pandas as pd
import regex as re
import threading
import torch
from torch.nn import functional as F
from gensim.models import Phrases
from numpy.linalg import norm
from bs4 import BeautifulSoup 
from datetime import datetime
from torch.utils.data import IterableDataset
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='bs4')


def WordChar_vs_Digits(x):
    """
    Calculate the ratio of digits vs. word characters in the text.
    Set the article relvance to False if the digits count is > 1/2 * word character count
    """
    num_digit = len(re.findall(r'\d', x))
    num_alpha = len(re.findall('[A-Za-z]', x))
    if num_alpha/2 > num_digit:
        return True
    else:
        return False
    
    
def CleanNews(news, delete_duplicates=False, to_lowercase=True):
    """
    Clean raw news for subsequent tasks
    """   
    if pd.isnull(news) == False and news != 'NULL NULL' and news != ' ': 
        # Removing html tags
        x = BeautifulSoup(news).get_text(separator=' ')
        x = x.replace('\n', ' ')
        x = x.replace('\r', ' ')
        x = x.replace('\t', ' ')

        if WordChar_vs_Digits(x):
            # Remove content in brackets < > ( ) and \n
            x = re.sub(r'\([^)]*\)|\[[^\]]*\]|\{[^}]*\}', '', x).replace('\n', ' ')  #\<[^>]*\>|
            
            # Remove multiple --, == and ..
            x = re.sub(r'\s-*-\s|(?<=\w)-*-(?=[^A-Za-z0-9])|\b-+-\b|\s-+-\b|=*= |\.\.*\.|\"', ' ', x)

            # e.g. EXPLAINER-The case for -> EXPLAINER The case for 
            x = re.sub(r'\b[A-Z][A-Z][A-Z][A-Z]+\b-', ' ', x)

            # Lower case and strip:
            if to_lowercase:
                x = x.lower()

            # Remove Articles starting with "nyse order imbalance"
            x = re.sub(r'^nyse order imbalance.*', '', x, re.IGNORECASE)

            # Remove everything after:

            x = re.sub('(for a richer, multimedia version of top news visit:.*)|(for expanded, multimedia reuters top news visit:.*)', '', x, re.IGNORECASE)
            x = re.sub('for enquiries to customer help desks:.*', '', x, re.IGNORECASE)  
            x = re.sub('(visit:.*)|(eikon:.*)|(thomson one:.*)|(click:.*)|(note:.*)|(source text.*)|(source link:.*)|(editor:.*)|(keywords:.*)|(double click:.*)', '', x, re.IGNORECASE)

            # Remove URLs
            url_pattern = "http(.|)://(www.|)\S*\.\S*(\s|\s?)"
            while re.findall(url_pattern, x, re.IGNORECASE):
                x = re.sub(url_pattern, '', x, re.IGNORECASE)

            # Remove E-Mail adresses
            x = re.sub('\S*@\S*\s?', '', x)

            # Remove whitespaces before . and ,
            x = re.sub(r'(?<=\w) +\.', '.', x)
            x = re.sub(r'(?<=\w) +\,', ',', x)

            # Substitue
            x = re.sub(r'%', ' percent', x)

            # Remove everything, except
            x = re.sub(r'[^A-Za-z\-\.&]', ' ', x)

            # Remove '-' that appear not between letters
            x = re.sub(r"^\-\b|\s\-\b|\b\-\s|\b\-$|\s-\s", " ", x) 

            # Remove single characters
            x = re.sub(r"\s[a-zA-Z\-\.&]\s", " ", x)

            # Remove dots at the end of sentences
            x = re.sub(r'(?<=\w\w)\.\s?', ' ', x)

            # Remove 3rd, 2nd, 1st, 10th
            x = re.sub(r'\s(st|nd|rd|th)\s', ' ', x)

            # Remove duplicate whitespaces
            x = re.sub(' +', ' ', x)
            x = x.strip()   
        else:
            x = None
    else: 
        x = None
    return x
    
    
def CleanNews_w2v(news, delete_duplicates=False, to_lowercase=True):
    """
    Clean raw news for training with Word2Vec 
    """
    if pd.isnull(news) == False and news != 'NULL NULL' and news != ' ':  
        # Removing html tags
        x = BeautifulSoup(news).get_text(separator=' ')
        x = x.replace('\n', ' ')
        x = x.replace('\r', ' ')
        x = x.replace('\t', ' ')
        
        if WordChar_vs_Digits(x):

            # Remove content in brackets < > ( ) and \n
            x = re.sub(r'\([^)]*\)|\[[^\]]*\]|\{[^}]*\}', '', x).replace('\n', ' ')   # \<[^>]*\>|

            # Remove multiple --, == and ..
            x = re.sub(r'\s-*-\s|(?<=\w)-*-(?=[^A-Za-z0-9])|\b-+-\b|\s-+-\b|=*= |\.\.*\.|\"', ' ', x)
            
            # Replace multiple *** with *
            x = re.sub(r'\*+', '* ', x)
            
            # e.g. EXPLAINER-The case for -> EXPLAINER The case for 
            x = re.sub(r'\b[A-Z][A-Z][A-Z][A-Z]+\b-', ' ', x)

            # Lower case and strip:
            if to_lowercase:
                x = x.lower()

            # Remove Articles starting with "nyse order imbalance"
            x = re.sub(r'^nyse order imbalance.*', '', x, re.IGNORECASE)
           
            # Remove everything after:
            x = re.sub('(for a richer, multimedia version of top news visit:.*)|(for expanded, multimedia reuters top news visit:.*)', '', x, re.IGNORECASE)
            x = re.sub('for enquiries to customer help desks:.*', '', x, re.IGNORECASE)  
            x = re.sub('(visit:.*)|(eikon:.*)|(thomson one:.*)|(click:.*)|(note:.*)|(source text.*)|(source link:.*)|(editor:.*)|(keywords:.*)|(double click:.*)', '', x, re.IGNORECASE)

            # Remove URLs
            url_pattern = "http(.|)://(www.|)\S*\.\S*(\s|\s?)"
            while re.findall(url_pattern, x, re.IGNORECASE):
                x = re.sub(url_pattern, '', x, re.IGNORECASE)
                
            # Remove E-Mail adresses
            x = re.sub('\S*@\S*\s?', '', x)
            
            # Remove whitespaces before . and ,
            x = re.sub(r'(?<=\w) +\.', '.', x)
            x = re.sub(r'(?<=\w) +\,', ',', x)
            
            # Substitue
            x = re.sub(r'%', ' percent', x)
            
            # Remove all non alphabetic caracters
            x = re.sub("[^a-zA-Z\-\.\*&]", " ", x)
            
            # Remove '-' that appear not between letters
            x = re.sub(r"^\-\b|\s\-\b|\b\-\s|\b\-$|\s-\s", " ", x) 
        
            # Remove single characters
            x = re.sub(r"\s[a-zA-Z\-\.&]\s", " ", x)
            
            # Remove 3rd, 2nd, 1st, 10th
            x = re.sub(r'\s(st|nd|rd|th)\s', ' ', x)
        
            # Remove duplicate whitespaces
            x = re.sub(' +', ' ', x)
            x = x.strip()  
        
        else:
            x = None
    else: 
        x = None
    return x


# Convert to relative changes
def ToPercent(x):
    pattern = r'to\s\$\d*\.?\d*\sfrom\s\$\d*\.?\d*'
    try:
        matches = re.findall(pattern, x, re.IGNORECASE)
        if matches:
            for match in matches:
                num = re.findall(r'\b\d+\.?\d*', match, re.IGNORECASE)
                change = round((float(num[0])-float(num[1]))/float(num[1])*100,1)
                x = re.sub(r'(to|TO)\s\$'+str(num[0])+'\s(from|FROM)\s\$'+str(num[1]), f'by {abs(change)} percent', x, re.IGNORECASE)
    except:
        pass
    return x



def FilterNews(df, country, industry, company, topic, exclude_top):
    selcountry, selindustry, selcompany, seltopic, selexcl_top = True, True, True, True, True
    
    if country:
        for i, country_code in enumerate(country):
            if i == 0:
                selcountry = ((df.loc[:, 'Country_1':'Country_8'] == country_code).sum(axis=1) >= 1).to_frame(name=country_code)
            else:
                selcountry = pd.concat([selcountry, ((df.loc[:, 'Country_1':'Country_8'] == country_code).sum(axis=1) >= 1).to_frame(name=country_code)], axis=1)

        selcountry = selcountry.sum(axis=1) >= 1
        
    if industry:
        for i, industry_code in enumerate(industry):
            if i == 0:
                selindustry = ((df.loc[:, 'Industry_1':'Industry_4'] == industry_code).sum(axis=1) >= 1).to_frame(name=industry_code)
            else:
                selindustry = pd.concat([selindustry, ((df.loc[:, 'Industry_1':'Industry_4'] == industry_code).sum(axis=1) >= 1).to_frame(name=industry_code)], axis=1)

        selindustry = selindustry.sum(axis=1) >= 1        
        
    if company:
        if company[0] == 'ANY':
            selcompany  = df.loc[:, 'Company_1'].notna()
        else:
            for i, ticker in enumerate(company):
                if i == 0:
                    selcompany = ((df.loc[:, 'Company_1':'Company_15'] == ticker).sum(axis=1) >= 1).to_frame(name=ticker)
                else:
                    selcompany = pd.concat([selcompany, ((df.loc[:, 'Company_1':'Company_15'] == ticker).sum(axis=1) >= 1).to_frame(name=ticker)], axis=1)

            selcompany = selcompany.sum(axis=1) >= 1   
            
    if topic:
        seltopic    = df.loc[:, topic].sum(axis=1) >= 1
    if exclude_top:
        selexcl_top = df.loc[:, exclude_top].sum(axis=1) == 0
        
    if len(country) + len(industry) + len(company) + len(topic) + len(exclude_top) >= 1:
        return df.loc[seltopic & selcountry & selindustry & selcompany & selexcl_top]
    else:
        return df



def GetTotalExamples(dirname, y_start=None, y_end=None):
    print('Count total number of examples ...')
    total_examples = 0
    for fname in sorted(os.listdir(dirname)):
        year = int(re.findall(r'\d+', fname)[1])
        if y_start != None and y_end != None:
            if (year < y_start) or (year > y_end):
                continue
            
        print(year)
        with open(os.path.join(dirname, fname)) as f:
            for i, l in enumerate(f):
                pass
        total_examples += i + 1
    return total_examples   



class Dataloader(object):
    def __init__(self, path, start_year=None, end_year=None, print_epoch=False, split=False):
        self.dirname      = path
        self.epoch        = -1
        self.print_epoch  = print_epoch
        self.split        = split
        self.start_year   = start_year
        self.end_year     = end_year
        
    def __iter__(self):
        for fname in sorted(os.listdir(self.dirname)):
            year = int(re.findall(r'\d+', fname)[1])
            if self.start_year != None and self.end_year != None:
                if (year < self.start_year) or (year > self.end_year):
                    continue

            if self.print_epoch and year == self.start_year:
                self.epoch += 1
                print(f"Epoch {self.epoch :>1}. ({datetime.now().strftime('%H:%M:%S')})")
            print(f"Load data from {fname}")
            for i, line in enumerate(open(os.path.join(self.dirname, fname), encoding='utf-8')):
                if self.split:    
                    yield line.split()
                else:
                    yield line



# Thread safe iterator   
# https://sagivtech.com/2017/09/19/optimizing-pytorch-training-code/
class MyIterableDataset(IterableDataset):
    def __init__(self, 
                 file_dir, 
                 filename, 
                 start_year, 
                 end_year, 
                 chunksize, 
                 batch_size,
                 max_tokens, 
                 bigram_model,
                 df_filter,
                 embeddings_dict,
                 vocab):
        
        self.dirname    = file_dir
        self.start      = start_year
        self.end        = end_year
        self.filename   = filename
        self.chunksize  = chunksize
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.bigram_model    = Phrases.load(bigram_model)
        self.lock            = threading.Lock()
        self.finished        = False
        self.df_filter       = df_filter
        self.embeddings_dict = embeddings_dict
        self.vocab           = vocab
        
    
    def __iter__(self):
        with self.lock: 
            for year in list(np.arange(self.start, self.end)):
                print(year)
                fname  = re.sub(r'####', str(year), self.filename)+'.csv'                 
                df_obj = pd.read_csv(os.path.join(self.dirname, fname), iterator=True, chunksize=self.chunksize, dtype='object')
                
                for df_chunk in df_obj:
                    df_chunk = df_chunk.dropna(subset=['Body']).reset_index(drop=True)
                    df_chunk = df_chunk[df_chunk.duplicated(subset='Headline', keep='first') == False]
                    
                    if self.df_filter:
                        df_chunk = FilterNews(df_chunk, 
                                              country     = self.df_filter['country'], 
                                              industry    = self.df_filter['industry'], 
                                              company     = self.df_filter['company'], 
                                              topic       = self.df_filter['topic'], 
                                              exclude_top = self.df_filter['exclude_subj']
                                              )
                    
                    df_chunk = df_chunk.reset_index()
        
                    for i in range(0, df_chunk.shape[0], self.batch_size):
                        self.vecs_batch, self.timestamp_batch, self.id_batch = [], [], []
                        df_chunk.iloc[i:i+self.batch_size, :].apply(self.text2vec, axis=1)          
                        yield self.timestamp_batch, self.id_batch, torch.stack(self.vecs_batch)
                        
            self.finished = True
              
    
    def text2vec(self, x):   
        try:
            # Get bigrams
            grams_list = self.bigram_model[x.Body.split()[:self.max_tokens]]
            self.vecs_batch.append(self.pad_tensor(self.embeddings_dict[[w for w in grams_list if w in self.vocab]]))
            self.timestamp_batch.append(x.Timestamp) 
            self.id_batch.append(x.ID) 
        except:
            print('Warning: No array to concatonate.')
            pass
        
    
    def pad_tensor(self, vec):
        tensor = torch.as_tensor(vec, dtype=torch.float16)
        pad_tensor = (0, 0, 0, self.max_tokens-tensor.shape[0])
        return F.pad(tensor, pad_tensor, "constant", 0)



class ThreadKiller(object):
  """Boolean object for signaling a worker thread to terminate
  """
  def __init__(self):
    self.to_kill = False
  
  def __call__(self):
    return self.to_kill
  
  def set_tokill(self,tokill):
    self.to_kill = tokill
        
    
        
# Clear GPU Memory and Cache
def FreeGPU_Memory():
    try:
        del tensor3d
    except:
        pass
    try:
        del topic_tensor
    except:
        pass
    try:
        del dot_prod
    except:
        pass
    
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())
                
    
    
class SimilarityMeasure():
    def __init__(self, sim_measure='cos_similarity'):
        self.sim_measure = sim_measure
    
    def calc_similarity(self, X, Y):
        
        if self.sim_measure == 'cos_similarity':
            """
            Calculate the Cosine Similarity between X and Y
            X and Y can be either one- or two dimensional arrays
            """       
            if X.ndim > 1: 
                norm_X = norm(X, axis=1) 
            else: 
                norm_X = norm(X)
            if Y.ndim > 1: 
                norm_Y = norm(Y, axis=1) 
            else: 
                norm_Y = norm(Y)        

            if Y.ndim > X.ndim:
                cs = np.dot(Y,X)/(norm_X*norm_Y)
            else:
                cs = np.dot(X,Y)/(norm_X*norm_Y)       
            return cs
    

        elif self.sim_measure == 'cos_angle':     
            """
            Calculate the angle between two vectors
            """
            a = np.sqrt(np.dot(X,X))
            b = np.sqrt(np.dot(Y,Y))
            if a > b:
                cosine = np.arccos(b/a)
            elif b > a:
                cosine = np.arccos(a/b)
            else:
                cosine = 0
            return cosine
            
        else:
            print(f"{self.sim_measure} not available!")
            

class Get_PCA_Embds(object):
    def __init__(self, pca_embds):
        self.pca_embds = pca_embds
    
    def __getitem__(self, keys):
        if isinstance(keys, list):
            return np.vstack([self.pca_embds[key] for key in keys])
        else:
            return self.pca_embds[keys]  
        
        
class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = np.random.randn(self.hash_size, inp_dimensions)

    def generate_hash(self, inp_vector):
        bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        return ''.join(bools.astype('str'))

    def __setitem__(self, inp_vec, label):
        hash_value = self.generate_hash(inp_vec)
        self.hash_table[hash_value] = self.hash_table.get(hash_value, list()) + [label]

    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, [])


class LSH:
    def __init__(self, num_tables, hash_size, inp_dimensions):
        self.num_tables = num_tables
        self.hash_size  = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_tables = list()
        for i in range(self.num_tables):
            self.hash_tables.append(HashTable(self.hash_size, self.inp_dimensions))

    def __setitem__(self, inp_vec, label):
        for table in self.hash_tables:
            table[inp_vec] = label

    def __getitem__(self, inp_vec):
        results = list()
        for table in self.hash_tables:
            results.extend(table[inp_vec])
        return list(set(results))
    
    
def Unitvec(v):
    return v/norm(v)


def UnitColumns(v):
    return v/norm(v, axis=0)