# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 09:55:12 2021

@author: Stefan Salbrechter
"""

import os
import pandas as pd
import regex as re
import numpy as np
import multiprocessing
import argparse
from datetime import datetime
from pathlib import Path
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.models.word2vec import Word2Vec
from utilities import CleanNews, CleanNews_w2v, ToPercent, FilterNews, Dataloader, GetTotalExamples
from config import *



class TextualFactors_preprocessing():
    """
    This class contains all necessary preprocessing tasks for the Textual Factors class.
    Methods:
    clean_raw_news:        Clean raw news articles for subesquent tasks
    gen_w2v_training_data: Cleans raw news articles for the use of training the word2vec model. Output is a text file 
                           containing one sentence per line. Optionally it allows to generate also bigrams and trigrams
                           with the gensim Phraser model.
    generate_grams:        Generate bigrams and trigrams from unigrams
    train_word2vec:        Train the word2vec model
    """
    
    def __init__(self, root_dir, raw_data_dir, docs_dir, w2v_data_src, start_year, end_year):
        self.root_dir     = root_dir            # Textual Factors root directory
        self.raw_data_dir = raw_data_dir        # directory that contains the raw news articles
        self.docs_dir     = docs_dir            # directory that contains the cleaned news articles
        self.w2v_data_src = w2v_data_src        # directory that contains the word2vec training files
        self.start_year   = start_year
        self.end_year     = end_year
        
        Path(self.root_dir+'/models/').mkdir(parents=True, exist_ok=True) 
        Path(self.docs_dir).mkdir(parents=True, exist_ok=True) 
        Path(self.w2v_data_src).mkdir(parents=True, exist_ok=True) 
    
    
    def clean_raw_news(self, filename='news_data_fx_####'):
        print(f"Clean raw news articles ... {datetime.now().strftime('%H:%M:%S')}")
        
        for year in list(np.arange(self.start_year, self.end_year)):
            year = str(year)
            print(year)
            news = pd.read_csv(self.raw_data_dir + re.sub(r'####', str(year), filename)+'.csv', index_col=0, encoding='utf-8')  
            
            # Drop all news that contain no body text
            news = news.dropna(subset=['Body']).reset_index(drop=True)
            
            # Drop all news with duplicate headlines and keep the most recent one
            news = news[news.duplicated(subset='Headline', keep='first') == False]
            #news = news.reset_index(drop=True)

            # Clean Headlines and Body
            news.loc[:, 'Headline'] = news['Headline'].apply(CleanNews, to_lowercase=True)
            news.loc[:, 'Body']     =     news['Body'].apply(CleanNews, to_lowercase=True)

            # Convert price changes with the pattern 'to XX$ from XY$' to 'by YX percent'    
            news.loc[:, 'Headline'] = news['Headline'].apply(ToPercent)
            news.loc[:, 'Body']     =     news['Body'].apply(ToPercent)

            # Save to csv
            news.to_csv(self.docs_dir + re.sub(r'####', str(year), filename)+'_clean.csv', encoding='utf-8', index=True)
               
    
    class tiny_dataloader(object):
        def __init__(self, file_dir, fname):
            self.file_dir = file_dir
            self.fname    = fname
            
        def __iter__(self):    
            for i, line in enumerate(open(os.path.join(self.file_dir, self.fname))):
                yield line.split()
    
    
    def generate_grams(self, src_dir, output_dir, model_dir, min_count, threshold, gram_type='bigram'):
        """
        Generate bigrams or trigrams with gensim Phrases.
        src_dir:    directory that contains files with unigrams (to generate bigrams) or files with bigrams (to generate trigrams)
        output_dir: directory to save the generated bigram and trigram files
        model_dir:  directory to save the Phrase model
        min_count:  (float, optional) – Ignore all words and bigrams with total collected count lower than this value.
        threshold:  (float, optional) – Represent a score threshold for forming the phrases (higher means fewer phrases). 
                    A phrase of words a followed by b is accepted if the score of the phrase is greater than threshold.
        gram_type:  select either 'bigram' or 'trigram', default='bigram'
        """  
        files = sorted(os.listdir(src_dir))
        years = []
        for i, fname in enumerate(files):
            year = re.findall(r"\d\d\d\d", fname)[0]
            if int(year) >= self.start_year and int(year) <= self.end_year:
                years.append(year)
        print(years)
            
        print(f'Train the {gram_type} Phraser model ...')
        documents = Dataloader(path=src_dir, start_year=int(years[0]), end_year=int(years[-1]), split=True)
        model     = Phrases(documents, min_count=min_count, threshold=threshold, delimiter='_', scoring='default')

        # Export the trained model = use less RAM, faster processing. Model updates no longer possible.
        frozen_model = model.freeze()   
        
        # Add custom bigrams
        if len(custom_bigrams) >= 1:
            for g in self.custom_bigrams:
                frozen_model.phrasegrams[g] = float('inf')
        
        frozen_model.save(model_dir+gram_type+"_phrase_model_"+years[0]+'-'+years[-1]+".pkl")          
            
        print(f'Generate {gram_type} training files ...')
        for i, fname in enumerate(files):
            year = re.findall(r"\d\d\d\d", fname)[0]
            if int(year) >= self.start_year and int(year) <= self.end_year:
                documents = self.tiny_dataloader(file_dir=src_dir, fname=fname)
                
                with open(output_dir + "w2v_"+gram_type+"_phrases_"+year+".txt", 'w', encoding="utf-8") as f:
                    for i, sent in enumerate(documents):
                        f.write(' '.join(frozen_model[sent]))
                        f.write('\n')

            
        
    def gen_w2v_training_data(self, filename='news_data_fx_####', 
                               min_sent_len=5, generate_unigrams=True, generate_bigrams=True, generate_trigrams=True,
                               bigram_min_count=50, bigram_threshold=20, trigram_min_count=50, 
                               trigram_threshold=200, custom_bigrams=[]):
        
        """
        Clean News for word2vec and write to .txt files
        min_sent_len: Minium number of words that are considered as a sentence and are wirtten to the .txt files
        """
        
        self.custom_bigrams = custom_bigrams
        
        Path(self.w2v_data_src+'unigrams').mkdir(parents=True, exist_ok=True) 
                    
        if generate_unigrams:
            print(f"Clean raw news articles for word2vec ... {datetime.now().strftime('%H:%M:%S')}")
            for year in list(np.arange(self.start_year, self.end_year)):
                year = str(year)
                print(year)

                raw_news = pd.read_csv(self.raw_data_dir + re.sub(r'####', str(year), filename)+'.csv', index_col=0, encoding='utf-8')  

                # Drop all news that contain no body text
                raw_news = raw_news.dropna(subset=['Body']).reset_index(drop=True)

                # Drop all news with duplicate headlines and keep the most recent one
                raw_news = raw_news[raw_news.duplicated(subset='Headline', keep='last') == False]
                raw_news = raw_news.reset_index(drop=True)

                w2v_train      = pd.DataFrame(columns=['Body'])
                w2v_train.Body = raw_news.Body.apply(CleanNews_w2v, to_lowercase=True)
                w2v_train      = w2v_train.dropna().reset_index(drop=True)  

                with open(self.w2v_data_src+'unigrams/w2v_train_'+year+'.txt', 'w', encoding="utf-8") as f:
                    for article in w2v_train.Body:
                        l = re.split('(?<=\w\w)\.\s|\s\.\s|\.$|^\.|\*', article)
                        for s in l:
                            if len(s.split(' ')) > min_sent_len:
                                f.write('%s\n' % s.strip())
        
        
        if generate_bigrams:
            print(f"Generate bigrams ... {datetime.now().strftime('%H:%M:%S')}")
            Path(self.w2v_data_src+'bigrams').mkdir(parents=True, exist_ok=True) 
            self.generate_grams(src_dir    = self.w2v_data_src+'unigrams/', 
                                output_dir = self.w2v_data_src+'bigrams/',
                                model_dir  = self.root_dir+'models/',
                                min_count  = bigram_min_count, 
                                threshold  = bigram_threshold,
                                gram_type  ='bigram'
                               )
                          
        if generate_trigrams:
            print(f"Generate trigrams ... {datetime.now().strftime('%H:%M:%S')}")
            Path(self.w2v_data_src+'trigrams').mkdir(parents=True, exist_ok=True) 
            self.generate_grams(src_dir    = self.w2v_data_src+'bigrams/', 
                                output_dir = self.w2v_data_src+'trigrams/',
                                model_dir  = self.root_dir+'models/',
                                min_count  = trigram_min_count, 
                                threshold  = trigram_threshold,
                                gram_type  ='trigram'
                               )
                               
                
    def train_word2vec(self, w2v, model_name, gram_type='trigrams', epochs=10):
        """
        Train a Word2Vec model
        w2v:       dictionary containing the word2vec training parameters
        gram_type: select either 'unigrams', 'bigrams', 'trigrams'
        epochs:    number of epochs to train word2vec
        """
        print(f"Train the word2vec model {model_name}... {datetime.now().strftime('%H:%M:%S')}")
        examples  = GetTotalExamples(self.w2v_data_src+gram_type, y_start=self.start_year, y_end=self.end_year)
        sentences = Dataloader(self.w2v_data_src+gram_type, start_year=self.start_year, end_year=self.end_year, print_epoch=True, split=True) 
        self.w2v_model     = Word2Vec(sg=w2v['sg'], hs=w2v['hs'], vector_size=w2v['size'], 
                             negative=w2v['negative'],   window=w2v['window'], 
                             min_count=w2v['min_count'], alpha=w2v['alpha'],
                             min_alpha=w2v['min_alpha'], workers=w2v['workers'])

        # Train the model
        print('Build vocabulary ...')
        self.w2v_model.build_vocab(sentences)

        print('Train ...')
        self.w2v_model.train(sentences, total_examples=examples, epochs=epochs)  # optimal: 80 epochs

        # Save the model
        print('Save the model ...')
        self.w2v_model.save(self.root_dir+'/models/'+model_name+'.word2vec') 
        
        


if __name__ == '__main__':
     
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_w2v_training_files', type=str, required=False)
    parser.add_argument('--train_w2v', type=str, required=False)
    parser.add_argument('--clean_raw_news', type=str, required=False)
    
    args = parser.parse_args()        


    # Initialize    
    tf_pre = TextualFactors_preprocessing(
        root_dir     = root_dir,
        raw_data_dir = raw_data_dir,
        docs_dir     = docs_dir,
        w2v_data_src = w2v_data_src,
        start_year   = y_start, 
        end_year     = y_end        
        )
    
    
    if args.gen_w2v_training_files == 'True':
        tf_pre.gen_w2v_training_data(
            filename          = filename, 
            min_sent_len      = min_sent_len, 
            generate_unigrams = gen_unigrams, 
            generate_bigrams  = gen_bigrams, 
            generate_trigrams = gen_trigrams,
            bigram_min_count  = bigram_min_count, 
            bigram_threshold  = bigram_threshold, 
            trigram_min_count = trigram_min_count, 
            trigram_threshold = trigram_threshold,
            custom_bigrams    = custom_bigrams
        )

    
    if args.train_w2v == 'True':
        tf_pre.train_word2vec(
            w2v = w2v_params,
            model_name = w2c_model_name,
            gram_type  = gram_type,
            epochs     = n_epochs,
        )
    
    if args.clean_raw_news == 'True':
        tf_pre.clean_raw_news(
            filename   = filename
        )




#%%


# Initialize    
tf_pre = TextualFactors_preprocessing(
    root_dir     = root_dir,
    raw_data_dir = raw_data_dir,
    docs_dir     = docs_dir,
    w2v_data_src = w2v_data_src,
    start_year   = y_start, 
    end_year     = y_end        
    )


tf_pre.clean_raw_news(
    filename   = filename
)