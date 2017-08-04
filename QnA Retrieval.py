# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:55:42 2016
@author: Dattatray Shinde
"""
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from string import digits
import os

'''Ímport data - Save the Sample comments in Folder Named as Text_MCD in Documents folder'''

os.chdir('D:\\')
folderDestination = os.path.join(os.getcwd(),"\\Sample Text.csv")
rAw = pd.read_csv("\\Sample Text.csv",sep = ',',header = 0, encoding='latin-1')
'''rAw.ix[1:6,'review']''' '''to print first 6 rows of reviews'''
rAw_ls = rAw.review.tolist()
rAw_ls = [str(i) for i in rAw_ls] # convert all elements to strings

# cleaning the raw text
''' Preprocessing of the raw text includes lemmatizing , removal of special characters
removal of numbers, English stopwords, tokenizing basis special characters and removing 
words having frequency of 1'''
def preProcess(lst):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    #stop_words.extend(custom_stopW)
    stop_words = set(stop_words)
    punc = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
    doc = [word.lower() for word in lst]
    words = [punc.tokenize(text) for text in doc] # tokenize basis punctuations
    specialChar = [",","."," ","?","!","'"] #  remove special characthers
    words = [[term for term in word if term not in specialChar] for word in words]
    words_filter = [[word for word in document if word not in stop_words]
                        for document in words]      
    words_nodig = [[word.strip(digits) for word in document]
                        for document in words_filter]
    words_nodig = [[word for word in document if  word !=" "]
                        for document in words_nodig]
    words_stem = [[lemmatizer.lemmatize(word) for word in document]
                        for document in words_nodig]
    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for document in words_stem:
        for token in document:
            frequency[token] += 1
    
    words_final = [[token for token in document if frequency[token] > 1]
                        for document in words_stem]
    return words_final

cLeaned = preProcess(rAw_ls)


'''Create Dictionary & Corpus'''
class myCorpus1(object):
    '''Create Dictionary & Corpus
the dictionary and corpus is getting saved on disk and hence it is agnostic of RAM'''
    def __init__(self,cLeaned,dictionary):
        self.cLeaned = cLeaned
        self.dictionary = dictionary
    def __iter__(self):
        for text in self.cLeaned:
            yield self.dictionary.doc2bow(text)

class myCorpus(object):
    
    '''Create Dictionary & Corpus
the dictionary and corpus is getting saved on disk and hence it is agnostic of RAM'''

    def __init__(self,lst):
        self.lst = lst
                        
    def createCorpus(self,dictPath,corpusPath):
        cLeaned = preProcess(self.lst)
        dictionary = corpora.Dictionary(word for word in cLeaned)
        dictionary.save(dictPath)
        corpus_memory_friendly = myCorpus1(cLeaned,dictionary)
        corpora.MmCorpus.serialize(corpusPath, corpus_memory_friendly)


corpus1 = myCorpus(rAw_ls) #  Initiatilize the instance
# Load the method to create and save dictionary and Market Matrix Corpus on Disk
corpus1.createCorpus('\\mcd.dict','\\mcd.mm')


''' Load Corpus'''
corpus = corpora.MmCorpus('\\mcd.mm')


''' Feature Vector Transformation'''

#lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=400)
#index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it

class BuildSimMatrix(object):
    
    def __init__(self,corpusPath,dictPath):
        self.corpus = corpora.MmCorpus(corpusPath)
        self.dict = corpora.Dictionary.load(dictPath)        

    def lsiSim(self,modelPath,indexPath):
        lsi = models.LsiModel(self.corpus, id2word=self.dict, num_topics=400)
        lsi.save(modelPath)        
        index = similarities.MatrixSimilarity(lsi[self.corpus]) # transform corpus to LSI space and index it
        index.save(indexPath)
        
    def tfIdfSim(self,modelPath,indexPath):
        tfidf = models.TfidfModel(self.corpus)
        tfidf.save(modelPath)
        index = similarities.MatrixSimilarity(tfidf[self.corpus]) # transform corpus to LSI space and index it
        index.save(indexPath)
        
        '''Random Projection indexing is an advanced method in retrival of  documents'''
    def randomProj(self,modelPath,indexPath):
        rp = models.RpModel(self.corpus)
        rp.save(modelPath)
        index = similarities.MatrixSimilarity(rp[self.corpus]) # transform corpus to LSI space and index it
         .save(indexPath)
    
    
  
temp = BuildSimMatrix('\\mcd.mm',
                      '\\mcd.dict')
temp.lsiSim('\\tfidf.model','\\lsiindex.index',)
temp.tfIdfSim('\\tfidf.model','\\tfidfindex.index')
temp.randomProj('\\rp.model','\\rpindex.index')
   
class QueryQuestion():
    def __init__(self):
        self.query = input("Type in your query:")
            
    def lsi_query(self,indexPath,dictPAth,modelPath,orignaltext_list):
        index = similarities.MatrixSimilarity.load(indexPath)
        dict = corpora.Dictionary.load(dictPAth)
        lsi = models.LsiModel.load(modelPath)       
        vec_query = dict.doc2bow(self.query.lower().split())
        vec_lsi = lsi[vec_query]
        #print(vec_lsi)
        sims = index[vec_lsi] # perform a similarity query against the corpus
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        temp = [sims[x] for x in list(range(5))]
        temp_values  = [x[0] for x in temp]
        keys = [x[1] for x in temp]        
        values = [orignaltext_list[x] for x in temp_values]
        simDocs = { k:v for (k,v) in zip(keys, values)}
        return print(simDocs)
    
    def tfidf_query(self,indexPath,dictPAth,modelPath,orignaltext_list):
        index = similarities.MatrixSimilarity.load(indexPath)
        dict = corpora.Dictionary.load(dictPAth)
        tfidf = models.LsiModel.load(modelPath)       
        vec_query = dict.doc2bow(self.query.lower().split())
        vec_tfidf = tfidf[vec_query]
        #print(vec_lsi)
        sims = index[vec_tfidf] # perform a similarity query against the corpus
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        temp = [sims[x] for x in list(range(5))]
        temp_values  = [x[0] for x in temp]
        keys = [x[1] for x in temp]        
        values = [orignaltext_list[x] for x in temp_values]
        simDocs = { k:v for (k,v) in zip(keys, values)}
        return print(simDocs)
    
    def rp_query(self,indexPath,dictPAth,modelPath,orignaltext_list):
        index = similarities.MatrixSimilarity.load(indexPath)
        dict = corpora.Dictionary.load(dictPAth)
        rp = models.LsiModel.load(modelPath)       
        vec_query = dict.doc2bow(self.query.lower().split())
        vec_rp = rp[vec_query]
        #print(vec_lsi)
        sims = index[vec_rp] # perform a similarity query against the corpus
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        temp = [sims[x] for x in list(range(5))]
        temp_values  = [x[0] for x in temp]
        keys = [x[1] for x in temp]        
        values = [orignaltext_list[x] for x in temp_values]
        simDocs = { k:v for (k,v) in zip(keys, values)}
        return print(simDocs)


''' Initiate below class and type in the question. The answer will be retrieved basis the 
Retrieval model chossen'''
temp_Q = QueryQuestion()
temp_Q.rp_query(' \\rpindex.index','\\mcd.dict',
                '\\rp.model',rAw_ls)
temp_Q.lsi_query('\\lsiindex.index','\\mcd.dict',
                '\\lsi.model',rAw_ls)
temp_Q.tfidf_query('\\tfidfindex.index','\\mcd.dict',
                '\\tfidf.model',rAw_ls)
