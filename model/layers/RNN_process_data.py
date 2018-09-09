#-*- coding: utf-8 -*-

"""
what    : process data, generate batch
data    : twitter
"""
import numpy as np
import pickle
import random

from params import Params

class ProcessData:

    # store data
    train_set = []
    dev_set = []
    test_set = []
    
    def __init__(self, data_path):
        
        self.data_path = data_path
        
        # load data
        self.train_set = self.load_data(Params.DATA_TRAIN_TRANS, Params.DATA_TRAIN_LABEL)
        self.dev_set  = self.load_data(Params.DATA_DEV_TRANS, Params.DATA_DEV_LABEL)
        self.test_set  = self.load_data(Params.DATA_TEST_TRANS, Params.DATA_TEST_LABEL)
        
        self.dic_size = 0
        with open( data_path + Params.DIC ) as f:
            #self.dic_size = len( pickle.load(f)['id2word'] )
            self.dic_size = len( pickle.load(f) )

            
    def load_data(self, text_trans, label):
     
        print('load data : ' + text_trans  + ' ' + label )
        output_set = []

        tmp_text_trans        = np.load(self.data_path + text_trans)      
        
        context_text  = [ x[:100] for x in tmp_text_trans ]
        original_text  = [ x[100:] for x in tmp_text_trans ]   
        
        tmp_label     = np.load(self.data_path + label)

        for i in xrange( len(tmp_label) ) :
            output_set.append( [ context_text[i], original_text[i], tmp_label[i] ] )
        print('[completed] load data')
        
        return output_set
        
        
    def get_glove(self):
        return np.load( self.data_path + Params.GLOVE )
        
    def get_batch(self, data, batch_size, encoder_size, is_test=False, start_index=0):

        con_texts, con_seqs, ori_texts, ori_seqs, labels = [], [], [], [], []
        index = start_index
        
        # Get a random batch of encoder and encoderR inputs from data,
        # pad them if needed

        for _ in xrange(batch_size):

            if is_test is False:
                # train case -  random sampling
                con_text, ori_text, label = random.choice(data)
                
            else:
                # dev, test case = ordered data
                if index >= len(data):
                    con_text, ori_text, label = data[0]  # won't be evaluated
                    index += 1
                else: 
                    con_text, ori_text, label = data[index]
                    index += 1
                
            # find the seqN
            tmp_index = np.where( con_text == 0 )[0]   # find the pad index
            if ( len(tmp_index) > 0 ) :             # pad exists
                con_seqN =  np.min((tmp_index[0], encoder_size))
            else :                                  # no-pad
                con_seqN = encoder_size
                
            # find the seqN
            tmp_index = np.where( ori_text == 0 )[0]   # find the pad index
            if ( len(tmp_index) > 0 ) :             # pad exists
                ori_seqN =  np.min((tmp_index[0], encoder_size))
            else :                                  # no-pad
                ori_seqN = encoder_size


            if Params.ASSIGN_EMPYT_CONTEXT_TOK :
                if np.sum(con_text) == 0:
                    con_text[0] = self.dic_size -1
                    con_seqN = 1
                                
            con_texts.append( con_text[:encoder_size] )
            con_seqs.append( con_seqN )
            
            ori_texts.append( ori_text[:encoder_size] )
            ori_seqs.append( ori_seqN )
            
            #tmp_label = np.zeros( N_CATEGORY, dtype=np.float )
            #tmp_label[label] = 1
            labels.append( label )
            
        return con_texts, con_seqs, ori_texts, ori_seqs, labels
    