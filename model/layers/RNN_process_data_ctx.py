#-*- coding: utf-8 -*-

"""
what    : process data, generate batch
data    : twitter
"""
import numpy as np
import pickle
import random

from layers.RNN_params import Params

class ProcessData:

    # store data
    train_set = []
    dev_set = []
    test_set = []
    
    def __init__(self, data_path):
        
        self.data_path = data_path
        
        # load data
        self.train_set = self.load_data(Params.DATA_TRAIN_TRANS, Params.DATA_TRAIN_TYPE, Params.DATA_TRAIN_LABEL)
        self.dev_set  = self.load_data(Params.DATA_DEV_TRANS, Params.DATA_DEV_TYPE, Params.DATA_DEV_LABEL)
        self.test_set  = self.load_data(Params.DATA_TEST_TRANS, Params.DATA_TEST_TYPE, Params.DATA_TEST_LABEL)
        
        self.dic_size = 0
        # with open( data_path + Params.DIC, 'rb' ) as f:
        #     self.dic_size = len( pickle.load(f) )
        with open( data_path + "../" + Params.DIC, 'rb' ) as f:
            self.dic_size = len( pickle.load(f)['id2word'] )

            
    def load_data(self, text_trans, text_type, label):
     
        print ('load data : ' + text_trans + ' ' + text_type + ' ' + label)
        output_set = []

        tmp_text_trans        = np.load(self.data_path + text_trans) 
        # tmp_text_type         = np.load(self.data_path + text_type)        
        
        context_text  = [ x[:100] for x in tmp_text_trans ]
        original_text  = [ x[100:] for x in tmp_text_trans ]
        
        # context_type = [ np.where(x[:5]==1)[0][0] for x in tmp_text_type]
        # original_type = [ np.where(x[5:]==1)[0][0] for x in tmp_text_type]     
        
        tmp_label     = np.load(self.data_path + label)

        for i in range( len(tmp_label) ) :
            # output_set.append( [ context_text[i], context_type[i], original_text[i], original_type[i], tmp_label[i] ] )
            output_set.append( [ context_text[i], original_text[i], tmp_label[i] ] )
        print ('[completed] load data')
        
        return output_set
        
        
    def get_glove(self):
        # return np.load( self.data_path + Params.GLOVE )
        return np.load( self.data_path + "../" + Params.GLOVE )
        
    
    """
        inputs: 
            data            : data to be processed (train/dev/test)
            batch_size      : mini-batch size
            encoder_size    : max encoder time step
            
            is_test         : True, inference stage (ordered input)  ( default : False )
            start_index     : start index of mini-batch

        return:
            encoder_input_con   : [batch, time_step(==encoder_size)]
            encoder_seq_con     : [batch] - valid word sequence
            type_con                  : [batch] - tweet type
            
            encoder_input_ori   : [batch, time_step(==encoder_size)]
            encoder_seq_ori     : [batch] - valid word sequence
            type_ori                  : [batch] - tweet type
            
            labels                    : [batch, category] - category is one-hot vector
    """
    def get_batch(self, data, batch_size, encoder_size, is_test=False, start_index=0):

        # con_texts, con_seqs, con_types, ori_texts, ori_seqs, ori_types, labels = [], [], [], [], [], [], []
        con_texts, con_seqs, ori_texts, ori_seqs, labels = [], [], [], [], []
        index = start_index
        
        # Get a random batch of encoder and encoderR inputs from data,
        # pad them if needed

        for _ in range(batch_size):

            if is_test is False:
                # train case -  random sampling
                # con_text, con_type, ori_text, ori_type, label = random.choice(data)
                con_text, ori_text, label = random.choice(data)
                
            else:
                # dev, test case = ordered data
                if index >= len(data):
                    # con_text, con_type, ori_text, ori_type, label = data[0]  # won't be evaluated
                    con_text, ori_text, label = data[0]  # won't be evaluated
                    index += 1
                else: 
                    # con_text, con_type, ori_text, ori_type, label = data[index]
                    con_text, ori_text, label = data[index]
                    index += 1
            
            
            #if ori_type == 0: ori_type=5
            # no_context, plain, mention, reply, quote   --> 0,    1, 2, 3, 4  for context
            #    retweet, plain, mention, reply, quote   --> 0(5), 1, 2, 3, 4  for original
            # concat con, ori data
            
            # ori_type_np = np.zeros( 5, dtype=np.float32 )
            # ori_type_np[ori_type] = 1
            
            # con_type_np = np.zeros( 5, dtype=np.float32 )
            # con_type_np[con_type%5] = 1
            
            
            # find the seqN
            tmp_index = np.where( con_text == 0 )[0]   # find the pad index
            if ( len(tmp_index) > 0 ) :             # pad exists
                con_seqN =  np.min((tmp_index[0], encoder_size))
            else :                                  # no-pad
                con_seqN = 100
                
            # find the seqN
            tmp_index = np.where( ori_text == 0 )[0]   # find the pad index
            if ( len(tmp_index) > 0 ) :             # pad exists
                ori_seqN =  np.min((tmp_index[0], encoder_size))
            else :                                  # no-pad
                ori_seqN = 100

            
            # code to concat ( context, ori )
            tmp = np.zeros( 200, dtype=np.int )
            tmp[:con_seqN] = con_text[:con_seqN]
            tmp[con_seqN:(con_seqN+ori_seqN)] = ori_text[:ori_seqN]
            
            ori_text_fin = tmp
            ori_seqN = ori_seqN + con_seqN
            
            tmp_con = np.zeros( 200, dtype=np.int )
            tmp_con[:100] = con_text[:100]
            con_text_fin = tmp_con
            
            '''
            if ASSIGN_EMPYT_CONTEXT_TOK :
                if np.sum(con_text) == 0:
                    con_text[0] = self.dic_size -1
                    con_seqN = 1
            '''
                                
            con_texts.append( con_text_fin[:encoder_size] )
            # con_types.append( con_type_np )
            con_seqs.append( con_seqN )
            
            ori_texts.append( ori_text_fin[:encoder_size] )
            # ori_types.append( ori_type_np )
            ori_seqs.append( ori_seqN )
            
            labels.append( label )
            
        # return con_texts, con_seqs, con_types, ori_texts, ori_seqs, ori_types, labels
        return con_texts, con_seqs, ori_texts, ori_seqs, labels
    