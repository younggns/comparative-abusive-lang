#-*- coding: utf-8 -*-

"""
what    : evaluation
data    : twitter
"""

from tensorflow.core.framework import summary_pb2
from random import shuffle
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from layers.RNN_params import Params

"""
    desc  : 
    
    inputs: 
        sess  : tf session
        model : model for test
        data  : such as the dev_set, test_set...
            
    return:
        sum_batch_ce : sum cross_entropy
        accr         : accuracy
        
"""
def run_test(sess, model, batch_gen, data):
    
    list_batch_ce = []
    list_batch_correct = []
    
    list_pred = []
    list_label = []

    max_loop  = int(len(data) / model.batch_size)
    remaining = int(len(data) % model.batch_size)

    # evaluate data ( N of chunk (batch_size) + remaining( +1) )
    for test_itr in range( max_loop + 1 ):
        
        raw_encoder_input_con, raw_encoder_seq_con, raw_encoder_type_con, raw_encoder_input_ori, raw_encoder_seq_ori, raw_encoder_type_ori, raw_label = batch_gen.get_batch(
                                        data=data,
                                        batch_size=model.batch_size,
                                        encoder_size=model.encoder_size,
                                        is_test=True,
                                        start_index= (test_itr* model.batch_size)
                                        )
        
        # prepare data which will be push from pc to placeholder
        input_feed = {}

        input_feed[model.encoder_inputs_c] = raw_encoder_input_con
        input_feed[model.encoder_seq_c] = raw_encoder_seq_con
        input_feed[model.encoder_type_c] = raw_encoder_type_con

        input_feed[model.encoder_inputs_o] = raw_encoder_input_ori
        input_feed[model.encoder_seq_o] = raw_encoder_seq_ori
        input_feed[model.encoder_type_o] = raw_encoder_type_ori
        
        input_feed[model.y_labels] = raw_label
        
        input_feed[model.dr_prob] = 1.0             # no drop out while evaluating
        input_feed[model.dr_prob_ltc] = 1.0             # no drop out while evaluating
    
        if (test_itr == max_loop) & (remaining==0) :  # no remaining case
            break
    
        try:
            bpred, bloss = sess.run([model.batch_pred, model.batch_loss], input_feed)
        except:
            print ("excepetion occurs in valid step : " + str(test_itr))
            pass
        
        # remaining data case (last iteration)
        if test_itr == (max_loop):
            bpred = bpred[:remaining]
            bloss = bloss[:remaining]
            raw_label = raw_label[:remaining]
        
        # batch loss
        list_batch_ce.extend( bloss )
        
        # batch accuracy
        list_pred.extend( np.argmax(bpred, axis=1) )
        list_label.extend( np.argmax(raw_label, axis=1) )
        
    
    accr_class = precision_score(y_true=list_label,
                           y_pred=list_pred,
                           average=None)
        
    recall_class = recall_score(y_true=list_label,
                           y_pred=list_pred,
                           average=None)
        
    f1_class = f1_score(y_true=list_label,
                  y_pred=list_pred,
                  average=None)
    
    # macro : unweighted mean
    # weighted : 
    accr_avg = precision_score(y_true=list_label,
                           y_pred=list_pred,
                           average=Params.ACCURACY_AVG)
        
    recall_avg = recall_score(y_true=list_label,
                           y_pred=list_pred,
                           average=Params.RECALL_AVG)
        
    f1_avg = f1_score(y_true=list_label,
                  y_pred=list_pred,
                  average=Params.F1_AVG)
    
    result_zip = [accr_class, recall_class, f1_class, accr_avg, recall_avg, f1_avg]
    
    sum_batch_ce = np.sum( list_batch_ce )
    
    value1 = summary_pb2.Summary.Value(tag="valid_loss", simple_value=sum_batch_ce)
    value2 = summary_pb2.Summary.Value(tag="valid_accuracy", simple_value=accr_avg )
    summary = summary_pb2.Summary(value=[value1, value2])
    
    return sum_batch_ce, accr_avg, f1_avg, result_zip, summary

