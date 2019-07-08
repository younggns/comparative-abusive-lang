###########################################
# RNN-LTC (word-level)
# each encoder_size = 100 
# ( context + original ) = 200
###########################################

CUDA_VISIBLE_DEVICES=0 python train_rnn.py --batch_size 256 --encoder_size 100 --num_layer 1 --hidden_dim 200 --lr 0.001 --num_train_steps 10000 --is_save 0 --dr 0.8 --use_glove 1 --attn 0 --ltc 1 --graph_prefix 'rnn-ltc' --data_path '../data/target/0/'


###########################################
# RNN-ctx ( context has been added to original text )
# original encoder_size = 200   ( context + original )
###########################################

## CUDA_VISIBLE_DEVICES=0 python train_rnn_ctx.py --batch_size 256 --encoder_size 200 --num_layer 1 --hidden_dim 200 --lr 0.001 --num_train_steps 10000 --is_save 0 --dr 0.8 --use_glove 1 --o_type 0 --c_text 1 --c_type 0 --attn 0 --ltc 0 --graph_prefix 'rnn-ctx' --data_path '../data/target/0/' # deprecated
CUDA_VISIBLE_DEVICES=0 python train_rnn_ctx.py --batch_size 256 --encoder_size 200 --num_layer 1 --hidden_dim 200 --lr 0.001 --num_train_steps 10000 --is_save 0 --dr 0.8 --use_glove 1 --c_text 1 --attn 0 --ltc 0 --graph_prefix 'rnn-ctx' --data_path '../data/target/0/' 



###########################################
# RNN (char-level)
# original encoder_size = 140
# currently 
#    - no context
#    - no glove
#    - no type
###########################################

#################################################################################################################################
# We do not release the preporcessed dataset for this script. (data.tar.gz does not include files for following script)
#################################################################################################################################
CUDA_VISIBLE_DEVICES=0 python train_rnn_char.py --batch_size 256 --encoder_size 100 --num_layer 1 --hidden_dim 200 --lr 0.001 --num_train_steps 10000 --is_save 0 --dr 0.8 --use_glove 0 --attn 0 --ltc 0 --graph_prefix 'rnn-char' --data_path '../data/target/0/'
