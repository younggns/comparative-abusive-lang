
###########################################
# SE train - Text
# each encoder_size = 100 
# ( context + original ) = 200
###########################################

CUDA_VISIBLE_DEVICES=1 python train_SE.py --batch_size 128 --encoder_size 100 --num_layer 1 --hidden_dim 100 --lr=0.001 --num_train_steps 10000 --is_save 0 --dr 0.8 --use_glove 1 --attn 1 --ltc 1 --graph_prefix 'SE_50d' --data_path '../data/target/0/' 


###########################################
# SE train - Character level
# original encoder_size = 140
# currently 
#    - no context
#    - no glove
#    - no type
###########################################

CUDA_VISIBLE_DEVICES=1 python train_SE_char.py --batch_size 128 --encoder_size 100 --num_layer 1 --hidden_dim 50 --lr=0.001 --num_train_steps 10000 --is_save 0 --dr 0.3 --use_glove 0 --o_type 0 --c_text 0 --c_type 0 --attn 0 --ltc 0 --graph_prefix 'SE_char' --data_path '../data/target/0/'



###########################################
# SE train - Single ( context has been added to original text )
# original encoder_size = 200   ( context + original )
###########################################

CUDA_VISIBLE_DEVICES=1 python train_SE_single.py --batch_size 128 --encoder_size 200 --num_layer 1 --hidden_dim 50 --lr=0.001 --num_train_steps 10000 --is_save 0 --dr 0.3 --use_glove 1 --o_type 0 --c_text 0 --c_type 0 --attn 0 --ltc 0 --graph_prefix 'SE_single_50d' --data_path '../data/target/0/' 
