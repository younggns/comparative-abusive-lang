class Params ():


    ################################
    #     dataset
    ################################     
    DATA_TRAIN_LABEL           = 'Label_train.npy'
    DATA_TRAIN_TYPE            = 'CtxtType_InputType_train.npy'
    DATA_TRAIN_TRANS          = 'CtxtText_InputText_train.npy'

    DATA_DEV_LABEL              = 'Label_valid.npy'
    DATA_DEV_TYPE                = 'CtxtType_InputType_valid.npy'
    DATA_DEV_TRANS             = 'CtxtText_InputText_valid.npy'

    DATA_TEST_LABEL            = 'Label_test.npy'
    DATA_TEST_TYPE             = 'CtxtType_InputType_test.npy'
    DATA_TEST_TRANS           = 'CtxtText_InputText_test.npy'

    CHAR_DATA_TRAIN_LABEL           = 'Label_train.npy'      # chracter-level
    CHAR_DATA_TRAIN_TRANS = 'Char_InputText_train.npy'

    CHAR_DATA_DEV_LABEL              = 'Label_valid.npy'
    CHAR_DATA_DEV_TRANS = 'Char_InputText_valid.npy'

    CHAR_DATA_TEST_LABEL            = 'Label_test.npy'
    CHAR_DATA_TEST_TRANS = 'Char_InputText_test.npy'

    DIC                                  = 'dic.pkl'
    GLOVE                             = 'W_embedding.npy'

    ################################
    #     training
    ################################
    EMBEDDING_TRAIN        = True      # True is better (fine-tuning)
    CAL_ACCURACY_FROM      = 0         # run iteration without excuting validation
    MAX_EARLY_STOP_COUNT   = 3
    EPOCH_PER_VALID_FREQ   = 0.3
    
    ################################
    #     model
    ################################    
    
    is_text_encoding_bidir = True     # use bidir GRU for encoding text
    
    is_context_use   = False          # context use model
    is_context_bidir = False          # use bidir GRU for encoding context
    ASSIGN_EMPYT_CONTEXT_TOK = True   # assign empty token when context is empty
    
    is_minority_use = False           # icml-18 fairness ML minority objectivev func implementation
    eta = 0.2                         # hyper-param for fairness ML obective func

    self_matching_layers  = 1             
    self_matching_bidir   = True
    
    N_LTC_TOPIC   = 2                 # for LTC method
    N_LTC_MEM_DIM = 100               # for LTC method
    LTC_dr_prob   = 0.8               # for LTC method
    
    
    DIM_WORD_EMBEDDING  = 10          # when using glove it goes to 300 automatically
    DIM_TYPE_EMBEDDING  = 10
    DIM_FF_LAYER        = 50
    N_CATEGORY          = 4           # number of class in dataset
    
    reverse_bw          = True        # reverse backward output of the bidir GRU
    
    
    ################################
    #     MEASRE
    # macro     : unweighted mean (cal among each class and then average the results)
    # weighted : ignore class unbalance
    ################################
    ACCURACY_AVG = 'weighted'
    RECALL_AVG   = 'weighted'
    F1_AVG       = 'weighted'

