class Params ():


    ################################
    #     dataset
    ################################     
    DATA_TRAIN_LABEL           = 'train_Label.npy'
    DATA_TRAIN_TRANS          = 'train_CtxtText_InputText.npy'

    DATA_DEV_LABEL              = 'valid_Label.npy'
    DATA_DEV_TRANS             = 'valid_CtxtText_InputText.npy'

    DATA_TEST_LABEL            = 'test_Label.npy'
    DATA_TEST_TRANS           = 'test_CtxtText_InputText.npy'

    CHAR_DATA_TRAIN_LABEL           = 'train_Label.npy'      # chracter-level
    CHAR_DATA_TRAIN_TRANS = 'train_InputText.npy'

    CHAR_DATA_DEV_LABEL              = 'valid_Label.npy'
    CHAR_DATA_DEV_TRANS = 'valid_InputText.npy'

    CHAR_DATA_TEST_LABEL            = 'test_Label.npy'
    CHAR_DATA_TEST_TRANS = 'test_InputText.npy'

    DIC                                  = 'vocab.pkl'
    GLOVE                             = 'embedding.npy'

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
    
    is_context_use = False            # context use model
    is_minority_use = True                   # icml-18 fairness ML minority objectivev func implementation
    eta = 0.2                         # hyper-param for fairness ML obective func
    
    DIM_WORD_EMBEDDING  = 300          # when using glove it goes to 300 automatically
    DIM_TYPE_EMBEDDING  = 10
    DIM_FF_LAYER        = 50
    N_CATEGORY = 4                    # # of class in dataset
    reverse_bw = True
    ASSIGN_EMPYT_CONTEXT_TOK = True   # assign empty token when context is empty
    
    N_LTC_TOPIC = 2                   # for LTC method
    N_LTC_MEM_DIM = 100               # for LTC method
    LTC_dr_prob   = 0.8
    
    self_matching_layers = 1             
    self_matching_bidir   = True
    
    
    ################################
    #     MEASRE
    # macro     : unweighted mean (cal among each class and then average the results)
    # weighted : ignore class unbalance
    ################################
    ACCURACY_AVG = 'weighted'
    RECALL_AVG =  'weighted'
    F1_AVG = 'weighted'

