class Options:
    def __init__(self):
        # High-level options
        self.gpus = "0"
        self.mode = "train"
        self.checkpoint_version = -1
        
        # Model hyperperameters
        self.embedding_size = 128
        self.bert_intermediate_size = 512
        self.n_disambiguation_layers = 4
        self.n_prediction_layers = 8
        self.n_attention_heads = 8
        self.use_disambiguation_layer = True
        self.max_senses_per_word = 8
        self.ml_coeff = 0.1
        self.dl_r = 1.5
        self.dropout = 0.1
        
        # Training hyperparameters
        self.n_batches = 6000000
        self.batch_size = 64
        self.max_seq_len = 128
        self.max_gradient_norm = 5.0
        self.learning_rate = 0.00003
        self.lr_warmup_steps = 10000
        self.anneal_lr = True
        self.dl_warmup_steps = 2000000
        self.ml_warmup_steps = 1000000
        self.mask_prob = 0.15
        self.masking_policy = "0.8 0.1 0.1"
        
        # Preprocessing parameters
        self.min_occurrences_for_vocab = 500
        self.min_occurrences_for_polysemy = 20000
        self.lemmatize = True
        
        # Display parameters
        self.test_words = ""
        self.print_every = 100
        self.save_every = 500
        self.test_every = 500
        self.keep = 1
        
        # File and directory parameters
        self.model_dir = None
        self.corpus_path = None
        self.vocab_path = None
        self.n_senses_file = None
        self.pos_tagger_root = None
        
        # Evaluation parameters
        self.sense_prob_source = "prediction"
        
        # WSI
        self.wsi_path = ""
        self.wsi_format = ""
        self.wsi_2013_thresh = 0.2
        
        # WIC
        self.wic_train = True
        self.wic_data_path = ""
        self.wic_gold_path = ""
        self.wic_use_contextualized_reps = True
        self.wic_use_sense_probs = True
        self.wic_classification_threshold = -1.0
        self.wic_model_path = ""
        self.wic_output_path = ""
        
        # WSD
        self.wsd_train_path = ""
        self.wsd_eval_path = ""
        self.wsd_train_gold_path = None
        self.wsd_eval_gold_path = None
        self.wsd_use_training_data = True
        self.wsd_use_usage_examples = True
        self.wsd_use_definitions = True
        self.wsd_use_frequency_counts = True
        self.wsd_ignore_semeval_07 = True