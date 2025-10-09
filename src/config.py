class Config:
    def __init__(self):
        #Embedding
        self.embedding_method = "GloVe"
        self.emb_batch = 512
        self.max_pre_pros_len = 128
        self.emb_lr = 0.0005
        self.emb_size = 200
        self.emb_epoch = 3
        self.emb_window_size = 5
        # Transformer
        self.batch_size = 64
        self.n_heads = 2
        self.n_layers = 2
        self.dropout = 0.2
        self.num_classes = 2
        self.lr = 0.00005
        self.wd = 0.003
        self.train_num_epoch = 7
        self.max_trans_len = 130
        self.label_smoothing = 0.0

