namespace EzTensor{

    struct ModelArgs {
       int dim = 2048;
       float ffn_dim_multiplier = 1.5;
       int multiple_of = 256;
       int n_heads =32;
       int n_kv_heads = 8;
       int n_layers = 16;
       float norm_eps = 0.00001;
       float rope_theta = 500000.0;
       bool use_scaled_rope = true;
       int vocab_size = 128256;
       int max_seq_len = 128;
       int max_batch_size = 4;
    };









}