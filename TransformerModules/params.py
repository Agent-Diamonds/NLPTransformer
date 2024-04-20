

params = {
    "num_layers" : 4, #l
    "hidden_size" : 256, #H for devlin, d_model for vaswani
    "num_attn_heads" :4, #A
    "seq_length" : 512,
    "batch_size" : 16,
    "vocab_size" : 50000,
    "dropout_prob" : 0.1,
    "num_epochs" : 40,
    "warmup_steps" : 10000,
    "lr" : 1e-4,
    "mlm_prob" : 0.15,
    "starting_step" : 0,
    "NSP_loss_scale" : 1,
    "MLM_loss_scale" : 1
}
assert(params["hidden_size"] % params["num_attn_heads"] == 0) #ensure d_model is properly split by attn heads
assert(params["NSP_loss_scale"] > 0)
assert(params["MLM_loss_scale"] > 0)


