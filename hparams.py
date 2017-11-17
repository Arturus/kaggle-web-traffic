import tensorflow.contrib.training as training
import re

# Manually selected params
params_s32 = dict(
    batch_size=256,
    #train_window=380,
    train_window=283,
    train_skip_first=0,
    rnn_depth=267,
    use_attn=False,
    attention_depth=64,
    attention_heads=1,
    encoder_readout_dropout=0.4768781146510798,

    encoder_rnn_layers=1,
    decoder_rnn_layers=1,

    # decoder_state_dropout_type=['outside','outside'],
    decoder_input_dropout=[1.0, 1.0, 1.0],
    decoder_output_dropout=[0.975, 1.0, 1.0],  # min 0.95
    decoder_state_dropout=[0.99, 0.995, 0.995],  # min 0.95
    decoder_variational_dropout=[False, False, False],
    # decoder_candidate_l2=[0.0, 0.0],
    # decoder_gates_l2=[0.0, 0.0],
    #decoder_state_dropout_type='outside',
    #decoder_input_dropout=1.0,
    #decoder_output_dropout=1.0,
    #decoder_state_dropout=0.995, #0.98,  # min 0.95
    # decoder_variational_dropout=False,
    decoder_candidate_l2=0.0,
    decoder_gates_l2=0.0,

    fingerprint_fc_dropout=0.8232342370695286,
    gate_dropout=0.9967589439360334,#0.9786,
    gate_activation='none',
    encoder_dropout=0.030490422531402273,
    encoder_stability_loss=0.0,  # max 100
    encoder_activation_loss=1e-06, # max 0.001
    decoder_stability_loss=0.0, # max 100
    decoder_activation_loss=5e-06,  # max 0.001
)

# Default incumbent on last smac3 search
params_definc = dict(
    batch_size=256,
    train_window=100,
    train_skip_first=0,
    rnn_depth=128,
    use_attn=True,
    attention_depth=64,
    attention_heads=1,
    encoder_readout_dropout=0.4768781146510798,

    encoder_rnn_layers=1,
    decoder_rnn_layers=1,

    decoder_input_dropout=[1.0, 1.0, 1.0],
    decoder_output_dropout=[1.0, 1.0, 1.0],
    decoder_state_dropout=[0.995, 0.995, 0.995],
    decoder_variational_dropout=[False, False, False],
    decoder_candidate_l2=0.0,
    decoder_gates_l2=0.0,
    fingerprint_fc_dropout=0.8232342370695286,
    gate_dropout=0.8961710392091516,
    gate_activation='none',
    encoder_dropout=0.030490422531402273,
    encoder_stability_loss=0.0,
    encoder_activation_loss=1e-05,
    decoder_stability_loss=0.0,
    decoder_activation_loss=5e-05,
)

# Found incumbent 0.35503610596060753
#"decoder_activation_loss='1e-05'", "decoder_output_dropout:0='1.0'", "decoder_rnn_layers='1'", "decoder_state_dropout:0='0.995'", "encoder_activation_loss='1e-05'", "encoder_rnn_layers='1'", "gate_dropout='0.7934826952854418'", "rnn_depth='243'", "train_window='135'", "use_attn='1'", "attention_depth='17'", "attention_heads='2'", "encoder_readout_dropout='0.7711751356092252'", "fingerprint_fc_dropout='0.9693950737901414'"
params_foundinc = dict(
    batch_size=256,
    train_window=135,
    train_skip_first=0,
    rnn_depth=243,
    use_attn=True,
    attention_depth=17,
    attention_heads=2,
    encoder_readout_dropout=0.7711751356092252,

    encoder_rnn_layers=1,
    decoder_rnn_layers=1,

    decoder_input_dropout=[1.0, 1.0, 1.0],
    decoder_output_dropout=[1.0, 1.0, 1.0],
    decoder_state_dropout=[0.995, 0.995, 0.995],
    decoder_variational_dropout=[False, False, False],
    decoder_candidate_l2=0.0,
    decoder_gates_l2=0.0,
    fingerprint_fc_dropout=0.9693950737901414,
    gate_dropout=0.7934826952854418,
    gate_activation='none',
    encoder_dropout=0.0,
    encoder_stability_loss=0.0,
    encoder_activation_loss=1e-05,
    decoder_stability_loss=0.0,
    decoder_activation_loss=1e-05,
)

# 81 on smac_run0 (0.3552077534247418 x 7)
#{'decoder_activation_loss': 0.0, 'decoder_output_dropout:0': 0.85, 'decoder_rnn_layers': 2, 'decoder_state_dropout:0': 0.995,
# 'encoder_activation_loss': 0.0, 'encoder_rnn_layers': 2, 'gate_dropout': 0.7665920904244501, 'rnn_depth': 201,
#  'train_window': 143, 'use_attn': 1, 'attention_depth': 17, 'attention_heads': 2, 'decoder_output_dropout:1': 0.975,
# 'decoder_state_dropout:1': 0.99, 'encoder_dropout': 0.0304904225, 'encoder_readout_dropout': 0.4444295965935664, 'fingerprint_fc_dropout': 0.26412480387331017}
params_inst81 = dict(
    batch_size=256,
    train_window=143,
    train_skip_first=0,
    rnn_depth=201,
    use_attn=True,
    attention_depth=17,
    attention_heads=2,
    encoder_readout_dropout=0.4444295965935664,

    encoder_rnn_layers=2,
    decoder_rnn_layers=2,

    decoder_input_dropout=[1.0, 1.0, 1.0],
    decoder_output_dropout=[0.85, 0.975, 1.0],
    decoder_state_dropout=[0.995, 0.99, 0.995],
    decoder_variational_dropout=[False, False, False],
    decoder_candidate_l2=0.0,
    decoder_gates_l2=0.0,
    fingerprint_fc_dropout=0.26412480387331017,
    gate_dropout=0.7665920904244501,
    gate_activation='none',
    encoder_dropout=0.0304904225,
    encoder_stability_loss=0.0,
    encoder_activation_loss=0.0,
    decoder_stability_loss=0.0,
    decoder_activation_loss=0.0,
)
# 121 on smac_run0 (0.3548671560628074 x 3)
# {'decoder_activation_loss': 1e-05, 'decoder_output_dropout:0': 0.975, 'decoder_rnn_layers': 2, 'decoder_state_dropout:0': 1.0,
# 'encoder_activation_loss': 1e-05, 'encoder_rnn_layers': 1, 'gate_dropout': 0.8631496699358483, 'rnn_depth': 122,
#  'train_window': 269, 'use_attn': 1, 'attention_depth': 29, 'attention_heads': 4, 'decoder_output_dropout:1': 0.975,
# 'decoder_state_dropout:1': 0.975, 'encoder_readout_dropout': 0.9835390239895767, 'fingerprint_fc_dropout': 0.7452161827064421}

# 83 on smac_run1 (0.355050330259362 x 7)
# {'decoder_activation_loss': 1e-06, 'decoder_output_dropout:0': 0.925, 'decoder_rnn_layers': 2, 'decoder_state_dropout:0': 0.98,
#  'encoder_activation_loss': 1e-06, 'encoder_rnn_layers': 1, 'gate_dropout': 0.9275441207192259, 'rnn_depth': 138,
# 'train_window': 84, 'use_attn': 1, 'attention_depth': 52, 'attention_heads': 2, 'decoder_output_dropout:1': 0.925,
# 'decoder_state_dropout:1': 0.98, 'encoder_readout_dropout': 0.6415488109353416, 'fingerprint_fc_dropout': 0.2581296623398802}


params_inst83 = dict(
    batch_size=256,
    train_window=84,
    train_skip_first=0,
    rnn_depth=138,
    use_attn=True,
    attention_depth=52,
    attention_heads=2,
    encoder_readout_dropout=0.6415488109353416,

    encoder_rnn_layers=1,
    decoder_rnn_layers=2,

    decoder_input_dropout=[1.0, 1.0, 1.0],
    decoder_output_dropout=[0.925, 0.925, 1.0],
    decoder_state_dropout=[0.98, 0.98, 0.995],
    decoder_variational_dropout=[False, False, False],
    decoder_candidate_l2=0.0,
    decoder_gates_l2=0.0,
    fingerprint_fc_dropout=0.2581296623398802,
    gate_dropout=0.9275441207192259,
    gate_activation='none',
    encoder_dropout=0.0,
    encoder_stability_loss=0.0,
    encoder_activation_loss=1e-06,
    decoder_stability_loss=0.0,
    decoder_activation_loss=1e-06,
)

def_params = params_s32

sets = {
    's32':params_s32,
    'definc':params_definc,
    'foundinc':params_foundinc,
    'inst81':params_inst81,
    'inst83':params_inst83,
}


def build_hparams(params=def_params):
    return training.HParams(**params)


def build_from_set(set_name):
    return build_hparams(sets[set_name])


