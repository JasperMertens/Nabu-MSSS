"""@file model_factory.py
contains the model factory"""

from . import dblstm, plain_variables, linear, concat, leaky_dblstm, sigmoid,relu, reconstruction_layer, \
  multi_averager, feedforward, leaky_dblstm_iznotrec, leaky_dblstm_notrec, dbrnn,\
  capsnet, dbr_capsnet, dbgru, leaky_dbgru, dbresetlstm, dlstm, dresetlstm,\
  leaky_dlstm, encoder_decoder_cnn, f_conv_capsnet, f_cnn, cnn_2d, conv2d_capsnet, \
    enc_dec_capsnet, enc_dec_capsnet_xl, conv2d_caps_sep, seg_caps, seg_cnn, seg_caps_lstm

def factory(architecture):
    """get a model class

    Args:
        conf: the model conf

    Returns:
        a model class"""

    if architecture == 'dblstm':
        return dblstm.DBLSTM
    elif architecture == 'leaky_dblstm':
        return leaky_dblstm.LeakyDBLSTM
    elif architecture == 'leaky_dblstm_iznotrec':
        return leaky_dblstm_iznotrec.LeakyDBLSTMIZNotRec
    elif architecture == 'leaky_dblstm_notrec':
        return leaky_dblstm_notrec.LeakyDBLSTMNotRec
    elif architecture == 'dbrnn':
        return dbrnn.DBRNN
    elif architecture == 'linear':
        return linear.Linear
    elif architecture == 'feedforward':
        return feedforward.Feedforward
    elif architecture == 'plain_variables':
        return plain_variables.PlainVariables
    elif architecture == 'concat':
        return concat.Concat
    elif architecture == 'multiaverage':
        return multi_averager.MultiAverager
    elif architecture == 'capsnet':
        return capsnet.CapsNet
    elif architecture == 'dbr_capsnet':
        return dbr_capsnet.DBRCapsNet
    elif architecture == 'f_conv_capsnet':
        return f_conv_capsnet.FConvCapsNet
    elif architecture == 'f_cnn':
        return f_cnn.FCNN
    elif architecture == 'cnn_2d':
        return cnn_2d.CNN2D
    elif architecture == 'dbgru':
        return dbgru.DBGRU
    elif architecture == 'leaky_dbgru':
        return leaky_dbgru.LeakyDBGRU
    elif architecture == 'dbresetlstm':
        return dbresetlstm.DBResetLSTM
    elif architecture == 'dlstm':
        return dlstm.DLSTM
    elif architecture == 'dresetlstm':
        return dresetlstm.DResetLSTM
    elif architecture == 'leaky_dlstm':
        return leaky_dlstm.LeakyDLSTM
    elif architecture == 'encoder_decoder_cnn':
        return encoder_decoder_cnn.EncoderDecoderCNN
    elif architecture == 'conv2d_capsnet':
        return conv2d_capsnet.Conv2DCapsNet
    elif architecture == 'enc_dec_capsnet':
        return enc_dec_capsnet.EncDecCapsNet
    elif architecture == 'enc_dec_capsnet_xl':
        return enc_dec_capsnet_xl.EncDecCapsNetXL
    elif architecture == 'conv2d_caps_sep':
        return conv2d_caps_sep.Conv2DCapsSep
    elif architecture == 'seg_caps':
        return seg_caps.SegCapsNet
    elif architecture == 'seg_cnn':
        return seg_cnn.SegCNN
    elif architecture == 'seg_caps_lstm':
        return seg_caps_lstm.SegCapsLSTM
    else:
        raise Exception('undefined architecture type: %s' % architecture)
