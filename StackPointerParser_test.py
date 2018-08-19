from __future__ import print_function

import sys
import os

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid
import json

import numpy as np
import torch
from neuronlp2.io import get_logger, conllx_stacked_data
from neuronlp2.models import StackPtrNet
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser

def main():
    args_parser = argparse.ArgumentParser(description='Testing with stack pointer parser')

    args_parser.add_argument('--model_path', help='path for parser model directory', required=True)
    args_parser.add_argument('--model_name', help='parser model file', required=True)
    args_parser.add_argument('--output_path', help='path for result with parser model', required=True)
    args_parser.add_argument('--test', required=True)
    args_parser.add_argument('--beam', type=int, default=1, help='Beam size for decoding')
    args_parser.add_argument('--use_gpu', action='store_true', help='use the gpu')
    args_parser.add_argument('--batch_size', type=int, default=32)

    args = args_parser.parse_args()

    logger = get_logger("PtrParser Decoding")
    model_path = args.model_path
    model_name = os.path.join(model_path, args.model_name)
    output_path = args.output_path
    beam = args.beam
    use_gpu = args.use_gpu
    test_path = args.test
    batch_size = args.batch_size

    def load_args():
        with open("{}.arg.json".format(model_name)) as f:
            key_parameters = json.loads(f.read())

        return key_parameters['args'], key_parameters['kwargs']

    # arguments = [word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, char_num_filters, char_window, eojul_num_filters, eojul_window,
    #              mode, input_size_decoder, hidden_size, encoder_layers, decoder_layers,
    #              num_types, arc_space, type_space]
    # kwargs = {'p_in': p_in, 'p_out': p_out, 'p_rnn': p_rnn, 'biaffine': True, 'pos': use_pos, 'char': use_char, 'eojul': use_eojul, 'prior_order': prior_order,
    #           'skipConnect': skipConnect, 'grandPar': grandPar, 'sibling': sibling}
    arguments, kwarguments = load_args()
    mode = arguments[10]
    input_size_decoder = arguments[11]
    hidden_size = arguments[12]
    arc_space = arguments[16]
    type_space = arguments[17]
    encoder_layers = arguments[13]
    decoder_layers = arguments[14]
    char_num_filters = arguments[6]
    eojul_num_filters = arguments[8]
    p_rnn = kwarguments['p_rnn']
    p_in = kwarguments['p_in']
    p_out = kwarguments['p_out']
    prior_order = kwarguments['prior_order']
    skipConnect = kwarguments['skipConnect']
    grandPar = kwarguments['grandPar']
    sibling = kwarguments['sibling']
    use_char = kwarguments['char']
    use_pos = kwarguments['pos']
    use_eojul = kwarguments['eojul']

    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(model_path, 'alphabets/')
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_stacked_data.load_alphabets(alphabet_path)
    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    logger.info("Reading Data")

    data_test = conllx_stacked_data.read_stacked_data_to_variable(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, use_gpu=use_gpu, prior_order=prior_order)
    num_data = sum(data_test[1])

    word_table = None
    word_dim = arguments[0]
    char_table = None
    char_dim = arguments[2]
    pos_table = None
    pos_dim = arguments[4]

    char_window = arguments[7]
    eojul_window = arguments[9]

    if arguments[1] != num_words:
        print("Mismatching number of word vocabulary({} != {})".format(arguments[1], num_words))
        exit()
    if arguments[3] != num_chars:
        print("Mismatching number of character vocabulary({} != {})".format(arguments[3], num_chars))
        exit()
    if arguments[5] != num_pos:
        print("Mismatching number of part-of-speech vocabulary({} != {})".format(arguments[5], num_pos))
        exit()
    if arguments[15] != num_types:
        print("Mismatching number types of vocabulary({} != {})".format(arguments[14], num_types))
        exit()

    network = StackPtrNet(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, char_num_filters, char_window, eojul_num_filters, eojul_window,
                          mode, input_size_decoder, hidden_size, encoder_layers, decoder_layers,
                          num_types, arc_space, type_space,
                          embedd_word=word_table, embedd_char=char_table, embedd_pos=pos_table, p_in=p_in, p_out=p_out, p_rnn=p_rnn,
                          biaffine=True, pos=use_pos, char=use_char, eojul=use_eojul, prior_order=prior_order,
                          skipConnect=skipConnect, grandPar=grandPar, sibling=sibling)

    if use_gpu:
        network.cuda()

    print("loading model: {}".format(model_name))
    if use_gpu:
        network.load_state_dict(torch.load(model_name))
    else:
        network.load_state_dict(torch.load(model_name, map_location='cpu'))

    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)

    logger.info("Embedding dim: word=%d, char=%d, pos=%d" % (word_dim, char_dim, pos_dim))
    logger.info("Char CNN: filter=%d, kernel=%d" % (char_num_filters, char_window))
    logger.info("Eojul CNN: filter=%d, kernel=%d" % (eojul_num_filters, eojul_window))
    logger.info("RNN: %s, num_layer=(%d, %d), input_dec=%d, hidden=%d, arc_space=%d, type_space=%d" % (
    mode, encoder_layers, decoder_layers, input_size_decoder, hidden_size, arc_space, type_space))
    logger.info("dropout(in, out, rnn): (%.2f, %.2f, %s)" % (p_in, p_out, p_rnn))
    logger.info('prior order: %s, grand parent: %s, sibling: %s, ' % (prior_order, grandPar, sibling))
    logger.info('skip connect: %s, beam: %d, use_gpu: %s' % (skipConnect, beam, use_gpu))

    network.eval()

    pred_filename = '%s/pred_test.txt' % (output_path, )
    pred_writer.start(pred_filename)
    gold_filename = '%s/gold_test.txt' % (output_path, )
    gold_writer.start(gold_filename)

    test_ucorrect = 0.0
    test_lcorrect = 0.0
    test_total = 0

    test_total_inst = 0

    test_root_correct = 0.0
    test_total_root = 0
    num_back = 0
    for batch in conllx_stacked_data.iterate_batch_stacked_variable(data_test, batch_size, use_gpu=use_gpu):
        input_encoder, _, sentences = batch
        word, char, pos, heads, types, masks, lengths = input_encoder

        heads_pred, types_pred, _, _ = network.decode(word, char, pos, mask=masks, length=lengths, beam=beam, leading_symbolic=conllx_stacked_data.NUM_SYMBOLIC_TAGS)

        word = word.data.cpu().numpy()
        pos = pos.data.cpu().numpy()
        lengths = lengths.cpu().numpy()
        heads = heads.data.cpu().numpy()
        types = types.data.cpu().numpy()

        pred_writer.write(sentences, word, pos, heads_pred, types_pred, lengths, symbolic_root=True)
        gold_writer.write(sentences, word, pos, heads, types, lengths, symbolic_root=True)

        stats, _, stats_root, num_inst = parser.eval(word, pos, heads_pred, types_pred, heads, types, word_alphabet, pos_alphabet, lengths, punct_set=None,
                                                                symbolic_root=True)
        ucorr, lcorr, total, _, _ = stats
        corr_root, total_root = stats_root

        test_ucorrect += ucorr
        test_lcorrect += lcorr
        test_total += total

        test_root_correct += corr_root
        test_total_root += total_root

        test_total_inst += num_inst

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)

        log_info = "({:.1f}%){}/{}".format(test_total_inst * 100 / num_data, test_total_inst, num_data)

        sys.stdout.write(log_info)
        sys.stdout.flush()
        num_back = len(log_info)

    pred_writer.close()
    gold_writer.close()

    sys.stdout.write("\b" * num_back)
    sys.stdout.write(" " * num_back)
    sys.stdout.write("\b" * num_back)

    print('----------------------------------------------------------------------------------------------------------------------------')
    print('best test W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
        test_ucorrect, test_lcorrect, test_total, test_ucorrect * 100 / test_total, test_lcorrect * 100 / test_total))
    print('best test Root: corr: %d, total: %d, acc: %.2f%%' % (test_root_correct, test_total_root, test_root_correct * 100 / test_total_root))
    print('============================================================================================================================')


main()