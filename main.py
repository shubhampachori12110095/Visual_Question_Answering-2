import os
import sys
import numpy as np
import sklearn.metrics as metrics
import word2vec as w2v
import argparse
import time
import json

import utils
import nn_utils
from movieqa_importer import MovieQA

# Seed random number generators
rng = np.random
rng.seed(1234)

def dmn_start():
    print "==> parsing input arguments"
    parser = argparse.ArgumentParser()
    parser.add_argument('--story_source', type=str, default="subtitle", help='story source text: split_plot | dvs | subtitle | script')
    parser.add_argument('--learning_rate', type=float, default="0.001", help='set learning rate')
    parser.add_argument('--network', type=str, default="dmn_tied", help='embeding size (50, 100, 200, 300 only)')
    parser.add_argument('--word_vector_size', type=int, default=300, help='embeding size (50, 100, 200, 300 only)')
    parser.add_argument('--sent_vector_size', type=int, default=300, help='embeding size (50, 100, 200, 300 only)')
    parser.add_argument('--dim', type=int, default=300, help='number of hidden units in input module GRU')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--load_state', type=str, default="", help='state file path')
    parser.add_argument('--answer_module', type=str, default="feedforward", help='answer module type: feedforward or recurrent')

    parser.add_argument('--mode', type=str, default="train", help='mode: train or test. Test mode required load_state')
    #parser.add_argument('--mode', type=str, default="test", help='mode: train or test. Test mode required load_state')

    parser.add_argument('--input_mask_mode', type=str, default="sentence", help='input_mask_mode: word or sentence')
    parser.add_argument('--memory_hops', type=int, default=3, help='memory GRU steps')
    parser.add_argument('--batch_size', type=int, default=1, help='no commment')

    #parser.add_argument('--babi_id', type=str, default="1", help='babi task ID')
    #parser.add_argument('--babi_id', type=str, default="22", help='babi task ID')

    #parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
    parser.add_argument('--l2', type=float, default=0.0001, help='L2 regularization')
    #parser.add_argument('--l2', type=float, default=0.000001, help='L2 regularization')

    parser.add_argument('--normalize_attention', type=bool, default=False, help='flag for enabling softmax on attention vector')
    parser.add_argument('--log_every', type=int, default=1, help='print information every x iteration')
    parser.add_argument('--save_every', type=int, default=1, help='save state every x epoch')
    parser.add_argument('--prefix', type=str, default="", help='optional prefix of network name')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
    parser.add_argument('--babi_test_id', type=str, default="", help='babi_id of test set (leave empty to use --babi_id)')

    #parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate (between 0 and 1)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate (between 0 and 1)')

    #parser.add_argument('--dropout_in', type=float, default=0.1, help='dropout rate for input (between 0 and 1)')
    parser.add_argument('--dropout_in', type=float, default=0.0, help='dropout rate for input (between 0 and 1)')

    parser.add_argument('--batch_norm', type=bool, default=False, help='batch normalization')
    parser.set_defaults(shuffle=False)
    args = parser.parse_args()

    return args

def dmn_mid(args):
    #assert args.word_vector_size in [50, 100, 200, 300]
    print 'Evaluating Improved Dynamic Memory Networks on MovieQA using: %s' % args.story_source

    network_name = args.prefix + '%s.mh%d.n%d.bs%d%s%s%s' % (
        args.network, 
        args.memory_hops, 
        args.dim, 
        args.batch_size, 
        ".na" if args.normalize_attention else "", 
        ".bn" if args.batch_norm else "", 
        (".d" + str(args.dropout)) if args.dropout>0 else "")
    
    # Get list of MAs and movies
    mqa = MovieQA.DataLoader()

    ### Process story source
    stories, QAs = mqa.get_story_qa_data('full', args.story_source)
    stories = normalize_documents(stories)
    
    #babi_train_raw, babi_test_raw = utils.get_babi_raw(args.babi_id, args.babi_test_id)

    #word2vec = utils.load_glove(args.word_vector_size)
    #word2vec = {}

    args_dict = dict(args._get_kwargs())
    args_dict['stories'] = stories
    args_dict['QAs'] = QAs
    #args_dict['babi_train_raw'] = babi_train_raw
    #args_dict['babi_test_raw'] = babi_test_raw
    #args_dict['word2vec'] = word2vec

    if args.network == 'dmn_tied':
        import dmn_tied
        if (args.batch_size != 1):
            print "==> not using minibatch training in this mode"
            args.batch_size = 1
        dmn = dmn_tied.DMN_tied(**args_dict)

    elif args.network == 'dmn_untied':
        import dmn_untied
        if (args.batch_size != 1):
            print "==> not using minibatch training in this mode"
            args.batch_size = 1
        dmn = dmn_untied.DMN_untied(**args_dict)
        
    else:
        raise Exception("No such network known: " + args.network)
    

    if args.load_state != "":
        dmn.load_state(args.load_state)

    return args, network_name, dmn


def do_epoch(mode, epoch, skipped=0):
    # mode is 'train' or 'test'
    y_true = []
    y_pred = []
    avg_loss = 0.0
    prev_time = time.time()
    
    if mode == 'train':
        batches_per_epoch = len(dmn.train_range)
        perm = rng.permutation(dmn.train_range)
    elif mode == 'train_val':
        batches_per_epoch = len(dmn.train_val_range)
        perm = rng.permutation(dmn.train_val_range)
    elif mode == 'test':
        batches_per_epoch = len(dmn.val_range)
        perm = rng.permutation(dmn.val_range)

    for i, idx in enumerate(perm):
        step_data = dmn.step(idx, mode)
        prediction = step_data["prediction"]
        answers = step_data["answers"]
        current_loss = step_data["current_loss"]
        current_skip = (step_data["skipped"] if "skipped" in step_data else 0)
        log = step_data["log"]
        
        skipped += current_skip
        
        if current_skip == 0:
            avg_loss += current_loss
            
            for x in answers:
                y_true.append(x)
            
            for x in prediction.argmax(axis=1):
                y_pred.append(x)
            
            # TODO: save the state sometimes
            if (i % args.log_every == 0):
                cur_time = time.time()
                #%50 is ther so train/test doesn't take up too much terminal screen space 
                #if (i % 50) == 0:
                print ("  %sing: %d.%d / %d \t loss: %.3f \t avg_loss: %.3f \t skipped: %d \t %s \t time: %.2fs" % 
                        (mode, epoch, i * args.batch_size, batches_per_epoch * args.batch_size, 
                        current_loss, avg_loss / (i + 1), skipped, log, cur_time - prev_time))
                prev_time = cur_time
        
        if np.isnan(current_loss):
            print "==> current loss IS NaN. This should never happen :) " 
            exit()

    avg_loss /= batches_per_epoch
    print "\n  %s loss = %.5f" % (mode, avg_loss)
    print "confusion matrix:"
    print metrics.confusion_matrix(y_true, y_pred)
    
    accuracy = sum([1 if t == p else 0 for t, p in zip(y_true, y_pred)]) * 100.0 / batches_per_epoch / args.batch_size
    print "accuracy: %.2f percent" % accuracy
    
    return avg_loss, skipped, accuracy

def dmn_finish(args, network_name, dmn):
    acc_list = []
    max_acc = [0, 0]
    if args.mode == 'train':
        print "==> training"    
        skipped = 0
        for epoch in range(args.epochs):
            start_time = time.time()
            
            if args.shuffle:
                dmn.shuffle_train_set()
            
            _, skipped, train_acc = do_epoch('train', epoch, skipped)
            #epoch_loss_train, skipped = do_epoch('train', epoch, skipped)
            
            epoch_loss, skipped, test_acc = do_epoch('train_val', epoch, skipped)
            acc_list.append([epoch, train_acc, test_acc])
            if test_acc > max_acc[1]:
                max_acc[0] = epoch
                max_acc[1] = test_acc
            state_name = 'states/%s.epoch%d.test%.5f.state' % (network_name, epoch, epoch_loss)
            
            if (epoch % args.save_every == 0):    
                print "==> saving ... %s" % state_name
                dmn.save_params(state_name, epoch)
            print "current_max: " + str(max_acc[0]) + '\t' + str(max_acc[1])
            print "epoch %d took %.3fs" % (epoch, float(time.time()) - start_time)
            
        log_name = './acc_log_' + args.network + '_' + args.story_source + '_' + str(args.learning_rate) + '.txt'
        with open(log_name, 'w') as f_log:
            for acc in acc_list:
                output = str(acc[0]) + '\t' + str(acc[1]) + '\t' + str(acc[2]) + '\n'
                f_log.write(output)
            f_log.write('max: '+str(max_acc[0])+'\t'+str(max_acc[1]))

    elif args.mode == 'test':
        file = open('last_tested_model.json', 'w+')
        data = dict(args._get_kwargs())
        data["id"] = network_name
        data["name"] = network_name
        data["description"] = ""
        data["vocab"] = dmn.vocab.keys()
        json.dump(data, file, indent=2)
        do_epoch('test', 0)

    else:
        raise Exception("unknown mode")
    

def normalize_documents(stories, normalize_for=('lower', 'alphanumeric'), max_words=40):
    """Normalize all stories in the dictionary, get list of words per sentence.
    """

    for movie in stories.keys():
        for s, sentence in enumerate(stories[movie]):
            sentence = sentence.lower()
            if 'alphanumeric' in normalize_for:
                sentence = utils.normalize_alphanumeric(sentence)
            sentence = sentence.split(' ')[:max_words]
            stories[movie][s] = sentence
    return stories

if __name__ == "__main__":
    args = dmn_start()
    args, network_name, dmn = dmn_mid(args)
    dmn_finish(args, network_name, dmn)
