import numpy as np
import theano
import theano.tensor as T
import lasagne
import pickle
import sys
import time
import utils
import config
import logging
import nn_layers
import lasagne.layers as L
from nn_layers import QuerySliceLayer
from nn_layers import AttentionSumLayer
from nn_layers import GatedAttentionLayerWithQueryAttention
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn import preprocessing
from nltk import sent_tokenize, word_tokenize
from theano.compile.nanguardmode import NanGuardMode
from lasagne.nonlinearities import LeakyRectify
from lasagne.nonlinearities import sigmoid
from lasagne.nonlinearities import elu
from lasagne.nonlinearities import rectify
from lasagne.regularization import l1
from lasagne.regularization import regularize_layer_params
import pandas as pd

def gen_examples(x1, x2, x3, y, x4, batch_size, concat=False):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = utils.get_minibatches(len(x1), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_x1 = [x1[t] for t in minibatch]
        mb_x2 = [x2[t] for t in minibatch]
        mb_x3 = [x3[t * 4 + k] for t in minibatch for k in range(4)]
        mb_y = [y[t] for t in minibatch]
        mb_x4 = [x4[t] for t in minibatch]
        
        mb_x1, mb_mask1 = utils.prepare_data(mb_x1)
        mb_x2, mb_mask2 = utils.prepare_data(mb_x2)
        mb_x3, mb_mask3 = utils.prepare_data(mb_x3)
        mea_num = mb_x4[0].shape[-1]
        mb_x4, mb_mask4 = utils.prepare_data(mb_x4, mea_num)
        all_ex.append((mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3, mb_y, mb_x4, mb_mask4))
    return all_ex


def build_fn(args, embeddings):
    """
        Build training and testing functions.
    """
    in_x1 = T.imatrix('x1')
    in_x3 = T.imatrix('x3')
    in_mask1 = T.matrix('mask1')
    in_mask3 = T.matrix('mask3')
    in_y = T.ivector('y')
    
    #batch x word_num x mea_num
    in_x4 = T.ftensor3('x4')

    l_in1 = lasagne.layers.InputLayer((None, None), in_x1)
    l_mask1 = lasagne.layers.InputLayer((None, None), in_mask1)
    l_emb1 = lasagne.layers.EmbeddingLayer(l_in1, args.vocab_size,
                                           args.embedding_size, W=embeddings)
	
    l_in3 = lasagne.layers.InputLayer((None, None), in_x3)
    l_mask3 = lasagne.layers.InputLayer((None, None), in_mask3)
    l_emb3 = lasagne.layers.EmbeddingLayer(l_in3, args.vocab_size,
                                           args.embedding_size, W=l_emb1.W)
									   
    l_in4 = lasagne.layers.InputLayer((None, None, args.mea_num), in_x4)
    if not args.tune_embedding:
        l_emb1.params[l_emb1.W].remove('trainable')
        l_emb3.params[l_emb3.W].remove('trainable')

	assert args.model is None

	#weighted mean: passage embedding
    if args.freezeMlP:
        weight_mlp_np = np.array([[1.]])
        b_mlp = np.array([0.])
        l_weight = lasagne.layers.DenseLayer(l_in4, 1, num_leading_axes=-1, 
                                             W=weight_mlp_np, b=b_mlp, nonlinearity=None) 
        l_weight.params[l_weight.W].remove('trainable')
        l_weight.params[l_weight.b].remove('trainable')
    else:
#        weight_mlp_np = np.zeros((args.mea_num, 1)) + 0.01*np.random.randn(args.mea_num, 1)
        weight_mlp_np = np.zeros((args.mea_num, 1))
        weight_mlp_np[-5] = 1.
        b_mlp = np.array([0.])
#        l_weight = lasagne.layers.DenseLayer(l_in4, 1, num_leading_axes=-1, 
#                                             nonlinearity=args.actiMlP)
#        l_weight = lasagne.layers.DenseLayer(l_in4, 1, num_leading_axes=-1, 
#                                             W=weight_mlp_np, b=b_mlp, 
#                                             nonlinearity=None)
#        l_weight = lasagne.layers.DenseLayer(l_in4, 1, num_leading_axes=-1, 
#                                             nonlinearity=None)
        l_weight1 = lasagne.layers.DenseLayer(l_in4, 1, num_leading_axes=-1, 
                                             W=weight_mlp_np, b=b_mlp, 
                                             nonlinearity=None)
        l_weight = nn_layers.WeightedNormLayer(l_weight1)
        
#        l_weight.params[l_weight.W].remove('trainable')
#        l_weight.params[l_weight.b].remove('trainable')
#        l_weight = lasagne.layers.DenseLayer(l_in4, 1, num_leading_axes=-1, 
#                                             W=lasagne.init.Constant(0.), b=lasagne.init.Constant(1.), 
#                                             nonlinearity=args.actiMlP)
#        l_weight.params[l_weight.W].remove('trainable')
    
#    weight_mlp_np = np.zeros((15, 1))
#    weight_mlp_np[-2] = 1.
#    weight_mlp_np = np.array([[1.]])
#    b_mlp = np.array([0.])
#    l_weight = lasagne.layers.DenseLayer(l_in4, 1, num_leading_axes=-1, 
#                                         W=weight_mlp_np, b=b_mlp, nonlinearity=None)
#    l_weight1 = lasagne.layers.DenseLayer(l_in4, 2, num_leading_axes=-1, nonlinearity=LeakyRectify(0.1))
#    l_weight = lasagne.layers.DenseLayer(l_weight1, 1, num_leading_axes=-1, nonlinearity=sigmoid)
#    l_weight.params[l_weight.W].remove('trainable')
#    l_weight.params[l_weight.b].remove('trainable')
    att = nn_layers.WeightedAverageLayer([l_emb1, l_weight, l_mask1])
	#mean: option embedding
    network3 = nn_layers.AveragePoolingLayer(l_emb3, mask_input=l_mask3)
    network3 = lasagne.layers.ReshapeLayer(network3, (in_x1.shape[0], 4, args.embedding_size))
	#predict answer
    network = nn_layers.DotLayer([network3, att], args.embedding_size)
    if args.pre_trained is not None:
        dic = utils.load_params(args.pre_trained)
        lasagne.layers.set_all_param_values(network, dic['params'])
        del dic['params']
        logging.info('Loaded pre-trained model: %s' % args.pre_trained)
        for dic_param in dic.iteritems():
            logging.info(dic_param)

    logging.info('#params: %d' % lasagne.layers.count_params(network, trainable=True))
    logging.info('#fixed params: %d' % lasagne.layers.count_params(network, trainable=False))
    for layer in lasagne.layers.get_all_layers(network):
        logging.info(layer)

    # Test functions
    weight = lasagne.layers.get_output(l_weight, deterministic=True)
    
    test_prob = lasagne.layers.get_output(network, deterministic=True)
    loss_test = lasagne.objectives.categorical_crossentropy(test_prob, in_y).mean()
    test_prediction = T.argmax(test_prob, axis=-1)
    acc = T.sum(T.eq(test_prediction, in_y))
#    test_fn = theano.function([in_x1, in_mask1, in_x3, in_mask3, in_y, in_x4],
#                              [acc, test_prediction, test_prob], on_unused_input='warn', 
#                              mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    test_fn = theano.function([in_x1, in_mask1, in_x3, in_mask3, in_y, in_x4],
                              [acc, test_prediction, test_prob, weight, loss_test], on_unused_input='warn')

    # Train functions
    train_prediction = lasagne.layers.get_output(network)
    train_prediction = T.clip(train_prediction, 1e-7, 1.0 - 1e-7)
    loss = lasagne.objectives.categorical_crossentropy(train_prediction, in_y).mean()
    
    # TODO: lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
#    l1_penalty = regularize_layer_params(l_weight, l1) * 1e-4
#    loss = loss + l1_penalty
#    params = lasagne.layers.get_all_params(network)#, trainable=True)
    params = lasagne.layers.get_all_params(network, trainable=True)
    all_params = lasagne.layers.get_all_params(network)
    if args.optimizer == 'sgd':
        updates = lasagne.updates.sgd(loss, params, args.learning_rate)
    elif args.optimizer == 'adam':
        updates = lasagne.updates.adam(loss, params, learning_rate=args.learning_rate)
    elif args.optimizer == 'rmsprop':
        updates = lasagne.updates.rmsprop(loss, params, learning_rate=args.learning_rate)
    else:
        raise NotImplementedError('optimizer = %s' % args.optimizer)
#    train_fn = theano.function([in_x1, in_mask1, in_x3, in_mask3, in_y, in_x4],
#                               loss, updates=updates, on_unused_input='warn', 
#                               mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    train_fn = theano.function([in_x1, in_mask1, in_x3, in_mask3, in_y, in_x4],
                               loss, updates=updates, on_unused_input='warn')

    return train_fn, test_fn, params, all_params


def eval_acc(test_fn, all_examples):
    """
        Evaluate accuracy on `all_examples`.
    """
    acc = 0
    n_examples = 0
    prediction = []
    test_probs = []
    weights = []
    ys = []
    loss_tests = []
    for x1, mask1, x2, mask2, x3, mask3, y, x4, mask4 in all_examples:
        tot_acc, pred, test_prob, weight, loss_test = test_fn(x1, mask1, x3, mask3, y, x4)
        acc += tot_acc
        ys += y
        prediction += pred.tolist()
        test_probs += test_prob.tolist()
        weights += weight.tolist()
        loss_tests.append(loss_test)
        n_examples += len(x1)
    return acc * 100.0 / n_examples, ys, prediction, test_probs, weights, np.mean(np.array(loss_tests))

#def eval_loss(test_fn, all_examples):
#    """
#        Evaluate accuracy on `all_examples`.
#    """
#    loss_es = 0
#    n_examples = 0
#    for idx, (mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, 
#              mb_mask3, mb_y, mb_x4, mb_mask4) in enumerate(all_examples):
#        tot_acc, pred, test_prob, weight = test_fn(mb_x1, mb_mask1, mb_x3, mb_mask3, mb_y, mb_x4)
#        loss_test = lasagne.objectives.categorical_crossentropy(test_prob, np.array(mb_y)).mean().eval()
##        loss = train_fn(mb_x1, mb_mask1, mb_x3, mb_mask3, mb_y, mb_x4)
#        loss_es = loss_es + loss_test * mb_x1.shape[0]
#        n_examples += mb_x1.shape[0]
#    return loss_es/n_examples

def attention_func(attention_fn, all_examples):
    """
        Get attention weights.
    """
    alpha = []
    for x1, mask1, x2, mask2, x3, mask3, y in all_examples:
        alpha_ex = attention_fn(x1, mask1, x2, mask2)
        alpha.append(alpha_ex)
    return alpha

def tuple_part(tuple_x, y_index):
    """get part of tuple
    
    """
    tuple_y = ()
    for ele_i, ele_x in enumerate(tuple_x):
        if ele_i != 2:
            ele_y = [ele_tmp for i, ele_tmp in enumerate(ele_x) if i in y_index]
        else:
            ele_y = [ele_tmp for i, ele_tmp in enumerate(ele_x) if int(i/4) in y_index]
        tuple_y +=  (ele_y,)
    return tuple_y

def Prepocessing_func(Feas_train, Feas_test, args):
    """
    Data preprocessing: PCA and normalization
    """   
    varian_ratio_tol = args.pca_ratio
    #Normalizing
    scaler1 = preprocessing.StandardScaler().fit(Feas_train)
#    scaler1 = preprocessing.MinMaxScaler().fit(Feas_train)
    Feas_train = scaler1.transform(Feas_train)
    Feas_test = scaler1.transform(Feas_test)
    
    #keep the first n components
    if varian_ratio_tol < 1:
        Reducter = PCA()
        Reducter.fit(Feas_train)
        Feas_train_redu = Reducter.transform(Feas_train)
        Feas_test_redu = Reducter.transform(Feas_test)
        variance_ratio = np.cumsum(Reducter.explained_variance_ratio_)
        cpn_tol_flag = variance_ratio <= varian_ratio_tol
        logging.info('*' * 10 + 'PCA: ' + str(sum(cpn_tol_flag)) + 
                     '/' + str(len(cpn_tol_flag)))
        Feas_train = Feas_train_redu[:, cpn_tol_flag]
        Feas_test = Feas_test_redu[:, cpn_tol_flag]
        
        
    #Normalizing
    scaler2 = preprocessing.StandardScaler().fit(Feas_train)
#    scaler2 = preprocessing.MinMaxScaler().fit(Feas_train)
    Feas_train = scaler2.transform(Feas_train)
    Feas_test = scaler2.transform(Feas_test)
    if varian_ratio_tol!=1:
        pickle.dump({'scaler1':scaler1, 'scaler2':scaler2, 'Reducter':Reducter, 'cpn_tol_flag':cpn_tol_flag},
                    open(args.preprocessor, 'wb'))
    else:
        pickle.dump({'scaler1':scaler1, 'scaler2':scaler2},
                    open(args.preprocessor, 'wb'))
    return Feas_train, Feas_test
        
def PrepocessingApply_func(Feas_test, args):
    """
    Data preprocessing applying: PCA and normalization
    """   
    preprossors = pickle.load(open(args.preprocessor, 'rb'))
    Feas_test = preprossors['scaler1'].transform(Feas_test)
    if args.pca_ratio != 1:
        Feas_test_redu = preprossors['Reducter'].transform(Feas_test)
        Feas_test = Feas_test_redu[:, preprossors['cpn_tol_flag']]
    Feas_test = preprossors['scaler2'].transform(Feas_test)
    return Feas_test

def Norm_0_1_func(data):
    """normalizing each column to 0-1
    Args:
        data: a matrix or vector
    """
    if data.ndim==1:
        data = (data-min(data))/(max(data)-min(data))
    elif data.ndim==2:  
        data_min = np.nanmin(data, axis=0)[np.newaxis, :]
        data_max = np.nanmax(data, axis=0)[np.newaxis, :]
        data = (data-data_min)/(data_max-data_min)
    return data

def FeaExtract(feaList):
    fea_flat = []
    for feaList_tmp in feaList:
        fea_flat.extend(feaList_tmp.tolist())
    fea_flat = np.array(fea_flat)
    return fea_flat

def FeaMerge(fea_flat, fea_ref):
    fea_merge = []
    m = 0
    for i, fea_ref_tmp in enumerate(fea_ref):
        fea_flat_tmp = fea_flat[m:(m + fea_ref_tmp.shape[0])]
        fea_flat_tmp = Norm_0_1_func(fea_flat_tmp)
#        scaler2 = preprocessing.StandardScaler().fit(fea_flat_tmp)
#        scaler2 = preprocessing.MinMaxScaler().fit(fea_flat_tmp)
#        fea_flat_tmp = scaler2.transform(fea_flat_tmp)
        fea_merge.append(fea_flat_tmp)
        m = m+len(fea_ref_tmp)
    return fea_merge

def gen_embeddings_func(word_dict, dim, in_file=None,
                   init=lasagne.init.Uniform()):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """

    num_words = max(word_dict.values()) + 1
    #initialize embeddings (50000+2) x dim
    embeddings = init((num_words, dim))
    
    if in_file is not None:
        pre_trained = 0
        initialized = {}
        avg_sigma = 0
        avg_mu = 0
        for line in open(in_file).readlines():
            sp = line.split()
            assert len(sp) == dim + 1
            if sp[0] in word_dict:
                initialized[sp[0]] = True
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]
                mu = embeddings[word_dict[sp[0]]].mean()
                #print embeddings[word_dict[sp[0]]]
                sigma = np.std(embeddings[word_dict[sp[0]]])
                avg_mu += mu
                avg_sigma += sigma
        avg_sigma /= 1. * pre_trained
        avg_mu /= 1. * pre_trained
        for w in word_dict:
            if w not in initialized:
                embeddings[word_dict[w]] = np.random.normal(avg_mu, avg_sigma, (dim,))
        
        embeddings_dict={}
        embeddings_dict['UNK']=embeddings[0,:]
        embeddings_dict['|||']=embeddings[1,:]
        
        for w in word_dict:
            embeddings_dict[w]=embeddings[word_dict[w],:]
    return embeddings_dict

def main(args):
    logging.info('-' * 50)
    logging.info('Build dictionary..')
    word_dict = pickle.load(open("../../obj/dict.pkl", "rb"))
    logging.info('-' * 50)
    embeddings = utils.gen_embeddings(word_dict, args.embedding_size, args.embedding_file)
    (args.vocab_size, args.embedding_size) = embeddings.shape
    
    logging.info('-' * 50)
    logging.info('Load data files..')
    best_dev_acc_vals = []
    best_train_acc_vals = []
    best_all_acc_vals = []
    best_n_updates_vals = []
    for val_id in range(args.cross_val):
        logging.info('Compile functions..')
        train_fn, test_fn, params, all_params = build_fn(args, embeddings)
        logging.info('Done.')
        logging.info('-' * 50)
        logging.info(args)
        if not(args.test_only):
            logging.info('*' * 10 + ' All')
            all_examples = utils.load_data(args.all_file, args, relabeling=args.relabeling)
            sample_index = np.arange(len(all_examples[0]))
#            dev_ratio = args.dev_ratio
#            random.seed(args.random_seed)
#            dev_index= random.sample(sample_index, int(dev_ratio*len(sample_index)))
            val_sample_num = len(sample_index) * (1. / args.cross_val)
            if (val_id + 1) == args.cross_val:
                dev_index = sample_index[int(val_id*val_sample_num):]
            else:
                dev_index = sample_index[int(val_id*val_sample_num):int((val_id + 1)*val_sample_num)]
            train_index  = np.setdiff1d(sample_index, dev_index)
            dev_examples = tuple_part(all_examples, dev_index)
            train_examples = tuple_part(all_examples, train_index)
            #feature preprocessing
            train_fea_flat_np = FeaExtract(train_examples[-1])
            dev_fea_flat_np = FeaExtract(dev_examples[-1])
            train_fea_flat_np2, dev_fea_flat_np2 = Prepocessing_func(
                    train_fea_flat_np, dev_fea_flat_np, args)
#            train_fea_flat_np2 = train_fea_flat_np
#            dev_fea_flat_np2 = dev_fea_flat_np
            train_fea_merge = FeaMerge(train_fea_flat_np2, train_examples[-1])
            dev_fea_merge = FeaMerge(dev_fea_flat_np2, dev_examples[-1])
            train_examples = train_examples[:-1] + (train_fea_merge, )
            dev_examples = dev_examples[:-1] + (dev_fea_merge, )
            args.num_train = len(train_examples[0])
        else:
            logging.info('*' * 10 + ' Dev')
            dev_examples = utils.load_data(args.dev_file, args, args.max_dev, relabeling=args.relabeling)
            dev_fea_flat_np = FeaExtract(dev_examples[-1])
            dev_fea_flat_np2 = PrepocessingApply_func(dev_fea_flat_np, args)
            dev_fea_flat_np2 = dev_fea_flat_np
            dev_fea_merge = FeaMerge(dev_fea_flat_np2, dev_examples[-1])
            dev_examples = dev_examples[:-1] + (dev_fea_merge, )
        args.num_dev = len(dev_examples[0])

        logging.info('-' * 50)
        logging.info('Intial test..')
        dev_x1, dev_x2, dev_x3, dev_y, dev_x4 = utils.vectorize(dev_examples, word_dict, sort_by_len=not args.test_only, concat=args.concat)
        word_dict_r = {}
        word_dict_r[0] = "unk"
        assert len(dev_x1) == args.num_dev
        all_dev = gen_examples(dev_x1, dev_x2, dev_x3, dev_y, dev_x4, args.batch_size, args.concat)
        dev_acc, _, prediction, test_probs, weights, _ = eval_acc(test_fn, all_dev)
        logging.info('Dev accuracy: %.2f %%' % dev_acc.mean())
        print(dev_acc.mean())

        best_dev_acc = dev_acc
        best_train_acc = 0
        best_all_acc = 0
        if args.test_only:
            best_dev_acc_vals.append(best_dev_acc)
            best_train_acc_vals.append(best_train_acc)
            best_all_acc_vals.append(best_all_acc) 
            best_n_updates_vals.append(0)
            return best_dev_acc_vals, best_train_acc_vals, best_all_acc_vals, best_n_updates_vals
        utils.save_params(args.model_file, all_params, epoch=0, n_updates=0)
        # Training
        logging.info('-' * 50)
        logging.info('Start training..')
        train_x1, train_x2, train_x3, train_y, train_x4 = utils.vectorize(train_examples, word_dict, concat=args.concat)
        assert len(train_x1) == args.num_train
        start_time = time.time()
        n_updates = 0

        all_train = gen_examples(train_x1, train_x2, train_x3, train_y, train_x4, args.batch_size, args.concat)

        ini_train_acc, ini_train_label, _, ini_train_probs, train_weight, train_loss = eval_acc(test_fn, all_train)
        logging.info('initial train accuracy: acc = %.2f %%' % (ini_train_acc))
        pickle.dump({'train_acc':ini_train_acc, 
                     'train_label':ini_train_label, 
                     'train_probs':ini_train_probs}, open('ini.pickle', 'wb'))

        ini_dev_acc, _, _, ini_dev_probs, _, _ = eval_acc(test_fn, all_dev)
        logging.info('initial dev accuracy: acc = %.2f %%' % (ini_dev_acc))
        ini_all_acc, _, _, ini_all_probs, _, _ = eval_acc(test_fn, all_train + all_dev)
        logging.info('initial all accuracy: acc = %.2f %%' % (ini_all_acc))
        best_dev_acc = 0
        best_n_updates = n_updates + 0
        fail_update_num = 0
        break_epoch = False
        loss_curve = [train_loss]
        train_acc_curve = [ini_train_acc]
        dev_acc_curve = [ini_dev_acc]
        weight_curve = []
        weight_curve.append(train_weight)
        para_curve = []
        para_curve.append([x.get_value() for x in params])
        logging.info([x.get_value() for x in params])
        for epoch in range(args.num_epoches):
            if break_epoch:
                fig_train.savefig(args.Type + str(val_id) + '.png')
                break
#            np.random.shuffle(all_train)
            for idx, (mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3,
                      mb_mask3, mb_y, mb_x4, mb_mask4) in enumerate(all_train):
                #early stopping
                if fail_update_num > args.update_fail_tol:
                    break_epoch = True
                    break
                
                train_loss = train_fn(mb_x1, mb_mask1, mb_x3, mb_mask3, mb_y, mb_x4)
                #if idx % 100 == 0:
#                if n_updates % int(args.eval_iter/4) == 0:
#                if n_updates % args.eval_iter == 0:
                if epoch % args.eval_iter == 0:
                    logging.info('#Examples = %d, max_len = %d' % (len(mb_x1), mb_x1.shape[1]))
                    logging.info('Epoch = %d, iter = %d (max = %d), loss = %.2f, elapsed time = %.2f (s)' % (epoch, idx, len(all_train), train_loss, time.time() - start_time))
                n_updates += 1

#                if n_updates % args.eval_iter == 0:
                if epoch % args.eval_iter == 0:
                    logging.info([x.get_value() for x in params])
                    #print([x.get_value() for x in all_params])
                    samples = sorted(np.random.choice(args.num_train, min(args.num_train, args.num_dev),
                                                      replace=False))
                    sample_train = gen_examples([train_x1[k] for k in samples],
                                                [train_x2[k] for k in samples],
                                                [train_x3[k * 4 + o] for k in samples for o in range(4)],
                                                [train_y[k] for k in samples],
                                                [train_x4[k] for k in samples],
                                                args.batch_size, args.concat)
                    acc, _, pred, _, _, _ = eval_acc(test_fn, sample_train)
                    #logging.info('Train accuracy: %.2f %%' % acc)
                    train_acc, train_label, pred, train_probs, train_weight, train_loss = eval_acc(test_fn, all_train)
                    logging.info('train accuracy: %.2f %%' % train_acc)
                    dev_acc, _, pred, _, weights, _ = eval_acc(test_fn, all_dev)
                    logging.info('Dev accuracy: %.2f %%' % dev_acc)
                    all_acc, _, _, _, _, _ = eval_acc(test_fn, all_train + all_dev)
                    logging.info('All accuracy: %.2f %%' % all_acc)
                    if args.show_loss:
                        loss_curve.append(train_loss)
                        train_acc_curve.append(train_acc)
                        dev_acc_curve.append(dev_acc)
                        if (epoch % (20*args.eval_iter) == 0) & (idx==0):
                            fig_train = plt.figure(num='training')
                            plt.clf()
                            plt.subplot(121)
                            plt.plot(-np.array(loss_curve), '.--')
                            plt.title('train_loss')
                            plt.subplot(122)
                            plt.title('accuracy')
                            plt.plot(train_acc_curve, '.--', label='train_acc')
                            plt.plot(dev_acc_curve, '.--', label='dev_acc')
                            plt.legend()
                            plt.pause(1)
                        
                        weight_curve.append(train_weight)
                        para_curve.append([x.get_value() for x in params])
                        if (epoch % (20*args.eval_iter) == 0) & (idx==0):
                            fig_train_para = plt.figure(num='para')
                            plt.clf()
                            plt.subplot(121)
                            mlp_w_curve = np.array([para_tmp[0].reshape((-1)) for para_tmp in para_curve])
                            mlp_w_curve0 = np.concatenate((mlp_w_curve[:, :-5], mlp_w_curve[:, -4:]), axis=1)
                            mlp_w_curve1 = mlp_w_curve[:, -5]
                            plt.plot(mlp_w_curve0, '.--', color='grey')
                            plt.plot(mlp_w_curve1, '.--', color='k')
                            mlp_b_curve = np.array([para_tmp[1].reshape((-1)) for para_tmp in para_curve])
                            plt.plot(mlp_b_curve, '.--', color='r')
                            plt.title('mlp_w')
                            
                            plt.subplot(122)
                            word_weight = np.array([np.array(weight_tmp[0]).reshape((-1)) for weight_tmp in weight_curve])
                            plt.plot(word_weight, '.--')
                            plt.title('word_weight')
                            plt.pause(1)
                        
                    #print(weights[0])
                    if dev_acc > best_dev_acc:
                        fail_update_num = 0
                        best_n_updates = n_updates + 0
                        best_train_acc = train_acc
                        logging.info('Best train accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                                     % (epoch, n_updates, best_train_acc))
                        best_dev_acc = dev_acc
                        logging.info('Best dev accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                                     % (epoch, n_updates, best_dev_acc))
                        best_all_acc = all_acc
                        logging.info('Best all accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                                     % (epoch, n_updates, best_all_acc))
                        utils.save_params(args.model_file, all_params, epoch=epoch, n_updates=n_updates, )
                    else:
                        #updates that dev accuracy does not increase
                        fail_update_num = fail_update_num + 1
                    logging.info('failed updates: ' + str(fail_update_num))
                    logging.info('*'*50)
        
        best_dev_acc_vals.append(best_dev_acc)
        best_train_acc_vals.append(best_train_acc)
        best_all_acc_vals.append(best_all_acc)   
        best_n_updates_vals.append(best_n_updates)
        fig_train.savefig(args.Type + str(val_id) + '.png')
        fig_train_para.savefig(args.Type + str(val_id) + '_para.png')
        
        
        pickle.dump({'loss_curve':loss_curve, 
                     'train_acc_curve':train_acc_curve, 
                     'dev_acc_curve':dev_acc_curve, 
                     'weight_curve':weight_curve, 
                    'para_curve':para_curve}, open('final.pickle', 'wb'))
    return best_dev_acc_vals, best_train_acc_vals, best_all_acc_vals, best_n_updates_vals

if __name__ == '__main__':
    args = config.get_args()
    #set Type
#    Type = ['Mainly', 'Title', 'Fact', 'LPurpose']
    Type = ['Fact']
    WeightMea = ['IA_RUN_COUNT', 'Heat_resize_Norm_demean', 'WordFreqInDoc',
             'IA_SKIP', 'Heat_demean', 'Word_Fre', 'Heat', 'Word_len', 
             'Heat_resize_demean', 'Heat_Norm_demean',
             'IA_DWELL_TIME_%', 'IA_FIXATION_COUNT', 'Heat_Norm', 
             'RationaleTime5', 'IA_DWELL_TIME', 'RationaleTimeCtg1', 
             'RationaleTimeCtg1_pred', 'RationaleTimeCtg1_prob']
#    WeightMea = ['WordFreqInDoc', 'Word_Fre', 'Word_len', 'RationaleTime5', 'IA_DWELL_TIME']
#    WeightMea = ['WordFreqInDoc', 'Word_Fre', 'Word_len', 'IA_DWELL_TIME', 'RationaleTime5']
#    WeightMea = ['RationaleTime5']
    
    np.random.seed(args.random_seed)
    lasagne.random.set_rng(np.random.RandomState(args.random_seed))
    
    if args.train_file is None:
        raise ValueError('train_file is not specified.')

    if args.dev_file is None:
        raise ValueError('dev_file is not specified.')

    if args.rnn_type == 'lstm':
        args.rnn_layer = lasagne.layers.LSTMLayer
    elif args.rnn_type == 'gru':
        args.rnn_layer = lasagne.layers.GRULayer
    else:
        raise NotImplementedError('rnn_type = %s' % args.rnn_type)

    if args.embedding_file is not None:
        dim = utils.get_dim(args.embedding_file)
        if (args.embedding_size is not None) and (args.embedding_size != dim):
            raise ValueError('embedding_size = %d, but %s has %d dims.' %
                             (args.embedding_size, args.embedding_file, dim))
        args.embedding_size = dim
    elif args.embedding_size is None:
        raise RuntimeError('Either embedding_file or embedding_size needs to be specified.')

    if args.log_file is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=args.log_file,
                            filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

    logging.info(' '.join(sys.argv))
    #train
    multi_fea_acc = pd.DataFrame(columns=['Type', 'WeightMea', 'best_dev_acc_val', 'best_train_acc_val',
                                           'best_all_acc_val', 'best_n_updates_val', 'best_dev_acc',
                                           'best_train_acc', 'best_all_acc', 'actiMlP'])
    single_fea_acc = pd.DataFrame(columns=['Type', 'WeightMea', 'freezeMlP', 'best_dev_acc_val', 
                                           'best_train_acc_val', 'best_all_acc_val', 'best_n_updates_val',
                                           'best_dev_acc', 'best_train_acc', 'best_all_acc', 'actiMlP'])
    time_S = time.time()
    for Type_i in Type:
        args.Type = Type_i
        args.all_file = '../JsonData/tokenizedRationales'
        args.dev_file = '../JsonData/tokenizedRationales'
        args.all_file = args.all_file + '/' + args.Type
        args.dev_file = args.dev_file + '/' + args.Type
        #set params
        args.pca_ratio = 1
        args.random_seed = 520
#        args.eval_iter = 100
        args.eval_iter = 1
        args.cross_val = 5
#        args.update_fail_tol = 15
        args.update_fail_tol = 1500000
    #    args.pre_trained = args.model_file + args.Type + '.pkl.gz'
        args.pre_trained = None
        args.model_file = args.model_file + args.Type + '.pkl.gz'
        args.test_only = False
        args.preprocessor = args.preprocessor + args.Type + str(args.pca_ratio) + '.pickle'
        args.tune_embedding = False
        #activation of MLP
        args.actiMlP = sigmoid  #sigmoid, LeakyRectify(0.1), rectify
        #multi features
        logging.info('*'*10 + Type_i + ': multi features' + '*'*10)
        logging.info('*'*10 + Type_i + ', time passed: ' + str(time.time() - time_S))
        args.WeightMea = WeightMea
        args.mea_num = len(args.WeightMea)
        args.num_epoches = 10000
        args.freezeMlP = False
        args.show_loss = True
        best_dev_acc_vals, best_train_acc_vals, best_all_acc_vals, best_n_updates_vals = main(args)
        logging.info('Best train accuracy: acc = %s %%' % (best_dev_acc_vals))
        logging.info('Best dev accuracy: acc = %s %%' % (best_train_acc_vals))
        logging.info('Best all accuracy: acc = %s %%' % (best_all_acc_vals))
        multi_fea_acc_tmp = [args.Type, args.WeightMea, best_dev_acc_vals,
                             best_train_acc_vals, best_all_acc_vals, best_n_updates_vals,
                             np.mean(best_dev_acc_vals), np.mean(best_train_acc_vals),
                             np.mean(best_all_acc_vals), args.actiMlP]
        logging.info('-'*50)
        logging.info(multi_fea_acc_tmp)
        logging.info('-'*50)
        multi_fea_acc.loc[len(multi_fea_acc.index), :] = multi_fea_acc_tmp
        
#        #single features
#        for mea_tmp in WeightMea:
#            args.WeightMea = [mea_tmp]
#            args.mea_num = len(args.WeightMea)
#            logging.info('*'*10 + Type_i + ': single features: ' + mea_tmp + '*'*10)
#            for freezeMlP in [True, False]:
##            for freezeMlP in [False]:
#                #freeze MLP
#                args.freezeMlP = freezeMlP
#                if args.freezeMlP:
#                    args.num_epoches = 200
#                    args.update_fail_tol = 5
#                else:
#                    args.num_epoches = 3000
#                    args.update_fail_tol = 15
#                best_dev_acc_vals, best_train_acc_vals, best_all_acc_vals, best_n_updates_vals = main(args)
#                logging.info('Best train accuracy: acc = %s %%' % (best_dev_acc_vals))
#                logging.info('Best dev accuracy: acc = %s %%' % (best_train_acc_vals))
#                logging.info('Best all accuracy: acc = %s %%' % (best_all_acc_vals))
#                single_fea_acc_tmp = [args.Type, args.WeightMea, args.freezeMlP, 
#                        best_dev_acc_vals, best_train_acc_vals, best_all_acc_vals,
#                        best_n_updates_vals, np.mean(best_dev_acc_vals), np.mean(best_train_acc_vals), 
#                        np.mean(best_all_acc_vals), args.actiMlP]
#                logging.info('-'*50)
#                logging.info(single_fea_acc_tmp)
#                logging.info('-'*50)
#                single_fea_acc.loc[len(single_fea_acc.index), :] = single_fea_acc_tmp
#    pickle.dump({'multi_fea_acc':multi_fea_acc, 'single_fea_acc':single_fea_acc}, 
#                open('fea_acc.pickle', 'wb'))
    
    
#    test_mea = 'best_dev_acc'
#    plt.figure()
#    for i, Type_i in enumerate(Type):
#        single_fea_acc_tmp = single_fea_acc.loc[single_fea_acc.Type==Type_i, :]
#        single_fea_acc_true = single_fea_acc_tmp.loc[single_fea_acc_tmp.freezeMlP==True, :]
#        single_fea_acc_true = single_fea_acc_true.sort_values(by=test_mea, ascending=True)
#        single_fea_acc_false = single_fea_acc_tmp.loc[single_fea_acc_true.index + 1, :]
#        plt.subplot(2, 2, i+1)
#        y1 = np.arange(len(single_fea_acc_true.index)) + 0.15
#        y2 = np.arange(len(single_fea_acc_true.index)) - 0.15
#        plt.barh(y1, single_fea_acc_true[test_mea].values, 0.3, label='raw')
#        plt.barh(y2, single_fea_acc_false[test_mea].values, 0.3, label='sigmoid')
#        plt.barh(len(single_fea_acc_true.index), 
#                 multi_fea_acc.loc[multi_fea_acc.Type==Type_i, test_mea], 0.3, label='mlp')
#        plt.yticks(np.arange(len(single_fea_acc_true.index)), single_fea_acc_true.WeightMea.values)
#        plt.xlim(0, 70)
#        plt.legend(loc='lower right')
#        plt.title(Type_i)
            
        
        
        
        
        
        
        
        
    
    