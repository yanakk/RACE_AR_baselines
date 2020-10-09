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

def gen_examples(x1, x2, x3, y, batch_size, concat=False):
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
        mb_x1, mb_mask1 = utils.prepare_data(mb_x1)
        mb_x2, mb_mask2 = utils.prepare_data(mb_x2)
        mb_x3, mb_mask3 = utils.prepare_data(mb_x3)
        all_ex.append((mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3, mb_y))
    return all_ex


def build_fn(args, embeddings):
    """
        Build training and testing functions.
    """
    in_x1 = T.imatrix('x1')
    in_x2 = T.imatrix('x2')
    in_x3 = T.imatrix('x3')
    in_mask1 = T.matrix('mask1')
    in_mask2 = T.matrix('mask2')
    in_mask3 = T.matrix('mask3')
    in_y = T.ivector('y')

    l_in1 = lasagne.layers.InputLayer((None, None), in_x1)
    l_mask1 = lasagne.layers.InputLayer((None, None), in_mask1)
    l_emb1 = lasagne.layers.EmbeddingLayer(l_in1, args.vocab_size,
                                           args.embedding_size, W=embeddings)

    l_in2 = lasagne.layers.InputLayer((None, None), in_x2)
    l_mask2 = lasagne.layers.InputLayer((None, None), in_mask2)
    l_emb2 = lasagne.layers.EmbeddingLayer(l_in2, args.vocab_size,
                                           args.embedding_size, W=l_emb1.W)

    l_in3 = lasagne.layers.InputLayer((None, None), in_x3)
    l_mask3 = lasagne.layers.InputLayer((None, None), in_mask3)
    l_emb3 = lasagne.layers.EmbeddingLayer(l_in3, args.vocab_size,
                                           args.embedding_size, W=l_emb1.W)

    if not args.tune_embedding:
        l_emb1.params[l_emb1.W].remove('trainable')
        l_emb2.params[l_emb2.W].remove('trainable')
        l_emb3.params[l_emb3.W].remove('trainable')

    args.rnn_output_size = args.hidden_size * 2 if args.bidir else args.hidden_size
    if args.model == "GA":
        l_d = l_emb1
        # NOTE: This implementation slightly differs from the original GA reader. Specifically:
        # 1. The query GRU is shared across hops.
        # 2. Dropout is applied to all hops (including the initial hop).
        # 3. Gated-attention is applied at the final layer as well.
        # 4. No character-level embeddings are used.

        l_q = nn_layers.stack_rnn(l_emb2, l_mask2, 1, args.hidden_size,
                                       grad_clipping=args.grad_clipping,
                                       dropout_rate=args.dropout_rate,
                                       only_return_final=False,
                                       bidir=args.bidir,
                                       name='q',
                                       rnn_layer=args.rnn_layer)
        q_length = nn_layers.LengthLayer(l_mask2)
        network2 = QuerySliceLayer([l_q, q_length])
        for layer_num in xrange(args.num_GA_layers):
            l_d = nn_layers.stack_rnn(l_d, l_mask1, 1, args.hidden_size,
                                      grad_clipping=args.grad_clipping,
                                      dropout_rate=args.dropout_rate,
                                      only_return_final=False,
                                      bidir=args.bidir,
                                      name='d' + str(layer_num),
                                      rnn_layer=args.rnn_layer)
            l_d = GatedAttentionLayerWithQueryAttention([l_d, l_q, l_mask2])
        network1 = l_d
    else:
        assert args.model is None
        network1 = nn_layers.stack_rnn(l_emb1, l_mask1, args.num_layers, args.hidden_size,
                                       grad_clipping=args.grad_clipping,
                                       dropout_rate=args.dropout_rate,
                                       only_return_final=(args.att_func == 'last'),
                                       bidir=args.bidir,
                                       name='d',
                                       rnn_layer=args.rnn_layer)

        network2 = nn_layers.stack_rnn(l_emb2, l_mask2, args.num_layers, args.hidden_size,
                                       grad_clipping=args.grad_clipping,
                                       dropout_rate=args.dropout_rate,
                                       only_return_final=True,
                                       bidir=args.bidir,
                                       name='q',
                                       rnn_layer=args.rnn_layer)
    if args.att_func == 'mlp':
        att = nn_layers.MLPAttentionLayer([network1, network2], args.rnn_output_size,
                                          mask_input=l_mask1)
    elif args.att_func == 'bilinear':
        att = nn_layers.BilinearAttentionLayer([network1, network2], args.rnn_output_size,
                                               mask_input=l_mask1)
        att_weightLayer= nn_layers.BilinearAttentionWeightLayer([network1, network2], args.rnn_output_size,
                                               mask_input=l_mask1)
    elif args.att_func == 'avg':
        att = nn_layers.AveragePoolingLayer(network1, mask_input=l_mask1)
    elif args.att_func == 'last':
        att = network1
    elif args.att_func == 'dot':
        att = nn_layers.DotProductAttentionLayer([network1, network2], mask_input=l_mask1)
    else:
        raise NotImplementedError('att_func = %s' % args.att_func)
    network3 = nn_layers.stack_rnn(l_emb3, l_mask3, args.num_layers, args.hidden_size,
                                   grad_clipping=args.grad_clipping,
                                   dropout_rate=args.dropout_rate,
                                   only_return_final=True,
                                   bidir=args.bidir,
                                   name='o',
                                   rnn_layer=args.rnn_layer)
    network3 = lasagne.layers.ReshapeLayer(network3, (in_x1.shape[0], 4, args.rnn_output_size))
    network = nn_layers.BilinearDotLayer([network3, att], args.rnn_output_size)
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
    test_prob = lasagne.layers.get_output(network, deterministic=True)
    test_prediction = T.argmax(test_prob, axis=-1)
    acc = T.sum(T.eq(test_prediction, in_y))
    test_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_x3, in_mask3, in_y], [acc, test_prediction], on_unused_input='warn')

    # Train functions
    train_prediction = lasagne.layers.get_output(network)
    train_prediction = T.clip(train_prediction, 1e-7, 1.0 - 1e-7)
    loss = lasagne.objectives.categorical_crossentropy(train_prediction, in_y).mean()
        
    # Attention functions
    att_weight=lasagne.layers.get_output(att_weightLayer, deterministic=True)
    attention_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2], att_weight, on_unused_input='warn')
    
    # TODO: lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    params = lasagne.layers.get_all_params(network)#, trainable=True)
    all_params = lasagne.layers.get_all_params(network)
    if args.optimizer == 'sgd':
        updates = lasagne.updates.sgd(loss, params, args.learning_rate)
    elif args.optimizer == 'adam':
        updates = lasagne.updates.adam(loss, params, learning_rate=args.learning_rate)
    elif args.optimizer == 'rmsprop':
        updates = lasagne.updates.rmsprop(loss, params, learning_rate=args.learning_rate)
    else:
        raise NotImplementedError('optimizer = %s' % args.optimizer)
    train_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_x3, in_mask3, in_y],
                               loss, updates=updates, on_unused_input='warn')

    return train_fn, test_fn, attention_fn, params, all_params


def eval_acc(test_fn, all_examples):
    """
        Evaluate accuracy on `all_examples`.
    """
    acc = 0
#    acc = []
    n_examples = 0
#    n_examples = []
    prediction = []
    for x1, mask1, x2, mask2, x3, mask3, y in all_examples:
        tot_acc, pred = test_fn(x1, mask1, x2, mask2, x3, mask3, y)
        acc += tot_acc
#        acc.append(float(tot_acc))
        prediction += pred.tolist()
        n_examples += len(x1)
#        n_examples.append(len(x1))
    return np.array(acc)/np.array(n_examples), np.array(n_examples), prediction, all_examples

def attention_func(attention_fn, all_examples):
    """
        Get attention weights.
    """
    alpha = []
    for x1, mask1, x2, mask2, x3, mask3, y in all_examples:
        alpha_ex = attention_fn(x1, mask1, x2, mask2)
        alpha.append(alpha_ex)
    return alpha

def main(args):
    logging.info('-' * 50)
    logging.info('Load data files..')
    question_belong = []
    if args.debug:
        logging.info('*' * 10 + ' Train')
        train_examples = utils.load_data(args.train_file, 100, relabeling=args.relabeling)
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, 100, relabeling=args.relabeling, question_belong=question_belong)
    else:
#        logging.info('*' * 10 + ' Train')
#        train_examples = utils.load_data(args.train_file, relabeling=args.relabeling)
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, args.max_dev, relabeling=args.relabeling, question_belong=question_belong)

#    args.num_train = len(train_examples[0])
    args.num_dev = len(dev_examples[0])

    logging.info('-' * 50)
    logging.info('Build dictionary..')
#    word_dict = utils.build_dict(train_examples[0] + train_examples[1] + train_examples[2], args.max_vocab_size)
    word_dict = pickle.load(open("../../obj/dict.pkl", "rb"))
    logging.info('-' * 50)
    embeddings = utils.gen_embeddings(word_dict, args.embedding_size, args.embedding_file)
    (args.vocab_size, args.embedding_size) = embeddings.shape
    logging.info('Compile functions..')
    train_fn, test_fn, attention_fn, params, all_params = build_fn(args, embeddings)
    logging.info('Done.')
    logging.info('-' * 50)
    logging.info(args)

    logging.info('-' * 50)
    logging.info('Intial test..')
    dev_x1, dev_x2, dev_x3, dev_y = utils.vectorize(dev_examples, word_dict, sort_by_len=not args.test_only, concat=args.concat)
    word_dict_r = {}
    word_dict_r[0] = "unk"
    assert len(dev_x1) == args.num_dev
    all_dev = gen_examples(dev_x1, dev_x2, dev_x3, dev_y, args.batch_size, args.concat)
    dev_acc, n_examples, prediction, all_examples= eval_acc(test_fn, all_dev)
    
    logging.info('Dev accuracy: %.2f %%' % dev_acc.mean())
    print(dev_acc.mean())
    
    alpha= attention_func(attention_fn, all_dev)
    
    if args.test_only:
        return dev_acc,n_examples, prediction, all_examples, alpha


if __name__ == '__main__':
    
    args = config.get_args()
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
    dev_acc, n_examples, pred, all_examples, alpha=main(args)
    
    Label=[examples_ex[-1] for examples_ex in all_examples]
    Label=np.array(Label)
    Label=np.reshape(Label, [-1], order='C')
    TF = pred==Label
    print('Fact: ' + str(np.mean(TF[:200])))
    print('Mainly: ' + str(np.mean(TF[200:])))
    
    F=open('./AccSAR_P','wb')
    pickle.dump([dev_acc, n_examples, pred, Label, alpha, all_examples], F)
    F.close()
    
    #%% the distribution of attention    
#    F=open('./AccSAR_FC','rb')
#    dev_acc, n_examples, pred,Label, alpha, all_examples=pickle.load(F)
#    F.close()
#    fig1=plt.figure()
#    plt.hist(dev_acc,bins=20)
##    plt.plot([MRC_Acc, MRC_Acc],[0,100],color='red')   
#    plt.plot([dev_acc.mean(), dev_acc.mean()],[0,100],color='black')       
#    plt.xticks([dev_acc.mean(),0.15,0.25,0.5,0.65])
#    plt.show
    
    #%% the attention map
#    word_dict = pickle.load(open("../../obj/dict.pkl", "rb"))
#    word_dict['Unk']=0
#    
#    word_dict_Value=word_dict.values()
#    word_dict_Key=word_dict.keys()
#    
#    
#    ducoX=all_examples[0][0]
#    ducoM=all_examples[0][1]    
#    alpha_ny=alpha[0]
#    word_lists=[]
#    alphas=[]
#    for i,alpha_tmp in enumerate(alpha_ny):
#        ducoX_tmp=ducoX[i,:]
#        ducoM_tmp=ducoM[i,:]
#        if 0 in alpha_tmp:
#            NonZeroIndex=np.where(alpha_tmp==0)[0][0]
#        else:
#            NonZeroIndex=len(alpha_tmp)+1
#            
#        ducoX_tmp=ducoX_tmp[:NonZeroIndex]
#        ducoM_tmp=ducoM_tmp[:NonZeroIndex]
#        alpha_tmp=alpha_tmp[:NonZeroIndex]
#        
#        word_list=[]
#        for wordX_tmp in ducoX_tmp:
#            word_list.append(word_dict_Key[np.where(word_dict_Value==wordX_tmp)[0][0]])
#        word_lists.append(word_list)
#        alphas.append(alpha_tmp)
#        
#    F=open('./AttentionWeights','wb')
#    pickle.dump([alphas,word_lists],F)
#    F.close()
        
        
        
        
        
        
        
        
        
        
    
    