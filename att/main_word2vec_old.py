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
    weight_mlp_np = np.zeros((15, 1))
    weight_mlp_np[-2] = 1.
#    weight_mlp_np = np.array([[1.]])
    b_mlp = np.array([0.])
    l_weight = lasagne.layers.DenseLayer(l_in4, 1, num_leading_axes=-1, W=weight_mlp_np, b=b_mlp)
#    l_weight = lasagne.layers.DenseLayer(l_in4, 1, num_leading_axes=-1)
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
    test_prob = lasagne.layers.get_output(network, deterministic=True)
    test_prediction = T.argmax(test_prob, axis=-1)
    acc = T.sum(T.eq(test_prediction, in_y))
    test_fn = theano.function([in_x1, in_mask1, in_x3, in_mask3, in_y, in_x4], [acc, test_prediction, test_prob], on_unused_input='warn')

    # Train functions
    train_prediction = lasagne.layers.get_output(network)
    train_prediction = T.clip(train_prediction, 1e-7, 1.0 - 1e-7)
    loss = lasagne.objectives.categorical_crossentropy(train_prediction, in_y).mean()
    
    # TODO: lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
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
    for x1, mask1, x2, mask2, x3, mask3, y, x4, mask4 in all_examples:
        tot_acc, pred, test_prob = test_fn(x1, mask1, x3, mask3, y, x4)
        acc += tot_acc
        prediction += pred.tolist()
        test_probs += test_prob.tolist()
        n_examples += len(x1)
    return acc * 100.0 / n_examples, prediction, test_probs

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
    if varian_ratio_tol!=1:
        Reducter = PCA()
        Reducter.fit(Feas_train)
        Feas_train_redu = Reducter.transform(Feas_train)
        Feas_test_redu = Reducter.transform(Feas_test)
        variance_ratio = np.cumsum(Reducter.explained_variance_ratio_)
        cpn_tol_flag = variance_ratio < varian_ratio_tol
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
        fea_merge.append(fea_flat_tmp)
        m = m+len(fea_ref_tmp)
    return fea_merge
    
def tokenize(st):
    #TODO: The tokenizer's performance is suboptimal
    ans = []
    for sent in sent_tokenize(st):
        ans += word_tokenize(sent)
    return " ".join(ans).lower()   

def Vectorize_func(string, embeddings, embeddings_Keys):
    str_vec=[]    
    m = 0
    for str_tmp in string.split():
        if str_tmp in embeddings_Keys:
            str_vec.append(embeddings[str_tmp]) 
        else:
            str_vec.append(embeddings['UNK']) 
            m = m + 1
#    print(float(m)/len(string.split()))
    return np.array(str_vec)

def CosSimilarity_func(rationale_vec,option_vec,ChoiceMethods):
    if ChoiceMethods == 'mean':
        rationale_vec_mean = np.mean(rationale_vec, axis=0)
        option_vec_mean = np.mean(option_vec, axis=0)
        Cos = np.dot(rationale_vec_mean,option_vec_mean)/(np.linalg.norm(
                rationale_vec_mean)*np.linalg.norm(option_vec_mean))    
    return Cos 


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

def PerPassPred(dev_examples, embeddings, embeddings_Keys):
    """Predicting answer for each passage
    
    """
    passage = dev_examples[0]
    option = dev_examples[2]
    answer = dev_examples[3]
    WeightMea = dev_examples[4]
    PredAns = []
    for doc_ind in range(len(passage)):
        Passage_token = passage[doc_ind]
        WeightMea_doc = WeightMea[doc_ind]
        Option = [option[(doc_ind*4)+option_id] for option_id in range(4)]
        
        #preprocessing: tokenized
        print('preprocessing: tokenized')
        Option_token=[tokenize(Option_tmp.decode('unicode-escape')) for 
                      Option_tmp in Option]
        #rationales
        print('Real')
        CosScore = []
        pass_tmp_Em = Vectorize_func(Passage_token, embeddings, embeddings_Keys)
        for o_id, Option_tmp in enumerate(Option_token):
            for dim1_ind in range(WeightMea_doc.shape[-1]):
                WeightMea_tmp = WeightMea_doc[:, dim1_ind]
                Passage_weight = Norm_0_1_func(WeightMea_tmp)
                weight_tmp = Passage_weight.reshape((-1, 1))
                Passage_weighted = weight_tmp * pass_tmp_Em
                
                Option_tmp_Em = Vectorize_func(Option_tmp, embeddings, embeddings_Keys)  
                
                CosScore_tmp = CosSimilarity_func(Passage_weighted, Option_tmp_Em,'mean')
                CosScore.append(CosScore_tmp)
        CosScoreNp = np.array(CosScore)
        CosScoreNp2 = CosScoreNp.reshape((-1, 4), order='F')
        PredAns.append(CosScoreNp2.argmax(axis=1)[0])
        
        
    return np.array(PredAns), np.array(answer)

def main(args):
    logging.info('-' * 50)
    logging.info('Load data files..')
    if not(args.test_only):
        logging.info('*' * 10 + ' All')
        all_examples = utils.load_data(args.all_file, relabeling=args.relabeling)
        dev_ratio = args.dev_ratio
        sample_index = np.arange(len(all_examples[0]))
        random.seed(1000)
        dev_index= random.sample(sample_index, int(dev_ratio*len(sample_index)))
        train_index  = np.setdiff1d(sample_index, dev_index)
        dev_examples = tuple_part(all_examples, dev_index)
        train_examples = tuple_part(all_examples, train_index)
        #feature preprocessing
        train_fea_flat_np = FeaExtract(train_examples[-1])
        dev_fea_flat_np = FeaExtract(dev_examples[-1])
        train_fea_flat_np2, dev_fea_flat_np2 = Prepocessing_func(
                train_fea_flat_np, dev_fea_flat_np, args)
        train_fea_flat_np2 = train_fea_flat_np
        dev_fea_flat_np2 = dev_fea_flat_np
        train_fea_merge = FeaMerge(train_fea_flat_np2, train_examples[-1])
        dev_fea_merge = FeaMerge(dev_fea_flat_np2, dev_examples[-1])
        train_examples = train_examples[:-1] + (train_fea_merge, )
        dev_examples = dev_examples[:-1] + (dev_fea_merge, )
        args.num_train = len(train_examples[0])
    else:
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, args.max_dev, relabeling=args.relabeling)
        dev_fea_flat_np = FeaExtract(dev_examples[-1])
        dev_fea_flat_np2 = PrepocessingApply_func(dev_fea_flat_np, args)
        dev_fea_flat_np2 = dev_fea_flat_np
        dev_fea_merge = FeaMerge(dev_fea_flat_np2, dev_examples[-1])
        dev_examples = dev_examples[:-1] + (dev_fea_merge, )

    args.num_dev = len(dev_examples[0])
    args.mea_num = dev_examples[4][0].shape[-1]

    logging.info('-' * 50)
    logging.info('Build dictionary..')
    word_dict = pickle.load(open("../../obj/dict.pkl", "rb"))
    logging.info('-' * 50)
    embeddings = utils.gen_embeddings(word_dict, args.embedding_size, args.embedding_file)
    (args.vocab_size, args.embedding_size) = embeddings.shape
    
    
    logging.info('Compile functions..')
    train_fn, test_fn, params, all_params = build_fn(args, embeddings)
    logging.info('Done.')
    logging.info('-' * 50)
    logging.info(args)

    logging.info('-' * 50)
    logging.info('Intial test..')
    dev_x1, dev_x2, dev_x3, dev_y, dev_x4 = utils.vectorize(dev_examples, word_dict, sort_by_len=not args.test_only, concat=args.concat)
    word_dict_r = {}
    word_dict_r[0] = "unk"
    assert len(dev_x1) == args.num_dev
    all_dev = gen_examples(dev_x1, dev_x2, dev_x3, dev_y, dev_x4, args.batch_size, args.concat)
    dev_acc, prediction, test_probs = eval_acc(test_fn, all_dev)
    logging.info('Dev accuracy: %.2f %%' % dev_acc.mean())
    print(dev_acc.mean())
#    #no net
#    embeddings_dict = gen_embeddings_func(word_dict, args.embedding_size, args.embedding_file)
#    embeddings_Keys = embeddings_dict.keys()
#    PredAns, answer = PerPassPred(dev_examples, embeddings_dict, embeddings_Keys)
#    logging.info('*' * 10 + 'no-net1: ' + str(np.mean(PredAns==answer)) + '*' * 10)
    
    best_dev_acc = dev_acc
    best_train_acc = 0
    if args.test_only:
        return dev_acc, best_train_acc, prediction, all_params, test_probs
	utils.save_params(args.model_file, all_params, epoch=0, n_updates=0)
    # Training
    logging.info('-' * 50)
    logging.info('Start training..')
    train_x1, train_x2, train_x3, train_y, train_x4 = utils.vectorize(train_examples, word_dict, concat=args.concat)
    assert len(train_x1) == args.num_train
    start_time = time.time()
    n_updates = 0

    all_train = gen_examples(train_x1, train_x2, train_x3, train_y, train_x4, args.batch_size, args.concat)
    for epoch in range(args.num_epoches):
        np.random.shuffle(all_train)
        for idx, (mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3, mb_y, mb_x4, mb_mask4) in enumerate(all_train):

            train_loss = train_fn(mb_x1, mb_mask1, mb_x3, mb_mask3, mb_y, mb_x4)
#            if idx % 100 == 0:
            if epoch % 100 == 0:
                logging.info('#Examples = %d, max_len = %d' % (len(mb_x1), mb_x1.shape[1]))
                logging.info('Epoch = %d, iter = %d (max = %d), loss = %.2f, elapsed time = %.2f (s)' % (epoch, idx, len(all_train), train_loss, time.time() - start_time))
            n_updates += 1

            if n_updates % args.eval_iter == 0:
#                print([x.get_value() for x in params])
#                print([x.get_value() for x in all_params])
                samples = sorted(np.random.choice(args.num_train, min(args.num_train, args.num_dev),
                                                  replace=False))
                sample_train = gen_examples([train_x1[k] for k in samples],
                                            [train_x2[k] for k in samples],
                                            [train_x3[k * 4 + o] for k in samples for o in range(4)],
                                            [train_y[k] for k in samples],
                                            [train_x4[k] for k in samples],
                                            args.batch_size, args.concat)
                acc, pred, test_probs = eval_acc(test_fn, sample_train)
                logging.info('Train accuracy: %.2f %%' % acc)
                train_acc, pred, test_probs = eval_acc(test_fn, all_train)
                logging.info('train accuracy: %.2f %%' % train_acc)
                dev_acc, pred, test_probs = eval_acc(test_fn, all_dev)
                logging.info('Dev accuracy: %.2f %%' % dev_acc)
                all_acc, _, _ = eval_acc(test_fn, all_train + all_dev)
                logging.info('All accuracy: %.2f %%' % all_acc)
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    logging.info('Best dev accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                                 % (epoch, n_updates, best_dev_acc))
                    best_train_acc = acc
                    logging.info('Best train accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                                 % (epoch, n_updates, best_train_acc))
                    utils.save_params(args.model_file, all_params, epoch=epoch, n_updates=n_updates, )
                    
    return best_dev_acc, best_train_acc, pred, all_params, test_probs

if __name__ == '__main__':
    args = config.get_args()
    #set Type
    args.Type = 'Mainly'
    args.pca_ratio = 1
    args.all_file = args.all_file + '/' + args.Type
    args.dev_file = args.dev_file + '/' + args.Type
#    args.pre_trained = args.model_file + args.Type + '.pkl.gz'
    args.pre_trained = None
    args.model_file = args.model_file + args.Type + '.pkl.gz'
    args.test_only = False
    args.preprocessor = args.preprocessor + args.Type + str(args.pca_ratio) + '.pickle'
    args.tune_embedding = False
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
    best_dev_acc, best_train_acc, dev_pred, params, test_probs = main(args)
    logging.info('Best dev accuracy: acc = %.2f %%' % (best_dev_acc))
    logging.info('Best train accuracy: acc = %.2f %%' % (best_train_acc))
    
#    Label=[examples_ex[-1] for examples_ex in all_examples]
#    Label=np.array(Label)
#    Label=np.reshape(Label, [-1], order='C')
#    TF = pred==Label
#    print('Fact: ' + str(np.mean(TF[:200])))
#    print('Mainly: ' + str(np.mean(TF[200:])))
#    
#    F=open('./AccSAR_P','wb')
#    pickle.dump([dev_acc, n_examples, pred, Label, alpha, all_examples], F)
#    F.close()
    
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
        
        
        
        
        
        
        
        
        
        
    
    