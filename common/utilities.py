import sys
import string
import numpy as np
import per_vb


def read_data(filename):
    """ Read all documents in the file and stores terms and counts in lists. """
    wordids = list()
    wordcts = list()
    fp = open(filename, 'r')
    while True:
        line = fp.readline()
        # check end of file
        if len(line) < 1:
            break
        terms = line.split(' ')
        doc_length = int(terms[0])
        # are you sure "word_counts --> int32"?  int16 might take less memory and do okay as well.
        ids = np.zeros(doc_length, dtype=np.int32)
        cts = np.zeros(doc_length, dtype=np.int32)
        for j in range(1, doc_length + 1):
            term_count = terms[j].split(':')
            ids[j - 1] = int(term_count[0])
            cts[j - 1] = int(term_count[1])
        wordids.append(ids)
        wordcts.append(cts)
    fp.close()
    return wordids, wordcts


def read_data_for_perpl(test_data_folder):
    """ Read data for computing perplexities. """
    filename_part1 = '%s/data_test_1_part_1.txt'%(test_data_folder)
    filename_part2 = '%s/data_test_1_part_2.txt'%(test_data_folder)
    (wordids_1, wordcts_1) = read_data(filename_part1)
    (wordids_2, wordcts_2) = read_data(filename_part2)
    return wordids_1, wordcts_1, wordids_2, wordcts_2


def read_minibatch_list_frequencies(fp, batch_size):
    """ Read mini-batch and stores terms and counts in lists. """
    wordids = list()
    wordcts = list()
    for i in range(batch_size):
        line = fp.readline()
        # check end of file
        if len(line) < 5:
            break
        terms = str.split(line)
        doc_length = int(terms[0])
        ids = np.zeros(doc_length, dtype = np.int32)
        cts = np.zeros(doc_length, dtype = np.int32)
        for j in range(1,doc_length + 1):
            term_count = terms[j].split(':')
            ids[j - 1] = int(term_count[0])
            cts[j - 1] = int(term_count[1])
        wordids.append(ids)
        wordcts.append(cts)
    return(wordids, wordcts)


def read_setting(file_name):
    """Read setting file.
    """
    f = open(file_name, 'r')
    settings = f.readlines()
    f.close()
    sets = list()
    vals = list()
    for i in range(len(settings)):
        #print'%s\n'%(settings[i])
        if settings[i][0] == '#':
            continue
        set_val = settings[i].split(':')
        sets.append(set_val[0])
        vals.append(float(set_val[1]))
    ddict = dict(zip(sets, vals))
    ddict['num_terms'] = int(ddict['num_terms'])
    ddict['num_topics'] = int(ddict['num_topics'])
    ddict['tops'] = int(ddict['tops'])
    ddict['batch_size'] = int(ddict['batch_size'])
    ddict['iter_infer'] = int(ddict['iter_infer'])
    ddict['iter_train'] = int(ddict['iter_train'])
    return(ddict)


def compute_perplexities_vb(beta, alpha, eta, max_iter,
                            wordids_1, wordcts_1, wordids_2, wordcts_2):
    """Compute perplexities, employing Variational Bayes."""
    vb = per_vb.VB(beta, alpha, eta, max_iter)
    LD2 = vb.compute_perplexity(wordids_1, wordcts_1, wordids_2, wordcts_2)
    return(LD2)


def compute_sparsity(doc_tp, batch_size, num_topics, _type):
    """Compute document sparsity.
    """
    sparsity = np.zeros(batch_size, dtype = np.float)
    if _type == 'z':
        for d in range(batch_size):
            N_z = np.zeros(num_topics, dtype = np.int)
            N = len(doc_tp[d])
            for i in range(N):
                N_z[doc_tp[d][i]] += 1.
            sparsity[d] = len(np.where(N_z != 0)[0])
    else:
        for d in range(batch_size):
            sparsity[d] = len(np.where(doc_tp[d] < 1e-20)[0])
            #sparsity[d] = len(np.where(doc_tp[d] > 0.01)[0])  # for Testing Sparsity of 1st theta
    sparsity /= num_topics
    return(np.mean(sparsity))
    #return (sparsity[0])  # for Testing Sparsity of 1st theta


def list_top(beta, tops):
    """Create list of top words of topics.
    """
    min_float = -sys.float_info.max
    num_tops = beta.shape[0]
    list_tops = list()
    for k in range(num_tops):
        top = list()
        arr = np.array(beta[k,:], copy = True)
        for t in range(tops):
            index = arr.argmax()
            top.append(index)
            arr[index] = min_float
        list_tops.append(top)
    return(list_tops)


def write_topics(beta, file_name):
    num_terms = beta.shape[1]
    num_topics = beta.shape[0]
    f = open(file_name, 'w')
    for k in range(num_topics):
        for i in range(num_terms - 1):
            f.write('%.10f '%(beta[k][i]))
        f.write('%.10f\n'%(beta[k][num_terms - 1]))
    f.close()

def write_topic_mixtures(theta, file_name):
    batch_size = theta.shape[0]
    num_topics = theta.shape[1]
    f = open(file_name, 'a')
    for d in range(batch_size):
        for k in range(num_topics - 1):
            f.write('%.5f '%(theta[d][k]))
        f.write('%.5f\n'%(theta[d][num_topics - 1]))
    f.close()

def write_perplexities(LD2, file_name):
    f = open(file_name, 'a')
    f.writelines('%f,'%(LD2))
    f.close()

def write_topic_top(list_tops, file_name):
    num_topics = len(list_tops)
    tops = len(list_tops[0])
    f = open(file_name, 'w')
    for k in range(num_topics):
        for j in range(tops - 1):
            f.write('%d '%(list_tops[k][j]))
        f.write('%d\n'%(list_tops[k][tops - 1]))
    f.close()

def write_sparsity(sparsity, file_name):
    f = open(file_name, 'a')
    f.write('%.10f,' % (sparsity))
    f.close()

def write_time(i, j, time_e, time_m, file_name):
    f = open(file_name, 'a')
    f.write('tloop_%d_iloop_%d, %f, %f, %f,\n'%(i, j, time_e, time_m, time_e + time_m))
    f.close()

def write_loop(i, j, file_name):
    f = open(file_name, 'w')
    f.write('%d, %d'%(i,j))
    f.close()

#changed ddict.keys() ---> list(ddict.keys())  &&   ddict.values() ---> list(ddict.values())
def write_setting(ddict, file_name):
    keys = list(ddict.keys())
    vals = list(ddict.values())
    f = open(file_name, 'w')
    for i in range(len(keys)):
        f.write('%s: %f\n'%(keys[i], vals[i]))
    f.close()

def diff_list_tops(list_tops, prev_list_tops, i):
    if i == 1:
        num_topics = len(list_tops)
        tops = len(list_tops[0])
        list_tops = np.array(list_tops)
        init = np.negative(np.ones([num_topics, tops], dtype=int))
        diff = init == list_tops
        diff_count = np.count_nonzero(diff)
        print(diff_count)
    else:
        list_tops = np.array(list_tops)
        diff = prev_list_tops == list_tops
        diff_count = np.count_nonzero(diff)
        print(diff_count)



def write_file(i, j, beta, time_e, time_m, theta, sparsity, LD2, list_tops, model_folder):
    #beta_file_name = '%s/beta_%d_%d.dat'%(model_folder, i, j)
    #theta_file_name = '%s/theta_%d.dat'%(model_folder, i)
    #per_file_name = '%s/perplexities_%d.csv'%(model_folder, i)
    #top_file_name = '%s/top%d_%d_%d.dat'%(model_folder, tops, i, j)
    list_tops_file_name = '%s/list_tops.txt'%(model_folder)
    #spar_file_name = '%s/sparsity_%d.csv'%(model_folder, i)
    #time_file_name = '%s/time_%d.csv'%(model_folder, i)
    #loop_file_name = '%s/loops.csv'%(model_folder)

    # write beta                                    ##
    #if j % 10 == 1:                                 ##
    #    write_topics(beta, beta_file_name)          ##
    # write theta                                   ##
    #write_topic_mixtures(theta, theta_file_name)    ##
    # write perplexities
    #write_perplexities(LD2, per_file_name)
    # write list top
    write_topic_top(list_tops, list_tops_file_name)
    # write sparsity
    #write_sparsity(sparsity, spar_file_name)
    # write time
    #write_time(i, j, time_e, time_m, time_file_name)
    # write loop
    #write_loop(i, j, loop_file_name)
