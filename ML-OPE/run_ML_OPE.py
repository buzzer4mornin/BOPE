#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os, shutil
import ML_OPE

sys.path.insert(0, '../common')
import utilities

"""
RUN ===>> ML-OPE ahmadli$ python ./run_ML_OPE.py ../data/ap.txt ../settings.txt ../models/nyt/ ../data
CTMP RUN> ML-OPE ahmadli$ python ./run_ML_OPE.py ../data/ctmp_input.txt ../settings.txt ../models/ctmp/ ../data
"""


# TODO: class attributes --> init --> self._attr (!! with underscores)

def main():
    # Check input
    if len(sys.argv) != 5:
        print("usage: python run_ML_OPE.py [train file] [setting file] [model folder] [test data folder]")
        exit()
    # Get environment variables
    train_file = sys.argv[1]
    setting_file = sys.argv[2]
    model_folder = sys.argv[3]
    test_data_folder = sys.argv[4]
    # Create model folder if it doesn't exist
    if os.path.exists(model_folder):
        shutil.rmtree(model_folder)
    os.makedirs(model_folder)
    # Read settings
    print('reading setting ...')
    ddict = utilities.read_setting(setting_file)
    print('write setting ...')
    file_name = '%s/setting.txt' % (model_folder)
    utilities.write_setting(ddict, file_name)
    # Read data for computing perplexities
    print('read data for computing perplexities ...')
    (wordids_1, wordcts_1, wordids_2, wordcts_2) = \
        utilities.read_data_for_perpl(test_data_folder)
    # ============================================= TILL HERE OKAY [0] =============================================
    # Initialize the algorithm
    print('initialize the algorithm ...')
    ml_ope = ML_OPE.MLOPE(ddict['num_terms'], ddict['num_topics'], ddict['alpha'],
                          ddict['tau0'], ddict['kappa'], ddict['iter_infer'])

    # Start
    print('start!!!')
    i = 0
    list_tops = []
    while i < ddict['iter_train']:
        i += 1
        print('\n***iter_train:%d***\n' % (i))
        datafp = open(train_file, 'r')
        j = 0
        while True:
            j += 1
            (wordids, wordcts) = utilities.read_minibatch_list_frequencies(datafp, ddict['batch_size'])
            # Stop condition
            if len(wordids) == 0:
                break
            # 
            print('---num_minibatch:%d---' % (j))
            (time_e, time_m, theta) = ml_ope.static_online(ddict['batch_size'], wordids, wordcts)
            # ========================= TILL HERE OKAY [1] ======================================
            # Compute sparsity
            sparsity = utilities.compute_sparsity(theta, theta.shape[0], theta.shape[1], 't')
            # print(sparsity)        # for Testing Sparsity of 1st theta
            # print(theta[0,:])      # for Testing Sparsity of 1st theta
            # Compute perplexities

            # LD2 = utilities.compute_perplexities_vb(ml_ope.beta, ddict['alpha'], ddict['eta'], ddict['iter_infer'], \
            #                                        wordids_1, wordcts_1, wordids_2, wordcts_2)
            LD2 = None

            # Saving previous list_tops for diff_list_tops() below
            prev_list_tops = list_tops

            # Search top words of each topics
            list_tops = utilities.list_top(ml_ope.beta, ddict['tops'])

            # TODO: add [last 25% avg diff count] to new file to compare later with other settings
            # Calculate and print difference between old and current list_tops
            utilities.diff_list_tops(list_tops, prev_list_tops, i)

            # Write files
            utilities.write_file(i, j, ml_ope.beta, time_e, time_m, theta, sparsity, LD2, list_tops, model_folder)
        datafp.close()
    # Write final model to file
    file_name = '%s/beta_final.dat' % (model_folder)
    utilities.write_topics(ml_ope.beta, file_name)
    # Finish
    print('done!!!')


if __name__ == '__main__':
    main()
