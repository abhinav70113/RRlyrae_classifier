import os
import numpy as np
from matplotlib import pyplot as plt

from astroML.datasets import fetch_rrlyrae_combined

import sys
import numpy as np
module_path = os.path.abspath(os.path.join('normalizing-flows/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from data.plot_samples import plot_samples_2d
from data.visu_density import plot_heatmap_2d
from data.dataset_loader import load_and_preprocess_uci
from normalizingflows.flow_catalog import NeuralSplineFlow
from utils.train_utils import train_density_no_tf, train_density_estimation, shuffle_split
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
import time
from utils.train_utils import sanity_check



class DataAnalyse():
    '''
    Builds the model and does the training
    '''
    def __init__(self, DATASET = "stars"):
        self.DATASET = DATASET
        if (DATASET == "stars"):
            self.mask_index = 0
        else:
            self.mask_index = 1
        # names of the features, i. e. color magnitudes
        self.Xfeatures = "u-g g-r r-i i-z".split()
    
        
    def prepare_data(self, BATCH_SIZE , intervals, data_train_all, data_validate_all, data_test_all):
        '''
        divide the data into batch sizes
        '''
        self.intervals = intervals
        self.data_train_all = data_train_all
        self.data_test_all = data_test_all
        self.data_validate_all = data_validate_all
        
        # masks data acc. to the dataset
        mask_train = self.data_train_all[:, 4] == self.mask_index
        mask_validate = self.data_validate_all[:, 4] == self.mask_index
        mask_test = self.data_test_all[:, 4] == self.mask_index

        self.data_train = self.data_train_all[mask_train][:, :4]
        self.data_validate = self.data_validate_all[mask_validate][:, :4]
        self.data_test = self.data_test_all[mask_test][:, :4]

        # slice into list-like objects, shape (4,)
        self.data_train = tf.data.Dataset.from_tensor_slices(self.data_train)
        self.data_validate = tf.data.Dataset.from_tensor_slices(self.data_validate)
        self.data_test = tf.data.Dataset.from_tensor_slices(self.data_test)

        # combine elements into batches, shape (None, 4)
        self.batched_train_data = self.data_train.batch(BATCH_SIZE)
        self.batched_val_data = self.data_validate.batch(BATCH_SIZE)
        self.batched_test_data = self.data_test.batch(BATCH_SIZE)

        # extract the shape of the input data,
        # we know it's 4, but for generality:
        sample_batch = next(iter(self.batched_train_data))
        self.input_shape = int(sample_batch.shape[1])

    def build_flow(self, LAYERS = 8, N_BINS = 24, SHAPE = [64, 64], BASE_LR = 0.005, f=None):
        '''
        # layers of bijectors
        LAYERS = 2
        # bins in the spline, Durkan et al.
        N_BINS = 64
        # shape of the layers
        SHAPE = [64, 64]
        BASE_LR = 0.005
        '''
        #----------------------------------
        # cast array [2, 3, 0, 1] to a tensor 
        permutation = tf.cast(np.array([3, 0, 1, 2])
                      , tf.int32)
        # 4-dim. standard normal
        base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(self.input_shape, tf.float32)) 

        bijector_chain = []
        for i in range(LAYERS):
    
            # using the NSF implemented by Rinder, based on Durkan et al.,
            # d_dim: trainable params (eqs. 9-11 in Durkan et al.), 
            # we choose half of the input-dim.
            bijector_chain.append(NeuralSplineFlow(input_dim=self.input_shape, 
                                                   d_dim=int(self.input_shape/2)+1, 
                                                   number_of_bins=N_BINS, 
                                                   b_interval=self.intervals))
    
            # permute to ensure all variables can interact
            bijector_chain.append(tfp.bijectors.Permute(permutation))

        # chain them together, in reverse
        # result: just another Bijector
        bijector = tfb.Chain(bijectors=list(reversed(bijector_chain)), name='chain_of_real_nvp')

        # the final flow with:
        #     * the 4-dim. standard normal as the base/target distribution
        #     * the bijector-chain as the transformation
        self.flow = tfd.TransformedDistribution(distribution=base_dist,
                                                   bijector=bijector)


        # number of trainable variables
        self.n_trainable_variables = self.flow.trainable_variables
        # len(n_trainable_variables)
        
        self.checkpoint_directory = "{}/tmp_{}_{}_{}_{}".format(self.DATASET, LAYERS, SHAPE[0], BASE_LR, N_BINS)
        self.checkpoint_prefix = os.path.join(self.checkpoint_directory, "ckpt")
        
        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=BASE_LR)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.flow)
        
        if f is not None:
            f.write("FLOW PARAMETERS\n")
            f.write(f"layers:{LAYERS},N_BINS:{N_BINS},SHAPE:{SHAPE}\n\n")
        
    
    def train_model(self, MAX_EPOCHS = 1000, DELTA_COUNT = 15, f=None):
        '''
        MAX_EPOCHS = 1000
        DELTA_COUNT = 15  : stop-cond. if too far from minimum
        '''
        if f is not None:
            f.write("TRAIN PARAMETERS\n")
            f.write(f"MAX_EPOCHS:{MAX_EPOCHS},DELTA_COUNT:{DELTA_COUNT}\n\n")
            f.write("BEGINNING TRAINING\n")
        
        global_step = []
        train_losses = []
        val_losses = []

        # high value to ensure that first loss < min_loss
        self.min_val_loss = tf.convert_to_tensor(np.inf, dtype=tf.float32)
        self.min_train_loss = tf.convert_to_tensor(np.inf, dtype=tf.float32)
        self.min_val_epoch = 0
        self.min_train_epoch = 0
        self.current_epoch = 0
        # threshold for early stopping
        delta = 0

        # start time
        t_start = time.time()
        
        for i in range(MAX_EPOCHS):
            # training of batches, one whole loop corresponds to all train_data 
            for batch in self.batched_train_data:
                self.train_loss = train_density_no_tf(self.flow, self.optimizer, batch)

            if i % int(10) == 0:
                batch_val_losses = []
                for batch in self.batched_val_data:
                    batch_loss = -tf.reduce_mean(self.flow.log_prob(batch))
                    batch_val_losses.append(batch_loss)
                self.val_loss = tf.reduce_mean(batch_val_losses)
                global_step.append(i)
                train_losses.append(self.train_loss)
                val_losses.append(self.val_loss)
                
                if f is not None:
                    f.write(f"EPOCH{i}, train_loss: {self.train_loss}, val_loss: {self.val_loss}\n")
        
                print(f"{i}, train_loss: {self.train_loss}, val_loss: {self.val_loss}")

                if self.train_loss < self.min_train_loss:
                    self.min_train_loss = self.train_loss
                    self.min_train_epoch = i

                if self.val_loss < self.min_val_loss:
                    self.min_val_loss = self.val_loss
                    self.min_val_epoch = i

                    self.checkpoint.write(file_prefix=self.checkpoint_prefix)

                elif i - self.min_val_epoch > DELTA_COUNT:
                    break
            self.current_epoch += 1
        
        self.train_time = time.time() - t_start

        print(f"Train time: {self.train_time//3600:.0f}:{(self.train_time//60) % 60:.0f}:{self.train_time%60:.0f} h")
        
        if f is not None:
            f.write(f"cehckpoint directory:{self.checkpoint_directory}\n")
        return self.checkpoint_directory
    
    
    def test_model(self, checkpoint_directory,f=None):
        '''
        loads the given checkpoint of a model and evaluates on the test data
        '''
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        self.checkpoint.restore(checkpoint_prefix)
        
        t_start = time.time()
        test_losses = []
        for batch in self.batched_test_data:
            batch_loss = -tf.reduce_mean(self.flow.log_prob(batch))
            test_losses.append(batch_loss)

        self.test_loss = tf.reduce_mean(test_losses)


        self.test_time = time.time() - t_start
        print(f"test loss: {self.test_loss}")
        print(f"Test time: {self.test_time//3600:.0f}:{(self.test_time//60) % 60:.0f}:{self.test_time%60:.3f} h")
        
    def summarize_training(self, f=None):

        if f is not None:
            f.write(f'Test loss: {self.test_loss} at epoch: {self.current_epoch}\n')
            f.write(f'Min val loss: {self.min_val_loss} at epoch: {self.min_val_epoch}\n')
            f.write(f'Min train loss: {self.min_train_loss} at epoch: {self.min_train_epoch}\n')
            f.write(f'Last val loss: {self.train_loss} at epoch: {self.current_epoch}\n')
            f.write(f'Training time: {self.train_time}\n')
            f.write(f'Test time: {self.test_time}\n')
            f.write(f'Train time: {self.train_time}\n')
            f.write(f'Train vaiables: {self.n_trainable_variables}\n')
            f.write("#############################################################\n\n")

        
        results = {
            'test_loss': float(self.test_loss),
            'min_val_loss': float(self.min_val_loss),
            'min_val_epoch': self.min_val_epoch,
            'val_loss': float(self.val_loss),
            'min_train_loss': float(self.min_train_loss),
            'min_train_epoch': self.min_train_epoch,
            'train_loss': float(self.train_loss),
            'train_time': self.train_time,
            'test_time': self.test_time,
            'trained_epochs': self.current_epoch,
            'trainable variables': self.n_trainable_variables,
        }
        
        return results
