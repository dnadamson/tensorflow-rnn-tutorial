# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import math

def rnn_sampling_sequencer(data, sample_size, sequence_size, nb_epochs):
    """
    Input data must already be a batch of long sequences of pairs (tmin, tmax).
    The sequencer will split them up into consecutive sequences of length sequence_size.
    The data is also averaged over sequences of sample_size.
    When training, you have to preserve the state of the RNN through an epoch
    but reset it for each new epoch.
    """
    batch_size = data.shape[0]
    data_len = data.shape[1]
    pair_size = data.shape[2] # should be 2
    
    # resample
    data_len = data_len//sample_size*sample_size
    data = np.reshape(data[:,:data_len], [batch_size, -1, sample_size, pair_size])
    data = np.mean(data, axis=2)
    
    # shift by one for targets and split into consecutive sequences
    rounded_data_len = (data.shape[1]-1)//sequence_size*sequence_size
    xdata = np.reshape(data[:, 0:rounded_data_len + 0], [batch_size, -1, sequence_size, pair_size])
    ydata = np.reshape(data[:, 1:rounded_data_len + 1], [batch_size, -1, sequence_size, pair_size])
    nb_batches = xdata.shape[1]

    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:,batch,:,:]
            y = ydata[:,batch,:,:]
            yield x, y, epoch
        
    
def rnn_minibatch_sequencer(data, batch_size, sequence_size, nb_epochs):
    """
    Divides the data into batches of sequences so that all the sequences in one batch
    continue in the next batch. This is a generator that will keep returning batches
    until the input data has been seen nb_epochs times. Sequences are continued even
    between epochs, apart from one, the one corresponding to the end of data.
    The remainder at the end of data that does not fit in an full batch is ignored.
    :param data: the training sequence
    :param batch_size: the size of a training minibatch
    :param sequence_size: the unroll size of the RNN
    :param nb_epochs: number of epochs to train on
    :return:
        x: one batch of training sequences
        y: one batch of target sequences, i.e. training sequences shifted by 1
        epoch: the current epoch number (starting at 0)
    """
    data_len = data.shape[0]
    # using (data_len-1) because we must provide for the sequence shifted by 1 too
    nb_batches = (data_len - 1) // (batch_size * sequence_size)
    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    rounded_data_len = nb_batches * batch_size * sequence_size
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
    ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size])

    whole_epochs = math.floor(nb_epochs)
    frac_epoch = nb_epochs - whole_epochs
    last_nb_batch = math.floor(frac_epoch * nb_batches)
    
    for epoch in range(whole_epochs+1):
        for batch in range(nb_batches if epoch < whole_epochs else last_nb_batch):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            x = np.roll(x, -epoch, axis=0)  # to continue the sequence from epoch to epoch (do not reset rnn state!)
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch

            
def dumb_minibatch_sequencer(data, batch_size, sequence_size, nb_epochs):
    """
    Divides the data into batches of sequences in the simplest way: sequentially.
    :param data: the training sequence
    :param batch_size: the size of a training minibatch
    :param sequence_size: the unroll size of the RNN
    :param nb_epochs: number of epochs to train on
    :return:
        x: one batch of training sequences
        y: one batch of target sequences, i.e. training sequences shifted by 1
        epoch: the current epoch number (starting at 0)
    """
    data_len = data.shape[0]
    nb_batches = data_len // (batch_size * sequence_size)
    rounded_size = nb_batches * batch_size * sequence_size
    xdata = data[:rounded_size]
    ydata = np.roll(data, -1)[:rounded_size]
    xdata = np.reshape(xdata, [nb_batches, batch_size, sequence_size])
    ydata = np.reshape(ydata, [nb_batches, batch_size, sequence_size])

    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            yield xdata[batch,:,:], ydata[batch,:,:], epoch
