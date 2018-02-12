import argparse
import numpy as np

from tensorflow.python.lib.io import file_io

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import CuDNNLSTM, CuDNNGRU, LSTM, TimeDistributed
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop

from .GCSModelCheckpoint import GCSModelCheckpoint

def train_model(train_file='russian punch cards/processedarrays.npy', job_dir='./tmp/example-5', log_dir='./tmp/logs',
                dropout=0.5, rnn_size=128, rnn_activation='tanh', rnn_layers=1, rnn_cell='LSTM', lr_decay=0,
                batch_size=64, epochs=100, saved_model='model.h5', test=False, **args):
    file_stream = file_io.FileIO(train_file, mode='rb')
    data_dict = np.load(file_stream)
    data_list = list(data_dict[()].values())
    data_unfolded = [np.ravel(d, order='C').astype(np.uint8) for d in data_list if d.shape[1] == 24]

    MAX_LEN = 2400
    data_repeated = [np.tile(x, MAX_LEN//x.shape[0]+1) for x in data_unfolded]
    pad = pad_sequences(data_repeated, maxlen=MAX_LEN, dtype=np.uint8, value=2,
                        padding='post', truncating='post')

    if rnn_cell == 'LSTM':
        if test:
            cell = LSTM
        else:
            cell = CuDNNLSTM
    elif rnn_cell == 'GRU':
        cell = CuDNNGRU
    else:
        print('unknown rnn cell type, defaulting to LSTM')
        cell = CuDNNLSTM

    model = Sequential()
    model.add(cell(rnn_size, return_sequences=True, batch_input_shape=(None, None, 1)))
    model.add(Dropout(dropout))
    for i in range(rnn_layers-1):
        model.add(cell(rnn_size, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(Activation(rnn_activation))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    optimizer = RMSprop(clipnorm=1., decay=lr_decay)

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['binary_accuracy'])

    if test:
        X = pad[:256, :-1, None]
        y = pad[:256, 1:, None]
    else:
        X = pad[:, :-1, None]
        y = pad[:, 1:, None]

    ckpt = GCSModelCheckpoint('epoch_{epoch}_' + saved_model, job_dir + '/models', monitor='val_binary_accuracy',
                              save_best_only=True, period=10)
    tb = TensorBoard(log_dir=log_dir+'/'+job_dir.split('/')[-1])
    model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[tb, ckpt])

    model.save(saved_model)

    # Save model.h5 on to google storage
    with file_io.FileIO(saved_model, mode='rb') as input_f:
        with file_io.FileIO(job_dir + '/' + saved_model, mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-file',
        help='GCS or local paths to training data',
        required=True
    )

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )

    parser.add_argument(
        '--log-dir',
        help='GCS location to write logs',
        required=True
    )

    parser.add_argument('--dropout', help='rnn dropout', type=float)
    parser.add_argument('--rnn_size', help='RNN unit size', type=int)
    parser.add_argument('--rnn_activation', help='RNN activation function', type=str)
    parser.add_argument('--rnn_layers', help='Num. of stacked RNN layers', type=int)
    parser.add_argument('--rnn_cell', help='type of RNN cell, only accepts GRU or LSTM', type=str)
    parser.add_argument('--batch_size', help='training batch size', type=int)
    parser.add_argument('--epochs', help='training epochs', type=int)
    parser.add_argument('--save-model', help='filename for saving trained model', type=str)
    #parser.add_argument('--optimizer', help='optimizer to use for sgd', type=str)
    parser.add_argument('--lr-decay', help='learning rate decay', type=float)

    parser.add_argument('--test', help='run in local test mode', type=bool)

    args = parser.parse_args()
    arguments = args.__dict__
    arguments = {k: v for k, v in arguments.items() if v is not None}

    train_model(**arguments)
