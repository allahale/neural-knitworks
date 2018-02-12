from PIL import Image
import numpy as np
from keras.models import load_model
from tensorflow.python.lib.io import file_io

import argparse


def plot_pattern(arr):
    img = np.reshape(arr, (-1, 24))
    return Image.fromarray(img.astype(np.uint8) * 255).resize((img.shape[1] * 13, img.shape[0] * 13))


def gen_predictions(model, seed, max_len, T):
    output = seed
    for i in range(max_len-seed.shape[1]):
        pred = model.predict(output)
        next_stitch = sample(pred[None, :, -1, :], T)
        output = np.append(output, next_stitch, axis=1)
    return output


def sample(p, T):
    p_t = p ** (1 / (T + 1e-3))
    np_t = (1 - p) ** (1 / (T + 1e-3))
    s = p_t + np_t
    p_t /= s

    return np.random.rand() < p_t


def generate(model_file, data_file, job_dir, n_examples=3, temps=None, gen_len=1200, seed_len=48):

    if not temps:
        temps = [0.1, 0.5, 1.0]

    #load real data to use as seed
    file_stream = file_io.FileIO(data_file, mode='rb')
    data_dict = np.load(file_stream)
    data_list = list(data_dict[()].values())
    data_unfolded = [np.ravel(d, order='C').astype(np.uint8) for d in data_list if d.shape[1] == 24]

    from keras.preprocessing.sequence import pad_sequences
    MAX_LEN = 2400
    data_repeated = [np.tile(x, MAX_LEN // x.shape[0] + 1) for x in data_unfolded]
    pad = pad_sequences(data_repeated, maxlen=MAX_LEN, dtype=np.uint8, value=2,
                        padding='post', truncating='post')

    #load keras models
    with file_io.FileIO(model_file, mode='rb') as input_f:
        with file_io.FileIO('tmp_model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())
    model = load_model('tmp_model.h5')

    seed_idx = np.random.choice(range(pad.shape[0]), n_examples)

    for i in seed_idx:
        seed = pad[None, i, :seed_len, None]
        for j, T in enumerate(temps):
            print(f'generating image {i*len(temps) + j +1} of {len(seed_idx)*len(temps)}')
            arr = gen_predictions(model, seed, gen_len, T)
            img = plot_pattern(arr)
            img_stream = file_io.FileIO(job_dir + f'S:{i}-{seed_len}_T:{T}.png', mode='wb+')
            img.save(img_stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-file',
        help='GCS or local paths to real data',
        required=True)

    parser.add_argument(
        '--model-file',
        help='GCS or local paths to trained model',
        required=True)

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True)

    parser.add_argument('--n_examples', help='number of seed examples to generate', type=int)
    parser.add_argument('--gen_len', help='length of the generated sequence', type=int)
    parser.add_argument('--seed_len', help='length of the seed given to generator', type=int)

    args = parser.parse_args()
    arguments = args.__dict__
    arguments = {k: v for k, v in arguments.items() if v is not None}

    generate(**arguments)
