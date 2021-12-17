import argparse
import pickle
import cv2
import os
import numpy as np
from keras.models import load_model, Model
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from keras.applications.vgg16 import VGG16, preprocess_input
from tqdm import tqdm

import config
from dataset.load_dataset import ClusterDataset
from models import block_size
import sklearn.externals
import joblib


def save_cluster(filename, km):
    # It is important to use binary access
#    with open(filename, 'wb') as f:
 #       pickle.dump(km, f)
    pickle.dump(km, open(filename, 'wb'))


def load_cluster(filename):
#    with open(filename, 'rb') as f:
 #       km = pickle.load(f)
    km = pickle.load(open(filename, 'rb'))

    return km

def get_model(layer='fc2'):
    # base_model.summary():
    #     ....
    #     block5_conv4 (Conv2D)        (None, 15, 15, 512)       2359808
    #     _________________________________________________________________
    #     block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
    #     _________________________________________________________________
    #     flatten (Flatten)            (None, 25088)             0
    #     _________________________________________________________________
    #     fc1 (Dense)                  (None, 4096)              102764544
    #     _________________________________________________________________
    #     fc2 (Dense)                  (None, 4096)              16781312
    #     _________________________________________________________________
    #     predictions (Dense)          (None, 1000)              4097000
    #
    base_model = VGG16(weights='imagenet', include_top=True)
    print(base_model.input)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer(layer).output)
    return model

def write_list(above_path, left_path, y_path, h, w):
    file_path = os.path.join(y_path[:-2], 'seq.txt')
    if os.path.exists(file_path): return 0
    return_path = []
    above_size = block_size[h][w][0]
    left_size = block_size[h][w][1]
    y_size = (h, w)
    error_num = 0
    for seq in tqdm(sorted(os.listdir(y_path))):
        above = np.int8(np.load(os.path.join(above_path, seq)))
        left = np.int8(np.load(os.path.join(left_path, seq)))
        y = np.int8(np.load(os.path.join(y_path, seq)))
        if above_size != above.shape or left_size != left.shape or y_size!=y.shape:
            error_num+=1
            continue
        return_path.append(seq)
    print(error_num)
    with open(file_path, 'w') as f:
        for i in return_path:
            f.write(i+"\n")

def check_list(y_path):
    with open(os.path.join(y_path[:-2], "seq.txt"), "r") as f:
        return_path =  f.read().splitlines()
    return return_path

def feature_vector(img_arr, model):
    if img_arr.shape[2] == 1:
        img_arr = img_arr.repeat(3, axis=2)

    # (1, 224, 224, 3)
    arr4d = np.expand_dims(img_arr, axis=0)
    arr4d_pp = preprocess_input(arr4d)
    print(arr4d_pp.shape)
    return model.predict(arr4d_pp)[0, :]

def feature_vectors(dataloader, model):
    f_vect = {}
    for i, (input_tensor, _) in enumerate(dataloader):
        f_vect[i] = model.predict(input_tensor)[0, :]
    return f_vect

def main(args):
    if args.data is None:
        args.data = config.train_numpy_path
    path_to_files = os.path.join(args.data,  str(args.quality), str(args.height)+"x"+str(args.width))
    path_to_save = os.path.join(config.cluster_checkpoint, str(args.height)+"x"+str(args.width) + "_" + str(args.quality) + '.pkl')

    # Create Keras NN model.
    model = get_model()

    k = 3
    if args.resume:
        km = load_cluster(path_to_save)
    else:
        km = KMeans(n_clusters=k, init='k-means++', max_iter=200)

    images = prepare_image(path_to_files)
    km = km.fit(images)
    save_cluster(path_to_save, km)

    images = prepare_image(os.path.join(config.valid_numpy_path,  str(args.quality), str(args.height)+"x"+str(args.width)))
    predict = inference(images, km)
    for i, value in enumerate(predict):
        if i%10==0: print()
        print(value, end=" ")


def prepare_image(path_to_files):
    imgs_dict = get_files(path_to_files=path_to_files, h=args.height, w=args.width)
    nsamples, nx, ny = imgs_dict.shape
    imgs_dict = imgs_dict.reshape((nsamples, nx * ny))
    return imgs_dict

def get_files(path_to_files, h, w):
    above_path = os.path.join(path_to_files, "above")
    left_path = os.path.join(path_to_files, "left")
    y_path = os.path.join(path_to_files, "y")

    write_list(above_path, left_path, y_path, h, w)
    samples = check_list(y_path)

    fn_imgs = []
    for seq in samples:
        above = np.int8(np.load(os.path.join(above_path, seq)))
        left = np.int8(np.load(os.path.join(left_path, seq)))
        picture = np.zeros((above.shape[1], above.shape[1]), np.uint8)
        picture[:above.shape[0], :above.shape[1]] = above
        picture[above.shape[0]:, :above.shape[0]] = left
        fn_imgs.append(picture)
    return np.array(fn_imgs)

def inference(images, km):
    return km.predict(images)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument("--data", "--dataset", type=str, help='path to dataset')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', action='store_true',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=25000,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=22,
        help="Quality (default: %(default)s)",
    )
    parser.add_argument("-hgt", "--height", type=int, default=32, help="block height")
    parser.add_argument("-wdt", "--width", type=int, default=32, help="block width")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)