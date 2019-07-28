from urllib.request import urlretrieve
import os
import tarfile
import shutil
import json
from tqdm import tqdm

from scipy.io import loadmat

import numpy as np
import pandas as pd

import tensorflow as tf


class MPII:

    def __init__(self, path='pose_dataset', test_size=0.1,
                 heatmap_variance=3, n_parts=16, batch_size=32):
        self.path = os.path.join(path, 'mpii')
        self.images_path = os.path.join(self.path, 'images')
        self.belief_maps_path = os.path.join(self.path, 'believes')
        self.joints_path = os.path.join(path, 'data.json')
        self.test_size = test_size
        self.variance = heatmap_variance
        self.n_parts = n_parts
        self.batch_size = batch_size
        self.image_paths = None
        self.joints_list = None
        self.tfrecord_paths = None
        self.img_names = None
        self.train_ind = None
        self.test_ind = None

    def generate_dataset(self):
        """Generates and prepares MPII dataset.

        Returns paths to images and belief maps for train and test.
        """
        self._download()
        self._save_joints()
        joints_df = pd.read_json(self.joints_path, lines=True)
        self._generate_img_paths_joints(joints_df)
        self._generate_tfrecords()
        self._train_test_split()
        train_img = self.image_paths[self.train_ind]
        train_tfrecord = self.tfrecord_paths[self.train_ind]
        test_img = self.image_paths[self.test_ind]
        test_tfrecord = self.tfrecord_paths[self.test_ind]

        train_ds = self.create_dataset(train_img, train_tfrecord)
        test_ds = self.create_dataset(test_img, test_tfrecord)
        return train_ds, test_ds

    def _download(self):
        if os.path.isdir(self.path):
            print('MPII dataset already exists.')
            return

        file_name1 = 'mpii_human_pose_v1_u12_1.tar.gz'
        url = 'http://datasets.d2.mpi-inf.mpg.de/leonid14cvpr/mpii_human_pose_v1_u12_1.tar.gz'
        print('Downloading file {} ...'.format(file_name1))
        urlretrieve(url, file_name1)

        print('Unzipping the file {} ...'.format(file_name1))
        with tarfile.open(file_name1) as zipf:
            zipf.extractall()
        shutil.move('mpii_human_pose_v1_u12_1/', self.path)

        file_name2 = 'mpii_human_pose_v1.tar.gz'
        url = 'http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz'
        print('Downloading file {} ...'.format(file_name2))
        urlretrieve(url, file_name2)

        print('Unzipping the file {} ...'.format(file_name2))
        with tarfile.open(file_name2) as zipf:
            zipf.extractall()

        os.remove(file_name1)
        os.remove(file_name2)

        shutil.move('images', self.path)
        print('Done. You can find the MPII dataset at ', self.path)

    def _save_joints(self):
        joint_data_fn = self.joints_path
        if os.path.exists(joint_data_fn):
            return
        mat = loadmat(os.path.join(self.path, 'mpii_human_pose_v1_u12_1.mat'))

        fp = open(joint_data_fn, 'w')

        for i, (anno, train_flag) in enumerate(
                zip(mat['RELEASE']['annolist'][0, 0][0],
                    mat['RELEASE']['img_train'][0, 0][0])):

            img_fn = anno['image']['name'][0, 0][0]
            train_flag = int(train_flag)

            if 'annopoints' in str(anno['annorect'].dtype):
                annopoints = anno['annorect']['annopoints'][0]
                head_x1s = anno['annorect']['x1'][0]
                head_y1s = anno['annorect']['y1'][0]
                head_x2s = anno['annorect']['x2'][0]
                head_y2s = anno['annorect']['y2'][0]
                for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(
                        annopoints, head_x1s, head_y1s, head_x2s, head_y2s):
                    if annopoint != []:
                        head_rect = [float(head_x1[0, 0]),
                                     float(head_y1[0, 0]),
                                     float(head_x2[0, 0]),
                                     float(head_y2[0, 0])]

                        # joint coordinates
                        annopoint = annopoint['point'][0, 0]
                        j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                        x = [x[0, 0] for x in annopoint['x'][0]]
                        y = [y[0, 0] for y in annopoint['y'][0]]
                        joint_pos = {}
                        for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                            joint_pos[str(_j_id)] = [float(_x), float(_y)]

                        # visiblity list
                        if 'is_visible' in str(annopoint.dtype):
                            vis = [v[0] if v else [0]
                                   for v in annopoint['is_visible'][0]]
                            vis = dict([(k, int(v[0])) if len(v) > 0 else v
                                        for k, v in zip(j_id, vis)])
                        else:
                            vis = None

                        if len(joint_pos) == 16:
                            data = {
                                'filename': img_fn,
                                'train': train_flag,
                                'head_rect': head_rect,
                                'is_visible': vis,
                                'joint_pos': joint_pos
                            }

                            print(json.dumps(data), file=fp)

    def _generate_img_paths_joints(self, joints_df):
        g = joints_df.groupby('filename')['joint_pos']

        img_paths = list()
        joints_list = list()

        for gn in g.groups:
            group = g.get_group(gn)
            persons = group.values

            joints = [list() for _ in range(self.n_parts)]
            for person in persons:
                for k, v in person.items():
                    joints[int(k)].append(v)
            joints_list.append(joints)
            img_paths.append(os.path.join(self.images_path, gn))
        self.image_paths = img_paths
        self.joints_list = joints_list
        img_names = [i.split('/')[-1].split('.')[0] for i in img_paths]
        self.img_names = img_names

    def _generate_tfrecords(self):

        def sort_func(elem):
            elem_fn = elem.split('/')[-1].split('.')[0]
            return img_names.index(elem_fn)

        tfrecord_paths = list()
        if os.path.exists(self.belief_maps_path):
            if len(os.listdir(self.belief_maps_path)) == len(self.image_paths):
                print('tfrecords already generated.')
                tfrecord_paths = os.listdir(self.belief_maps_path)
        else:
            os.mkdir(self.belief_maps_path)
            for i in tqdm(range(len(self.image_paths))):
                img_path = self.image_paths[i]
                joints = self.joints_list[i]
                img = cv2.imread(img_path)
                h, w, _ = img.shape
                belief_map = self._generate_believes(h, w, joints)
                st_path = self._save_to_sparse_tfrecord(img_path, h, w, belief_map)
                tfrecord_paths.append(st_path)

        img_names = self.img_names
        tfrecord_paths.sort(key=sort_func)
        self.tfrecord_paths = tfrecord_paths

    @staticmethod
    def _generate_joint_pos(row, joint):
        if not row['is_visible'][str(joint)]:
            return np.array([np.nan, np.nan])
        else:
            return np.array(row['joint_pos'][str(joint)])

    def _train_test_split(self):
        n_instances = len(self.image_paths)
        indices = np.arange(n_instances)
        np.random.shuffle(indices)
        n_test = int(n_instances * self.test_size)
        n_train = n_instances - n_test
        self.train_ind = indices[:n_train]
        self.test_ind = indices[n_train:]

    @staticmethod
    def _generate_gaussian_img(gaussian, img_height, img_width, c_x, c_y, variance):
        gaussian_map = np.zeros((img_height, img_width))
        ylt = int(max(0, int(c_y) - 4 * variance))
        yld = int(min(img_height, int(c_y) + 4 * variance))
        xll = int(max(0, int(c_x) - 4 * variance))
        xlr = int(min(img_width, int(c_x) + 4 * variance))

        if (xll >= xlr) or (ylt >= yld):
            return gaussian_map

        gaussian_map[ylt: yld,
                     xll: xlr] = gaussian[: yld - ylt,
                                          : xlr - xll]
        return gaussian_map

    @staticmethod
    def _make_gaussian(variance):
        size = int(8 * variance)
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / variance / variance)

    def _generate_believes(self, h, w, joint_pos):
        believes = np.zeros((h, w, self.n_parts))
        gaussian = self._make_gaussian(self.variance)
        for i, joint in enumerate(joint_pos):
            for person in joint:
                gaus_img = self._generate_gaussian_img(gaussian, h, w, person[0], person[1], self.variance)
                believes[:, :, i] += gaus_img
        return believes

    def _save_to_sparse_tfrecord(self, img_path, h, w, belief_map):
        file_name = img_path.split('/')[-1].split('.')[0]
        file_path = os.path.join(self.belief_maps_path, file_name + '.tfrecord')

        wheres = np.where(belief_map != 0)
        values = belief_map[wheres]

        my_example = tf.train.Example(features=tf.train.Features(feature={
            'index_0': tf.train.Feature(int64_list=tf.train.Int64List(value=wheres[0])),
            'index_1': tf.train.Feature(int64_list=tf.train.Int64List(value=wheres[1])),
            'index_2': tf.train.Feature(int64_list=tf.train.Int64List(value=wheres[2])),
            'values': tf.train.Feature(float_list=tf.train.FloatList(value=values)),
            'dense_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[h, w, self.n_parts]))
        }))
        my_example_str = my_example.SerializeToString()
        with tf.io.TFRecordWriter(file_path) as writer:
            writer.write(my_example_str)
        return file_path

    def create_dataset(self, img_paths, tfrecord_paths):
        def load_and_preprocess_image(path):
            image = tf.io.read_file(path)
            return preprocess_image(image)

        def preprocess_image(image):
            image = tf.image.decode_jpeg(image, channels=3)
            image /= 255  # normalize to [0,1] range
            return image

        def parse_bm(tfr):
            features = {'index_0': tf.io.VarLenFeature(dtype=tf.int64),
                        'index_1': tf.io.VarLenFeature(dtype=tf.int64),
                        'index_2': tf.io.VarLenFeature(dtype=tf.int64),
                        'values': tf.io.VarLenFeature(dtype=tf.float32),
                        'dense_shape': tf.io.FixedLenFeature(shape=(3,), dtype=tf.int64)}
            parsed = tf.io.parse_single_example(tfr, features=features)
            ind0 = tf.sparse.to_dense(parsed['index_0'])
            ind1 = tf.sparse.to_dense(parsed['index_1'])
            ind2 = tf.sparse.to_dense(parsed['index_2'])
            values = tf.sparse.to_dense(parsed['values'])
            shape = parsed['dense_shape']

            indices = tf.stack([ind0, ind1, ind2], axis=1)
            st = tf.SparseTensor(values=values, indices=indices, dense_shape=shape)
            return tf.sparse.to_dense(st)

        def load_data(img_path, tfr):
            img = load_and_preprocess_image(img_path)
            belief_maps = parse_bm(tfr)
            return img, belief_maps

        img_ds = tf.data.Dataset.from_tensor_slices(img_paths)
        bm_ds = tf.data.TFRecordDataset(tfrecord_paths)
        zipped = tf.data.Dataset.zip((img_ds, bm_ds))
        ds = zipped.shuffle(len(img_paths)).repeat().map(load_data).batch(self.batch_size).prefetch(1)
        return ds
