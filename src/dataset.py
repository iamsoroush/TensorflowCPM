from urllib.request import urlretrieve
import os
import tarfile
import shutil
from tqdm import tqdm
import pickle

from scipy.io import loadmat

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2

import tensorflow as tf


class MPIIDataset:

    def __init__(self,
                 path='datasets',
                 heatmap_variance=2):
        self.path = os.path.join(path, 'mpii')
        self.images_dir = os.path.join(self.path, 'images')
        self.belief_maps_dir = os.path.join(self.path, 'believes')
        self.annotations_path = os.path.join(self.path, 'mpii_human_pose_v1_u12_1.mat')
        self.images_pickle_path = os.path.join(self.path, 'images.pkl')
        self.variance = heatmap_variance
        self.n_parts = 16
        self.gaussian_sigma = heatmap_variance * 4
        self.n_train = None
        self.n_test = None
        self.images = None

    def generate_dataset(self, src_tfr_dir=None):
        """Downloads and prepares MPII dataset.

        Args:
            src_tfr_dir: if you have generated belied maps before, give the path to belief maps as this arg.

        Returns:
            a list of paths to training set images
            a list of paths to training set belief maps
            a list of paths to testing set images
            a list of paths to testing set belief maps
        """

        self._download()
        self._generate_images_list()
        self._generate_or_copy_tfrecords(src_tfr_dir)
        return self._get_train_test_paths()

    def _download(self):

        """Downloads MPII dataset and places it under self.path directory"""

        if os.path.isdir(self.path):
            print('MPII dataset has been downloaded and already exists.')
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
        print('MPII downloaded. You can find the MPII dataset at ', self.path)

    def _generate_images_list(self):

        """Generates a list of ImageMPII instances and points self.images to generated list.

        This method will try to load a pre-generated list first from self.images_pickle_path
        """

        if os.path.isfile(self.images_pickle_path):
            print('Images pickle exists.')
            with open(self.images_pickle_path, 'rb') as pkl:
                images = pickle.load(pkl)
        else:
            print('Generating Image instances from images ...')
            mat = loadmat(self.annotations_path)
            is_train = mat['RELEASE']['img_train'][0, 0][0]
            img_idx = np.arange(len(is_train))
            img_names = [i['name'][0, 0][0] for i in mat['RELEASE']['annolist'][0, 0]['image'][0, :]]

            images = list()
            with tqdm(total=len(is_train)) as pbar:
                for ind in img_idx:
                    img_name = img_names[ind]
                    anno_rect = mat['RELEASE']['annolist'][0, 0]['annorect'][0][ind][0]
                    img = ImageMPII(img_ind=ind,
                                    img_name=img_name,
                                    is_train=is_train[ind],
                                    anno_rect=anno_rect,
                                    images_dir=self.images_dir,
                                    belief_maps_dir=self.belief_maps_dir)
                    images.append(img)
                    pbar.update(1)
        self.images = images
        return

    def _generate_or_copy_tfrecords(self, src_tfr_dir):

        """Handles 'making sure of correctly generated belief maps' !"""

        if os.path.exists(self.belief_maps_dir):
            tfrecord_paths = [img.tfrecord_path for img in self.images]
            existing_records = [os.path.join(self.belief_maps_dir, p) for p in os.listdir(self.belief_maps_dir)]
            if all(elem in existing_records for elem in tfrecord_paths):
                print('Tfrecords already exists.')
            else:
                if src_tfr_dir is None:
                    self._generate_tfrecords()
                else:
                    tfrecord_names = [img.tfrecord_name for img in self.images]
                    names = os.listdir(src_tfr_dir)
                    if all(elem in tfrecord_names for elem in names):
                        self._copy_tfrecords(src_tfr_dir)
                    else:
                        print('Incompatible tfrecords found in {}.'.format(src_tfr_dir))
                        self._generate_tfrecords()
        else:
            os.mkdir(self.belief_maps_dir)
            self._generate_tfrecords()
        return

    def _get_train_test_paths(self):

        """Returns 4 lists which are paths to training and testing data (images and belief maps).
        """

        train_img_paths = list()
        train_tfrecord_paths = list()
        test_img_paths = list()
        test_tfrecord_paths = list()

        for img in self.images:
            if img.is_train:
                train_img_paths.append(img.path)
                train_tfrecord_paths.append(img.tfrecord_path)
            else:
                test_img_paths.append(img.path)
                test_tfrecord_paths.append(img.tfrecord_path)
        return train_img_paths, train_tfrecord_paths, test_img_paths, test_tfrecord_paths

    def _generate_tfrecords(self):

        """Generates belief maps for images in self.images and saves them in self.belief_maps_dir

        This method uses tfrecords and sparse tensors to store generated belief maps.
        """

        print('Generating tfrecords ...')
        with tqdm(total=len(self.images)) as pbar:
            for img in self.images:
                img_joints = self._generate_joints_dict(img)
                belief_map = self._generate_belief_maps(img.height,
                                                        img.width,
                                                        img_joints)
                self._save_belief_map_as_sparse_tfrecord(belief_map, img)
                pbar.update(1)
        return

    def _generate_joints_dict(self, img):

        """Generates joints dictionary for an ImageMPII.

        Args:
            img: an instances of ImageMPII.

        Returns:
            a dictionary with this structure:
                {joint_index: [joint1, joint2],
                 joint_index: [joint1, joint2, joint3],
                  ...}

            Each item in above list is an instance of Joint class.
        """

        img_joints = {k: list() for k in range(self.n_parts)}
        for person in img.persons:
            for vis_joint in person.visible_joints:
                img_joints[vis_joint.index].append(vis_joint)
        return img_joints

    def _generate_belief_maps(self, img_height, img_width, joints_dict):

        """Generates belief maps for provided joints.

        Args:
            img_height: height of original image
            img_width: width of original image
            joints_dict: a dictionary with keys as joint index and values a list of Joint instances.

        Returns:
            Generated belief map with shape (img_height, img_width, self.n_parts)
        """

        believes = np.zeros((img_height, img_width, self.n_parts))
        gaussian = self._make_gaussian()
        for joint_ind, joint_data in joints_dict.items():
            if not joint_data:
                continue
            else:
                for joint in joint_data:
                    gaussian_map = self._generate_gaussian_map(gaussian, img_height, img_width, joint.x, joint.y)
                    believes[:, :, joint_ind] += gaussian_map
        return believes

    def _make_gaussian(self):

        """Makes a matrix with 2d gaussian elements.

        This will return a 2d array with shape (self.gaussian_sigma * 2, self.gaussian_sigma * 2), and elements
            follow a 2d gaussian distribution with variance=self.variance.
        """

        size = int(self.gaussian_sigma * 2)
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / self.variance / self.variance)

    def _generate_gaussian_map(self, gaussian, img_height, img_width, c_x, c_y):

        """ Generates gaussian map for a single joint, and provided x-y coordinates."""

        gaussian_map = np.zeros((img_height, img_width))
        ylt = int(max(0, c_y - self.gaussian_sigma / 2 * self.variance))
        yld = int(min(img_height, c_y + self.gaussian_sigma / 2 * self.variance))
        xll = int(max(0, c_x - self.gaussian_sigma / 2 * self.variance))
        xlr = int(min(img_width, c_x + self.gaussian_sigma / 2 * self.variance))

        if (xll >= xlr) or (ylt >= yld):
            return gaussian_map

        gaussian_map[ylt: yld,
                     xll: xlr] = gaussian[: yld - ylt,
                                          : xlr - xll]
        return gaussian_map

    def _save_belief_map_as_sparse_tfrecord(self, belief_map, img):

        """Saves provided belief map as sparse tfrecord, to img.tfrecord_path"""

        file_path = img.tfrecord_path
        img_height = img.height
        img_width = img.width

        wheres = np.where(belief_map != 0)
        values = belief_map[wheres]

        my_example = tf.train.Example(features=tf.train.Features(feature={
            'index_0': tf.train.Feature(int64_list=tf.train.Int64List(value=wheres[0])),
            'index_1': tf.train.Feature(int64_list=tf.train.Int64List(value=wheres[1])),
            'index_2': tf.train.Feature(int64_list=tf.train.Int64List(value=wheres[2])),
            'values': tf.train.Feature(float_list=tf.train.FloatList(value=values)),
            'dense_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_height, img_width, self.n_parts]))
        }))
        my_example_str = my_example.SerializeToString()
        with tf.io.TFRecordWriter(file_path) as writer:
            writer.write(my_example_str)
        return

    def _copy_tfrecords(self, src_tfr_dir):

        """Copies generated tfrecords from provided src directory.
        """

        print('Copying tfrecords from {} to {} ...'.format(src_tfr_dir, self.belief_maps_dir))
        self._copytree(src_tfr_dir, self.belief_maps_dir)
        return

    @staticmethod
    def _copytree(src, dst):
        if not os.path.exists(dst):
            os.mkdir(dst)

        files = os.listdir(src)

        with tqdm(total=len(files)) as pbar:
            for name in files:
                src_path = os.path.join(src, name)
                dst_path = os.path.join(dst, name)

                shutil.copyfile(src_path, dst_path)
                pbar.update(1)


def create_tf_dataset(img_paths, tfrecord_paths, target_img_size, batch_size):

    """Creates tenosrflow dataset with resized images as input and resized belief maps as output.

    Get path lists to training and testing data using MPIIDataset, and create train and test datasets using this func.
    Warning: use tensorflow2.0
    """

    n_imgs = len(img_paths)
    target_size_list = [n_imgs * target_img_size]

    img_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    bm_ds = tf.data.TFRecordDataset(tfrecord_paths)
    target_size_ds = tf.data.Dataset.from_tensor_slices(target_size_list)

    zipped = tf.data.Dataset.zip((img_ds, bm_ds, target_size_ds))
    ds = zipped.shuffle(n_imgs).repeat().map(_load_data).batch(batch_size).prefetch(1)
    return ds


def _load_and_preprocess_image(path, target_size_ds):
    image = tf.io.read_file(path)
    return _preprocess_image(image, target_size_ds)


def _preprocess_image(image, target_size_ds):
    target_h, target_w = target_size_ds
    image = tf.image.decode_jpeg(image, channels=3)
    image /= 255  # normalize to [0,1] range
    image = tf.image.resize_with_pad(image, target_h, target_w)
    return image


def _parse_bm(tfr, target_size_ds):

    """Parses sparse belief maps saved in .tfrecord format, resizes to target size and returns resulted tensor."""

    target_h, target_w = target_size_ds

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
    belief_maps = tf.sparse.to_dense(st)

    resized = tf.image.resize_with_pad(belief_maps, target_h, target_w)
    return resized


def _load_data(img_path, tfr, target_size):
    img = _load_and_preprocess_image(img_path, target_size)
    belief_maps = _parse_bm(tfr, target_size)
    return img, belief_maps


class ImageMPII:

    def __init__(self,
                 img_ind,
                 img_name,
                 is_train,
                 anno_rect,
                 images_dir,
                 belief_maps_dir):
        self.ind = img_ind
        self.name = img_name
        self.path = os.path.join(images_dir, img_name)
        self.is_train = is_train
        sh = cv2.imread(self.path).shape
        self.height = sh[0]
        self.width = sh[1]
        self.persons = [Person(p, img_ind, i) for i, p in enumerate(anno_rect)]
        self.tfrecord_name = img_name.split('.')[0] + '.tfrecord'
        self.tfrecord_path = os.path.join(belief_maps_dir, self.tfrecord_name)

    def show(self, mode='joints', variance=5):
        img = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(img)

        if mode == 'joints':
            gaussian = self._make_gaussian(variance)
            hm = np.zeros((self.height, self.width), dtype=np.float32)
            for person in self.persons:
                for joint in person.visible_joints:
                    self._add_gaussian(hm, gaussian, joint.x, joint.y, variance)
            ax.imshow(hm, alpha=0.6, cmap='viridis')
        elif mode == 'head':
            for person in self.persons:
                head = person.head
                width = np.abs(head.x1 - head.x2)
                height = np.abs(head.y1 - head.y2)
                rect = patches.Rectangle((head.x1, head.y1),
                                         width=width,
                                         height=height,
                                         fill=False,
                                         linewidth=1,
                                         edgecolor='r',
                                         facecolor=None)
                ax.add_patch(rect)
        elif mode == 'persons':
            for person in self.persons:
                ax.scatter(person.x, person.y)
        else:
            raise Exception('Mode must be one of {joints, head, persons}.')
        plt.show()

    @staticmethod
    def _add_gaussian(heatmap, gaussian, c_x, c_y, variance):
        img_height, img_width = heatmap.shape
        ylt = int(max(0, int(c_y) - 4 * variance))
        yld = int(min(img_height, int(c_y) + 4 * variance))
        xll = int(max(0, int(c_x) - 4 * variance))
        xlr = int(min(img_width, int(c_x) + 4 * variance))

        if (xll >= xlr) or (ylt >= yld):
            return heatmap

        heatmap[ylt: yld,
                xll: xlr] = gaussian[: yld - ylt,
                                     : xlr - xll]
        return

    @staticmethod
    def _make_gaussian(variance):
        size = int(8 * variance)
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        return (np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / variance / variance)).astype(np.float32)


class Person:

    def __init__(self, person_anno, img_ind, person_ind):
        self.img_ind = img_ind
        self.person_ind = person_ind

        fields = person_anno.dtype.names

        if 'x1' in fields:
            self.head = Head(person_anno['x1'][0, 0],
                             person_anno['y1'][0, 0],
                             person_anno['x2'][0, 0],
                             person_anno['y2'][0, 0])
        else:
            self.head = Head(None,
                             None,
                             None,
                             None)
        if 'scale' in fields:
            self.scale = person_anno['scale'][0, 0]
        else:
            self.scale = None
        if 'objpos' in fields:
            self.x = person_anno['objpos'][0, 0][0][0][0]
            self.y = person_anno['objpos'][0, 0][1][0][0]
        else:
            self.x = None
            self.y = None
        if 'annopoints' in fields:
            joints = person_anno['annopoints'][0]['point'][0][0]
            self.visible_joints = [Joint(joint) for joint in joints if np.any(joint[3])]
        else:
            self.visible_joints = list()


class Head:

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class Joint:
    joint_names = ['r ankle', 'r knee', 'r hip', 'l hip', 'l knee',
                   'l ankle', 'pelvis', 'thorax', 'upper neck',
                   'head top', 'r wrist', 'r elbow', 'r shoulder',
                   'l shoulder', 'l elbow', 'l wrist']
    mapping_dict = dict(zip(range(len(joint_names)), joint_names))

    def __init__(self, joint):
        self.index = joint[2][0, 0]
        self.name = Joint.mapping_dict[self.index]
        self.x = joint[0][0, 0]
        self.y = joint[1][0, 0]
