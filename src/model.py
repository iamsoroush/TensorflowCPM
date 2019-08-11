import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras as tfk
tfkl = tfk.layers
tfkb = tfk.backend


class KerasModel:

    def __init__(self, model_name, models_dir):
        self.model_name = model_name

        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)

        self.model_dir = os.path.join(models_dir, model_name)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.logs_dir = os.path.join(self.model_dir, 'logs')
        if not os.path.exists(self.logs_dir):
            os.mkdir(self.logs_dir)

        self.checkpoints = list()
        self.logs = list()
        self._update_logs_checkpoints_paths()

        self.model_ = None

    def _update_logs_checkpoints_paths(self):
        self.checkpoints = [os.path.join(self.model_dir, f) for f in os.listdir(self.model_dir)]
        self.logs = [os.path.join(self.logs_dir, f) for f in os.listdir(self.logs_dir)]

    def load_model(self, checkpoint_name=None):
        if checkpoint_name is None:
            chkpt_path = self.checkpoints[-1]
        else:
            chkpt_path = os.path.join(self.model_dir, checkpoint_name)
            assert chkpt_path in self.checkpoints, "Model {} can not found in {}.".format(checkpoint_name,
                                                                                          self.model_dir)
        self._create_model()
        self.model_.load_weights(chkpt_path)
        print('Model loaded successfully.')

    def _create_model(self):
        pass


class CPM(KerasModel):

    def __init__(self,
                 input_shape=(None, None, 3),
                 dropout_rate=0.1,
                 n_parts=16,
                 n_middle_stages=2,
                 model_name='convolutional_pose_machine',
                 models_dir='./models'):
        super(CPM, self).__init__(model_name, models_dir)
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.n_parts = n_parts
        self.n_middle_stages = n_middle_stages
        self.model_ = None

    def create_and_compile_model(self):
        self._create_model()
        self._compile_model()

    def train(self,
              train_gen,
              n_iter_train,
              val_gen,
              n_iter_val,
              epochs):
        dt = datetime.now().strftime('%Y%m%d-%H%M%S')

        checkpoint_path = os.path.join(self.model_dir,
                                       'weights-' + dt + '-{val_loss:.2f}.hdf5')
        self.checkpoints.append(checkpoint_path)

        log_dir = os.path.join(self.logs_dir, dt)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.logs.append(log_dir)

        checkpoint_callback = tfk.callbacks.ModelCheckpoint(checkpoint_path,
                                                            monitor='val_loss',
                                                            verbose=0,
                                                            save_best_only=True,
                                                            save_weights_only=True,
                                                            mode='auto',
                                                            period=1)
        tensorboard_callback = tfk.callbacks.TensorBoard(log_dir=log_dir,
                                                         histogram_freq=0,
                                                         write_graph=False,
                                                         write_grads=False,
                                                         write_images=True,
                                                         update_freq=500)
        reducelr_callback = tfk.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                            factor=0.5,
                                                            patience=1,
                                                            verbose=0,
                                                            mode='auto',
                                                            min_delta=0.00001,
                                                            cooldown=0,
                                                            min_lr=0.00001)
        callbacks = [checkpoint_callback, tensorboard_callback, reducelr_callback]
        history = self.model_.fit(train_gen,
                                  steps_per_epoch=n_iter_train,
                                  validation_data=val_gen,
                                  validation_steps=n_iter_val,
                                  epochs=epochs,
                                  callbacks=callbacks)
        return history

    def predict_visualize(self, input_image, variance=5):
        heatmaps = self.model_.predict([input_image])[0]
        max_ind = np.argwhere(heatmaps.max(axis=(0, 1)) == heatmaps)
        plt.imshow(input_image)
        img_height, img_width, _ = input_image.shape
        gaussian = self._make_gaussian(variance)
        hm = np.zeros((img_height, img_width), dtype=np.float32)
        for c_x, c_y, ch in max_ind:
            hm = self._add_gaussian(hm, gaussian, c_x, c_y, variance)
        plt.imshow(hm, alpha=0.6, cmap='viridis')
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
        return heatmap

    @staticmethod
    def _make_gaussian(variance):
        size = int(8 * variance)
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        return (np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / variance / variance)).astype(np.float32)

    def _create_model(self):
        input_image = tfk.Input(shape=self.input_shape)

        features = self._feature_extractor(input_image)
        belief_maps = self._cpm_first_stage(features)

        for i in range(self.n_middle_stages):
            belief_maps = self._cpm_middle_stage(features, belief_maps, str(i + 2))

        out = tfkl.Activation('sigmoid', name='final_heatmaps')(belief_maps)
        model = tfk.Model(input_image, out)
        self.model_ = model

    def _compile_model(self):
        assert self.model_ is not None, "Create model first."

        loss = tfk.losses.MSE
        optimizer = tfk.optimizers.Adam(lr=0.001)
        self.model_.compile(loss=loss, optimizer=optimizer)

    def _cpm_first_stage(self, features):
        y = self._conv2d(features, filters=128, kernel_size=(3, 3))
        y = self._conv2d(y, filters=128, kernel_size=(3, 3))
        x = self._add_skip_connection(features, y)
        x = self._conv2d(x, filters=64, kernel_size=(1, 1))
        x = tfkl.Conv2D(self.n_parts, (1, 1), padding='same', name='first_stage_heatmap')(x)
        return x

    def _cpm_middle_stage(self, features, former_believes, prefix):
        x = tfkl.concatenate([former_believes, features], axis=-1)
        y = self._conv2d(x, filters=128, kernel_size=(3, 3))
        y = self._conv2d(y, filters=128, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)

        y = self._conv2d(x, filters=128, kernel_size=(3, 3))
        y = self._conv2d(y, filters=128, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)

        y = self._conv2d(x, filters=128, kernel_size=(3, 3))
        y = self._conv2d(y, filters=128, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)

        x = self._conv2d(x, filters=64, kernel_size=(1, 1))

        belief_maps = tfkl.Conv2D(self.n_parts, (1, 1), padding='same', name='stage_{}_heatmap'.format(prefix))(x)
        return belief_maps

    def _feature_extractor(self, input_image):
        y = self._conv2d(input_image, filters=32, kernel_size=(3, 3))
        y = self._conv2d(y, filters=32, kernel_size=(3, 3))
        x = self._add_skip_connection(input_image, y)
        x = tfkl.MaxPooling2D((2, 2), strides=1, padding='same')(x)

        y = self._conv2d(x, filters=64, kernel_size=(3, 3))
        y = self._conv2d(y, filters=64, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)
        x = tfkl.MaxPooling2D((2, 2), strides=1, padding='same')(x)

        y = self._conv2d(x, filters=64, kernel_size=(3, 3))
        y = self._conv2d(y, filters=64, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)
        x = tfkl.MaxPooling2D((2, 2), strides=1, padding='same')(x)

        y = self._conv2d(x, filters=128, kernel_size=(2, 2))
        y = self._conv2d(y, filters=128, kernel_size=(2, 2))
        x = self._add_skip_connection(x, y)
        return x

    def _conv2d(self, x, filters, kernel_size):
        out = tfkl.Conv2D(filters, kernel_size, padding='same')(x)
        out = tfkl.BatchNormalization()(out)
        out = tfkl.Activation('relu')(out)
        out = tfkl.SpatialDropout2D(self.dropout_rate)(out)
        return out

    def _add_skip_connection(self, x, y, scale_factor=0.5):
        channels = tfkb.int_shape(y)[-1]
        shortcut_branch = tfkl.Conv2D(filters=channels, kernel_size=(1, 1), padding='same')(x)
        out = self._weighted_add(shortcut_branch, y, scale_factor)
        return tfkl.Activation('relu')(out)

    @staticmethod
    def _weighted_add(shortcut_branch, inception_branch, scale_factor):
        return tfkl.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                           arguments={'scale': scale_factor})([shortcut_branch, inception_branch])
