from tensorflow import keras as tfk
tfkl = tfk.layers
tfkb = tfk.backend


class CPM:
    def __init__(self, input_shape=(None, None, 3), dropout_rate=0.1, n_parts=16, n_middle_stages=2):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.n_parts = n_parts
        self.n_middle_stages = n_middle_stages

    def create_model(self):
        input_image = tfk.Input(shape=self.input_shape)

        features = self._feature_extractor(input_image)
        belief_maps = self._cpm_first_stage(features)

        for i in range(self.n_middle_stages):
            belief_maps = self._cpm_middle_stage(features, belief_maps, str(i + 2))

        out = tfkl.Activation('sigmoid', name='final_heatmaps')(belief_maps)
        model = tfk.Model(input_image, out)
        return model

    def _cpm_first_stage(self, features):
        # x = self._conv2d(features, filters=512, kernel_size=(1, 1))
        x = self._conv2d(features, filters=256, kernel_size=(3, 3))
        x = self._conv2d(x, filters=256, kernel_size=(3, 3))
        x = self._conv2d(x, filters=128, kernel_size=(1, 1))
        x = tfkl.Conv2D(self.n_parts, (1, 1), padding='same', name='first_stage_heatmap')(x)
        return x

    def _cpm_middle_stage(self, features, former_believes, prefix):
        x = tfkl.concatenate([former_believes, features], axis=-1)
        y = self._conv2d(x, filters=128, kernel_size=(2, 2))
        y = self._conv2d(y, filters=128, kernel_size=(2, 2))
        y = self._conv2d(y, filters=128, kernel_size=(2, 2))
        x = self._add_skip_connection(x, y)

        y = self._conv2d(x, filters=128, kernel_size=(2, 2))
        y = self._conv2d(y, filters=128, kernel_size=(2, 2))
        y = self._conv2d(y, filters=128, kernel_size=(2, 2))
        x = self._add_skip_connection(x, y)

        y = self._conv2d(x, filters=128, kernel_size=(2, 2))
        y = self._conv2d(y, filters=128, kernel_size=(2, 2))
        y = self._conv2d(y, filters=128, kernel_size=(2, 2))
        x = self._add_skip_connection(x, y)

        x = self._conv2d(x, filters=64, kernel_size=(1, 1))

        belief_maps = tfkl.Conv2D(self.n_parts, (1, 1), padding='same', name='stage_{}_heatmap'.format(prefix))(x)
        return belief_maps

    def _feature_extractor(self, input_image):
        y = self._conv2d(input_image, filters=64, kernel_size=(3, 3))
        y = self._conv2d(y, filters=64, kernel_size=(3, 3))
        x = self._add_skip_connection(input_image, y)
        x = tfkl.MaxPooling2D((2, 2), strides=1, padding='same')(x)

        y = self._conv2d(x, filters=128, kernel_size=(3, 3))
        y = self._conv2d(y, filters=128, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)
        x = tfkl.MaxPooling2D((2, 2), strides=1, padding='same')(x)

        y = self._conv2d(x, filters=128, kernel_size=(3, 3))
        y = self._conv2d(y, filters=128, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)
        x = tfkl.MaxPooling2D((2, 2), strides=1, padding='same')(x)

        y = self._conv2d(x, filters=256, kernel_size=(2, 2))
        y = self._conv2d(y, filters=256, kernel_size=(2, 2))
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
