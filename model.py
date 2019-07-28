from tensorflow import keras as tfk
tfkl = tfk.layers
tfkb = tfk.backend


class CPM:
    def __init__(self, input_shape=(None, None, 3), dropout_rate=0.1, n_parts=16):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.n_parts = n_parts

    def create_model(self):
        input_image = tfk.Input(shape=self.input_shape)

        features = self._feature_extractor(input_image)
        first_stage_believes = self._cpm_first_stage(input_image)

        second_stage_believes = self._cpm_second_stage(features, first_stage_believes, prefix='stage2_')

        third_stage_believes = self._cpm_second_stage(features, second_stage_believes, prefix='stage3_')

        fourth_stage_believes = self._cpm_second_stage(features, third_stage_believes, prefix='stage4_')

        out = tfkl.Activation('sigmoid', name='final_heatmaps')(fourth_stage_believes)

        model = tfk.Model(input_image, out)
        return model

    def _cpm_first_stage(self, input_image):
        y = self._conv2d(input_image, filters=16, kernel_size=(3, 3))
        y = self._conv2d(y, filters=16, kernel_size=(3, 3))
        x = self._add_skip_connection(input_image, y)
        x = tfkl.MaxPooling2D(2, strides=2, padding='same')(x)

        y = self._conv2d(x, filters=32, kernel_size=(3, 3))
        y = self._conv2d(y, filters=32, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)
        x = tfkl.MaxPooling2D(2, strides=2, padding='same')(x)

        y = self._conv2d(x, filters=64, kernel_size=(3, 3))
        y = self._conv2d(y, filters=64, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)
        x = tfkl.MaxPooling2D(2, strides=2, padding='same')(x)

        y = self._conv2d(x, filters=128, kernel_size=(2, 2))
        y = self._conv2d(y, filters=128, kernel_size=(2, 2))
        x = self._add_skip_connection(x, y)

        y = self._conv2d(x, filters=256, kernel_size=(3, 3))
        y = self._conv2d(y, filters=256, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)

        x = self._conv2d(x, 64, kernel_size=(1, 1))

        out = tfkl.Conv2D(self.n_parts, (1, 1), padding='same', activation=None, name='stage1_repr')(x)
        out = tfkl.BatchNormalization()(out)
        out = tfkl.Activation('relu')(out)
        out = tfkl.SpatialDropout2D(self.dropout_rate)(out)
        return out

    def _cpm_second_stage(self, extracted_features, former_believes, prefix):
        input_tensor = tfkl.concatenate([extracted_features, former_believes],
                                        axis=-1, name=prefix + 'concat')

        y = self._conv2d(input_tensor, filters=64, kernel_size=(4, 4))
        y = self._conv2d(y, filters=64, kernel_size=(4, 4))
        x = self._add_skip_connection(input_tensor, y)

        y = self._conv2d(x, filters=128, kernel_size=(4, 4))
        y = self._conv2d(y, filters=128, kernel_size=(4, 4))
        x = self._add_skip_connection(x, y)

        y = self._conv2d(x, filters=256, kernel_size=(3, 3))
        y = self._conv2d(y, filters=256, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)

        x = self._conv2d(x, 64, kernel_size=(1, 1))
        out = tfkl.Conv2D(self.n_parts, (1, 1), padding='same', name=prefix + 'repr')(x)
        return out

    def _feature_extractor(self, input_image):
        y = self._conv2d(input_image, filters=16, kernel_size=(3, 3))
        y = self._conv2d(y, filters=16, kernel_size=(3, 3))
        x = self._add_skip_connection(input_image, y)
        x = tfkl.MaxPooling2D(2, strides=2, padding='same')(x)

        y = self._conv2d(x, filters=32, kernel_size=(3, 3))
        y = self._conv2d(y, filters=32, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)
        x = tfkl.MaxPooling2D(2, strides=2, padding='same')(x)

        y = self._conv2d(x, filters=64, kernel_size=(3, 3))
        y = self._conv2d(y, filters=64, kernel_size=(3, 3))
        x = self._add_skip_connection(x, y)
        x = tfkl.MaxPooling2D(2, strides=2, padding='same')(x)

        y = self._conv2d(x, filters=128, kernel_size=(2, 2))
        y = self._conv2d(y, filters=128, kernel_size=(2, 2))
        x = self._add_skip_connection(x, y)

        x = tfkl.Conv2D(self.n_parts, (1, 1), padding='same')(x)
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
