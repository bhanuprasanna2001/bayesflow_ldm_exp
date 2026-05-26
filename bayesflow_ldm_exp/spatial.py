import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras

from bayesflow.utils.serialization import serializable, serialize, deserialize


@serializable("ldm_exp")
class ConvDownsampler(keras.Layer):
    """Convolutional feature extractor for spatial latent encoders."""

    def __init__(
        self,
        widths=(32, 64),
        downsample_steps=1,
        activation="mish",
        kernel_size=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widths = tuple(widths)
        self.downsample_steps = int(downsample_steps)
        self.activation = activation
        self.kernel_size = int(kernel_size)
        self.layers_ = []

        for idx, width in enumerate(self.widths):
            self.layers_.extend(
                [
                    keras.layers.Conv2D(width, self.kernel_size, padding="same"),
                    keras.layers.Activation(self.activation),
                    keras.layers.Conv2D(width, self.kernel_size, padding="same"),
                    keras.layers.Activation(self.activation),
                ]
            )
            if idx < self.downsample_steps:
                self.layers_.append(keras.layers.AveragePooling2D(pool_size=2, strides=2))

    def call(self, x, training=False):
        for layer in self.layers_:
            x = layer(x, training=training)
        return x

    def build(self, input_shape):
        if self.built:
            return

        shape = input_shape
        for layer in self.layers_:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        shape = input_shape
        for layer in self.layers_:
            shape = layer.compute_output_shape(shape)
        return shape

    def get_config(self):
        base = super().get_config()
        return base | serialize(
            {
                "widths": self.widths,
                "downsample_steps": self.downsample_steps,
                "activation": self.activation,
                "kernel_size": self.kernel_size,
            }
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))


@serializable("ldm_exp")
class ConvUpsampler(keras.Layer):
    """Convolutional feature decoder for spatial latent models."""

    def __init__(
        self,
        widths=(64, 32),
        upsample_steps=1,
        activation="mish",
        kernel_size=3,
        interpolation="nearest",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widths = tuple(widths)
        self.upsample_steps = int(upsample_steps)
        self.activation = activation
        self.kernel_size = int(kernel_size)
        self.interpolation = interpolation
        self.layers_ = []

        for idx, width in enumerate(self.widths):
            self.layers_.extend(
                [
                    keras.layers.Conv2D(width, self.kernel_size, padding="same"),
                    keras.layers.Activation(self.activation),
                    keras.layers.Conv2D(width, self.kernel_size, padding="same"),
                    keras.layers.Activation(self.activation),
                ]
            )
            if idx < self.upsample_steps:
                self.layers_.append(keras.layers.UpSampling2D(size=2, interpolation=self.interpolation))

    def call(self, z, training=False):
        for layer in self.layers_:
            z = layer(z, training=training)
        return z

    def build(self, input_shape):
        if self.built:
            return

        shape = input_shape
        for layer in self.layers_:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        shape = input_shape
        for layer in self.layers_:
            shape = layer.compute_output_shape(shape)
        return shape

    def get_config(self):
        base = super().get_config()
        return base | serialize(
            {
                "widths": self.widths,
                "upsample_steps": self.upsample_steps,
                "activation": self.activation,
                "kernel_size": self.kernel_size,
                "interpolation": self.interpolation,
            }
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))


@serializable("ldm_exp")
class ConditionResizeUNet(keras.Layer):
    """U-Net wrapper that resizes spatial conditions to the target grid."""

    def __init__(self, unet_kwargs=None, interpolation="bilinear", **kwargs):
        super().__init__(**kwargs)
        import bayesflow as bf

        self.unet_kwargs = dict(unet_kwargs or {})
        self.interpolation = interpolation
        self.unet = bf.networks.UNet(**self.unet_kwargs)
        self.resizer = None

    def build(self, input_shape):
        if self.built:
            return

        x_shape, t_shape, cond_shape = input_shape
        if cond_shape is not None and tuple(cond_shape[1:3]) != tuple(x_shape[1:3]):
            self.resizer = keras.layers.Resizing(
                height=int(x_shape[1]),
                width=int(x_shape[2]),
                interpolation=self.interpolation,
            )
            cond_shape = (cond_shape[0], x_shape[1], x_shape[2], cond_shape[-1])

        self.unet.build((x_shape, t_shape, cond_shape))
        super().build(input_shape)

    def call(self, inputs, training=False):
        x, t, conditions = inputs
        if conditions is not None and self.resizer is not None:
            conditions = self.resizer(conditions, training=training)
        return self.unet((x, t, conditions), training=training)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0])

    def get_config(self):
        base = super().get_config()
        return base | serialize({"unet_kwargs": self.unet_kwargs, "interpolation": self.interpolation})

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
