import numpy as np

from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """

            return parameter - self.lr * parameter_grad

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            updater.inertia = self.momentum * updater.inertia + self.lr * parameter_grad
            return parameter - updater.inertia

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        return np.maximum(inputs, 0)

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        return grad_outputs * (self.forward_inputs >= 0)


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        shift_inputs = np.exp(inputs - np.max(inputs, axis=1).reshape(-1, 1))
        return shift_inputs / np.sum(shift_inputs, axis=1).reshape(-1, 1)

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        softmax = self.forward_impl(self.forward_inputs)
        gm = softmax * grad_outputs
        return gm - np.diag(gm.sum(axis=1)) @ softmax


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, c)), output values

                n - batch size
                d - number of input units
                c - number of output units
        """
        return self.biases + inputs @ self.weights

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of input units
                c - number of output units
        """
        self.weights_grad = self.forward_inputs.T @ grad_outputs
        self.biases_grad = grad_outputs.sum(axis=0)
        return grad_outputs @ self.weights.T


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((1,)), mean Loss scalar for batch

                n - batch size
                d - number of units
        """
        return np.mean(- np.sum(y_gt * np.log(y_pred), axis=1)).reshape((1,))

    def gradient_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n, d)), dLoss/dY_pred

                n - batch size
                d - number of units
        """
        y_pred = np.maximum(y_pred, eps)
        return - y_gt / y_pred / y_pred.shape[0]


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # 1) Create a Model
    model = Model(loss=CategoricalCrossentropy(), optimizer=SGD(lr=1e-3))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(input_shape=(784,), units=128))
    model.add(ReLU())
    model.add(Dense(input_shape=128, units=32))
    model.add(ReLU())
    model.add(Dense(input_shape=32, units=10))
    model.add(Softmax(input_shape=10))

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, batch_size=32, epochs=7)

    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    n, d, ih, iw = inputs.shape
    assert kernels.shape[1] == d
    c, _, kh, kw = kernels.shape
    if padding > 0:
        padded_inputs = np.zeros([n, d, ih + 2 * padding, iw + 2 * padding])
        padded_inputs[:, :, padding:ih + padding, padding:iw + padding] = inputs
        inputs = padded_inputs
        n, d, ih, iw = inputs.shape

    oh, ow = ih - kh + 1, iw - kw + 1
    result = np.zeros([n, c, oh, ow])
    for i_kn in range(kh):
        for j_kn in range(kw):
            cur_inputs = inputs[:, :, i_kn:i_kn + oh, j_kn:j_kn + ow]  # (n, d, oh, ow)
            cur_inputs = np.transpose(cur_inputs, axes=(2, 3, 0, 1)).reshape([oh, ow, n, 1, d])  # (oh, ow, n, 1, d)
            # ! using convolve we need to flip kernels in both directions, i.e. take these indexes
            cur_kernel = kernels[:, :, kh - i_kn - 1, kw - j_kn - 1].reshape([c, d])  # (c, d, 1, 1) -> (c, d)
            cur_conv = cur_inputs * cur_kernel  # (oh, ow, n, c, d)
            assert cur_conv.shape == (oh, ow, n, c, d)
            result += np.transpose(np.sum(cur_conv, axis=-1), axes=(2, 3, 0, 1))

    return result


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, c, h, w)), output values

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        return convolve(inputs, self.kernels, padding=self.kernel_size // 2) + self.biases.reshape(1, -1, 1, 1)

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        n, c, h, w = grad_outputs.shape
        d = self.input_shape[0]
        assert self.kernels.shape[:2] == (c, d)
        _, _, hk, wk = self.kernels.shape

        grad_kernels = np.flip(np.transpose(self.kernels, axes=(1, 0, 2, 3)), axis=(2, 3))
        grads = convolve(grad_outputs, grad_kernels, padding=self.kernel_size // 2)
        assert grads.shape == (n, d, h, w)

        self.kernels_grad = convolve(
            np.transpose(grad_outputs, axes=(1, 0, 2, 3)),
            np.transpose(np.flip(self.forward_inputs, axis=(2, 3)), axes=(1, 0, 2, 3)),
            padding=self.kernel_size // 2
        )
        self.biases_grad = grad_outputs.sum(axis=(0, 2, 3))

        return grads


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, ih, iw)), input values

            :return: np.array((n, d, oh, ow)), output values

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        n, d, ih, iw = inputs.shape
        assert d == self.output_shape[0]
        _, oh, ow = self.output_shape

        ps = self.pool_size
        blocks = inputs.reshape([n, d, oh, ps, -1, ps]).swapaxes(3, 4)
        assert blocks.shape[-2:] == (self.pool_size, self.pool_size)

        if self.pool_mode == 'avg':
            return np.mean(blocks, axis=(-1, -2))
        elif self.pool_mode == 'max':
            blocks = blocks.reshape([-1, ps ** 2])
            self.forward_idxs = np.argmax(blocks, axis=-1)
            return blocks[np.arange(len(self.forward_idxs)), self.forward_idxs].reshape([n, d, oh, ow])
        else:
            raise RuntimeError(f"Unexpected pool mode {self.pool_mode}")

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

            :return: np.array((n, d, ih, iw)), dLoss/dInputs

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        n, d, oh, ow = grad_outputs.shape
        assert self.input_shape[0] == d
        _, ih, iw = self.input_shape
        ps = self.pool_size
        blocks = np.ones_like(self.forward_inputs).reshape([n, d, oh, ps, -1, ps]).swapaxes(3, 4)
        if self.pool_mode == 'avg':
            blocks = blocks * grad_outputs.reshape([n, d, oh, ow, 1, 1]) / ps ** 2
            return blocks.swapaxes(3, 4).reshape([n, d, ih, iw])
        elif self.pool_mode == 'max':
            blocks = blocks.reshape([-1, ps ** 2]) * 0
            blocks[np.arange(len(self.forward_idxs)), self.forward_idxs] = grad_outputs.reshape(-1)
            return blocks.reshape([n, d, oh, ow, ps, ps]).swapaxes(3, 4).reshape([n, d, ih, iw])
        else:
            raise RuntimeError(f"Unexpected pool mode {self.pool_mode}")


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

        self.mean = None
        self.input_mean = None
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        if not self.is_training:
            self.inv_sqrt_var = 1 / np.sqrt(self.running_var + eps)
            self.norm_input = (inputs - self.running_mean.reshape([1, -1, 1, 1])) * \
                              self.inv_sqrt_var.reshape([1, -1, 1, 1])
            return self.norm_input * self.gamma.reshape([1, -1, 1, 1]) + self.beta.reshape([1, -1, 1, 1])

        self.mean = inputs.mean(axis=(0, 2, 3))
        self.input_mean = inputs - self.mean.reshape([1, -1, 1, 1])
        self.var = np.mean(self.input_mean ** 2, axis=(0, 2, 3))
        self.sqrt_var = np.sqrt(self.var + eps)
        self.inv_sqrt_var = 1 / self.sqrt_var

        batch_size = inputs.shape[0]
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.var * batch_size / \
                           (batch_size - 1)
        self.norm_input = self.input_mean * self.inv_sqrt_var.reshape([1, -1, 1, 1])
        return self.norm_input * self.gamma.reshape([1, -1, 1, 1]) + self.beta.reshape([1, -1, 1, 1])

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        self.gamma_grad += (grad_outputs * self.norm_input).sum(axis=(0, 2, 3))
        self.beta_grad += grad_outputs.sum(axis=(0, 2, 3))

        grad_outputs = grad_outputs * self.gamma.reshape([1, -1, 1, 1])
        if not self.is_training:
            return grad_outputs * self.inv_sqrt_var.reshape([1, -1, 1, 1])
        n, d, h, w = self.forward_inputs.shape
        dt = (grad_outputs * self.input_mean).sum(axis=(0, 2, 3))
        dr = dt * (self.inv_sqrt_var ** 2) * (-1)
        dsig = dr * self.inv_sqrt_var / 2
        ds = np.ones_like(self.forward_inputs) * dsig.reshape([1, -1, 1, 1]) / (n * h * w)
        dz = 2 * self.input_mean * ds + grad_outputs * self.inv_sqrt_var.reshape([1, -1, 1, 1])
        dm = (dz * (-1)).sum(axis=(0, 2, 3))
        return dz + dm.reshape([1, -1, 1, 1]) / (n * h * w)


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (np.prod(self.input_shape),)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, (d * h * w))), output values

                n - batch size
                d - number of input channels
                (h, w) - image shape
        """
        return inputs.reshape([inputs.shape[0], -1])

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of units
                (h, w) - input image shape
        """
        return grad_outputs.reshape(self.forward_inputs.shape)


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        if self.is_training:
            self.forward_mask = np.random.uniform(size=inputs.shape) > self.p
            return inputs * self.forward_mask
        else:
            return (1 - self.p) * inputs

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        return self.forward_mask * grad_outputs


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(loss=CategoricalCrossentropy(), optimizer=SGDMomentum(lr=3e-3))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    FIRST_CONV = 16
    LAST_LINEAR = 64

    model.add(Conv2D(input_shape=(3, 32, 32), output_channels=FIRST_CONV))
    model.add(SkipConnection(layers=[
        BatchNorm(),
        Conv2D(output_channels=FIRST_CONV),
        ReLU(),
    ]))
    model.add(Pooling2D())  # (16, 16)

    model.add(Conv2D(output_channels=FIRST_CONV * 2))
    model.add(SkipConnection(layers=[
        BatchNorm(),
        Conv2D(output_channels=FIRST_CONV * 2),
        ReLU(),
    ]))
    model.add(Pooling2D())  # (8, 8)

    model.add(Conv2D(output_channels=FIRST_CONV * 4))
    model.add(SkipConnection(layers=[
        BatchNorm(),
        Conv2D(output_channels=FIRST_CONV * 4),
        ReLU(),
    ]))
    model.add(Pooling2D())  # (4, 4)

    model.add(Flatten())  # (64, 4, 4) -> (1024,)
    model.add(Dense(units=LAST_LINEAR * 2))
    model.add(ReLU())
    model.add(Dense(units=LAST_LINEAR))
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, batch_size=16, epochs=7)

    return model

# ============================================================================
