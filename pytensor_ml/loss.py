from abc import ABC, abstractmethod
from typing import Literal

import pytensor.tensor as pt

from pytensor.tensor.basic import as_tensor_variable

Reductions = Literal["mean", "sum"]
reduction_dict = {"mean": pt.mean, "sum": pt.sum}


class Loss(ABC):

    @abstractmethod
    def loss(self, y_pred: pt.TensorVariable) -> pt.TensorVariable: ...

    def __call__(self, y_pred: pt.TensorVariable) -> pt.TensorVariable:
        return self.loss(y_pred)


class SquaredError(Loss):
    def __init__(self, reduction: Reductions = "mean"):
        self.reduction = reduction_dict[reduction]

    def loss(self, y_pred: pt.TensorVariable) -> pt.TensorVariable:
        y_true = y_pred.type(name='y_true')
        return self.reduction((y_true - y_pred) ** 2)


class CrossEntropy(Loss):
    def __init__(
        self,
        reduction: Reductions = "mean",
        expect_logits: bool = False,
        expect_onehot_labels: bool = False,
    ):
        """
        Cross-entropy loss function.

        Parameters
        ----------
        reduction: str, one of "sum" or "mean"
            How to aggregate loss across samples.
        expect_logits: bool
            Whether the function will expect logits as `y_pred`. If False, it expects expects probabilities.
        expect_onehot_labels: bool
            Whether the function will expect one-hot encoded labels as `y_true`. If False, it expects class indices.
            This argument changes the expected shape of y_true. If ``expect_onehot_labels`` is True, then y_true
            should be a tensor of shape (*batch_size, n_classes). Otherwise, it should be a tensor of shape
            (*batch_size,).
        """
        self.reduction = reduction_dict[reduction]
        self.expect_logits = expect_logits
        self.expect_onehot_labels = expect_onehot_labels

    def loss(self, y_pred: pt.TensorVariable) -> pt.TensorVariable:
        """

        Parameters
        ----------
        y_pred: Tensor variable
            Matrix of predicted values for each class. Should have shape (*batch_size, n_classes). The expected values
            depend on the `expect_logits`. If `expect_logits` is True, then `y_pred` should contain logits, otherwise
            it should contain probabilities (the last dimension should sum to 1)

        Returns
        -------
        loss: Tensor variable
            Scalar loss value
        """
        y_pred = as_tensor_variable(y_pred).type(name='y_pred')
        if y_pred.ndim < 2:
            raise ValueError("y_pred should have at least 2 dimensions, got {}".format(y_pred.ndim))

        # y_true will be either (*batch_size, n_classes) if expect_onehot_labels is True, or (*batch_size,) if False
        y_true = pt.tensor('y_true',
                           shape=y_pred.type.shape if self.expect_onehot_labels else y_pred.type.shape[:-1],
                           dtype=y_pred.type.dtype if self.expect_onehot_labels else 'int64')

        if self.expect_logits:
            log_softmax = pt.special.log_softmax(y_pred, axis=-1)
        else:
            log_softmax = pt.log(y_pred)

        if self.expect_onehot_labels:
            loss = -self.reduction((y_true * log_softmax).sum(axis=-1))
        else:
            log_softmax = pt.take_along_axis(log_softmax, y_true[..., None], axis=-1)
            loss = -self.reduction(log_softmax)

        return loss
