import numpy as np
import pytensor.tensor as pt

from pytensor.scan import scan

from pytensor_ml.activations import Activation, Tanh
from pytensor_ml.base import Layer
from pytensor_ml.params import trainable
from pytensor_ml.state import (
    Initializer,
    OrthogonalInitializer,
    XavierNormalInitializer,
    ZeroInitializer,
)


class RNN(Layer):
    r"""
    Elman recurrent layer over a sequence.

    Carry a hidden state along the time axis, updating it at each step from the step's input and the
    previous state:

    .. math::

        h_t = \phi\left(x_t W_{ih} + b + h_{t-1} W_{hh}\right),

    where :math:`\phi` is the activation.

    Time is the second-to-last axis and everything before it is a batch axis, so the input is
    ``(..., time, n_in)`` and the output ``(..., time, n_hidden)``, one hidden state per step. Slice the
    last step off the result -- ``out[..., -1, :]`` -- for the sequence-classification case; pytensor's
    ``scan_save_mem`` rewrite sees that the earlier steps are unused and stops storing them.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to "RNN" when None.
    n_in : int
        Size of the input feature axis.
    n_hidden : int
        Size of the hidden state.
    activation : Activation, optional
        Applied to each step's pre-activation. Default is :class:`~pytensor_ml.activations.Tanh`, which
        bounds the state and so keeps the recurrence from running away over a long sequence.
    bias : bool, optional
        Add the learned shift :math:`b`. One bias covers the step, rather than torch's separate
        ``b_ih`` and ``b_hh``, whose sum is the only thing the step can distinguish. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W_{ih}` is drawn. Xavier normal when omitted.
    recurrent_initializer : Initializer, optional
        How :math:`W_{hh}` is drawn. It meets the state once per step, so its singular values compound
        over the sequence: at one the state's norm survives any length, and spread around one the
        gradient explodes along some directions while vanishing along others. Orthogonal when omitted,
        as in keras and flax.
    bias_initializer : Initializer, optional
        How :math:`b` is drawn. Zeros when omitted.
    """

    def __init__(
        self,
        name: str | None,
        n_in: int,
        n_hidden: int,
        activation: Activation | None = None,
        bias: bool = True,
        *,
        weight_initializer: Initializer | None = None,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
    ):
        self.name = name if name else "RNN"
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.activation = activation if activation is not None else Tanh()
        self.bias = bias

        # Held directly rather than as a nested Linear: the projection runs inside the recurrence, and a
        # layer op there would bury its matmul in an inner graph where the scan rewrites cannot see it.
        W_ih_initializer = (
            XavierNormalInitializer() if weight_initializer is None else weight_initializer
        )
        self.W_ih = trainable(
            W_ih_initializer.initial_value((n_in, n_hidden)),
            f"{self.name}_W_ih",
            initializer=W_ih_initializer,
        )

        if bias:
            b_initializer = ZeroInitializer() if bias_initializer is None else bias_initializer
            self.b = trainable(
                b_initializer.initial_value((n_hidden,)),
                f"{self.name}_b",
                initializer=b_initializer,
            )

        W_hh_initializer = (
            OrthogonalInitializer() if recurrent_initializer is None else recurrent_initializer
        )
        self.W_hh = trainable(
            W_hh_initializer.initial_value((n_hidden, n_hidden)),
            f"{self.name}_W_hh",
            initializer=W_hh_initializer,
        )

    def __call__(
        self, X: pt.TensorLike, initial_state: pt.TensorLike | None = None
    ) -> pt.TensorVariable:
        """
        Run the recurrence over ``X`` and return every hidden state.

        Parameters
        ----------
        X : TensorVariable
            Input sequence, shape ``(..., time, n_in)``.
        initial_state : TensorVariable, optional
            The state the recurrence starts from, shape ``(..., n_hidden)``, over the same batch axes.
            Zeros when omitted.

        Returns
        -------
        TensorVariable
            Hidden states, shape ``(..., time, n_hidden)``.
        """
        X = pt.as_tensor(X)
        if X.ndim < 2:
            raise ValueError(
                f"{self.name} takes a sequence of shape (..., time, {self.n_in}), but got a "
                f"{X.ndim}-dimensional input, which has no time axis to recur over."
            )

        # Scan iterates the leading axis, so time moves to the front and back again on the way out.
        sequence = pt.moveaxis(X, -2, 0)

        if initial_state is None:
            # The state carries whatever the step produces, which promotes the input against the
            # parameters: a float64 sequence through a float32 network makes a float64 state.
            state_dtype = np.result_type(X.dtype, self.W_ih.dtype, self.W_hh.dtype)
            initial_state = pt.zeros((*X.shape[:-2], self.n_hidden), dtype=state_dtype)
        else:
            initial_state = pt.as_tensor(initial_state)
            if initial_state.ndim != X.ndim - 1:
                raise ValueError(
                    f"{self.name} starts from a state of shape (..., {self.n_hidden}), carrying the same "
                    f"batch axes as its input, so a {X.ndim}-dimensional input needs a "
                    f"{X.ndim - 1}-dimensional state; got a {initial_state.ndim}-dimensional one."
                )

        def step(x_t, h_prev):
            pre_activation = x_t @ self.W_ih + h_prev @ self.W_hh
            if self.bias:
                pre_activation = pre_activation + self.b
            return self.activation(pre_activation)

        # Not strict: the step closes over the layer's parameters and the activation's, and scan lifts
        # them in. A generator captured that way has no update, which `collect_default_updates` refuses.
        hidden = scan(
            step,
            sequences=[sequence],
            outputs_info=[initial_state],
            name=f"{self.name}_recurrence",
            return_updates=False,
        )

        out = pt.moveaxis(hidden, 0, -2)
        out.name = f"{self.name}_output"
        return out


__all__ = ["RNN"]
