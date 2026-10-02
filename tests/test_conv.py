import warnings

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.compile.builders import OpFromGraph
from pytensor.gradient import verify_grad
from pytensor.graph import rewrite_graph
from pytensor.graph.traversal import apply_ancestors
from scipy.signal import correlate

from pytensor_ml.activations import ReLU
from pytensor_ml.layers import BatchNorm, Conv1D, Conv2D, Flatten, Input, Linear, MaxPool2D
from pytensor_ml.layers.conv import Col2Im, ConvLayer, ConvLayerGrad, Im2Col
from pytensor_ml.loss import SquaredError
from pytensor_ml.model import Model
from pytensor_ml.optim import adam
from pytensor_ml.pytensorf import collect_trainable_params
from pytensor_ml.state import OneInitializer, ZeroInitializer

floatX = pytensor.config.floatX

# The reference sums channel contributions in a different order than the Dot, so the gap tracks
# the precision.
ATOL = 1e-6 if floatX == "float64" else 1e-4


@pytest.fixture
def rng():
    return np.random.default_rng(sum(map(ord, "pytensor_ml conv")))


def correlate_reference(X_np, W_np, stride=1, dilation=1):
    """A convolution written with scipy, one channel pair at a time, as the independent reference.

    ``scipy.signal.correlate`` is not another path through pytensor, so it cannot agree with a bug the
    implementation and a pytensor-based reference would share. It has no notion of stride or dilation,
    so dilation is spelled by zero-stuffing the kernel and stride by subsampling the result -- both
    definitions rather than reimplementations of what the layer does. ``stride`` and ``dilation`` each
    take a scalar or one value per spatial axis.
    """
    *kernel, in_channels, out_channels = W_np.shape
    strides = (stride,) * len(kernel) if isinstance(stride, int) else tuple(stride)
    dilations = (dilation,) * len(kernel) if isinstance(dilation, int) else tuple(dilation)
    spans = tuple(spacing * (extent - 1) + 1 for extent, spacing in zip(kernel, dilations))
    if any(spacing != 1 for spacing in dilations):
        stuffed = np.zeros((*spans, in_channels, out_channels), dtype=W_np.dtype)
        stuffed[
            (*(slice(None, None, spacing) for spacing in dilations), slice(None), slice(None))
        ] = W_np
        W_np = stuffed
    outputs = []
    for image in X_np:
        planes = []
        for out_channel in range(out_channels):
            total = None
            for in_channel in range(in_channels):
                term = correlate(
                    image[..., in_channel], W_np[..., in_channel, out_channel], mode="valid"
                )
                total = term if total is None else total + term
            planes.append(total)
        outputs.append(np.stack(planes, axis=-1))
    assert outputs[0].ndim == len(kernel) + 1
    stacked = np.stack(outputs)
    return stacked[(slice(None), *(slice(None, None, step) for step in strides), slice(None))]


def test_the_conv_op_correlates_like_scipy(rng):
    """Every channel pair, summed over input channels, against scipy's correlation rather than against
    another pytensor graph. The layers check ranks 1 and 2, so the op is checked at rank 3, which is
    also the regression: a dispatch that handles some ranks and refuses the rest does not fall back, it
    fails to compile."""
    in_channels, out_channels = 3, 4
    X = pt.tensor("X", shape=(None, None, None, None, in_channels))
    W = pt.tensor("W", shape=(2, 2, 2, in_channels, out_channels))
    op = ConvLayer(kernel_size=(2, 2, 2), stride=(1, 1, 1), dilation=(1, 1, 1))

    X_np = rng.normal(size=(2, 6, 6, 6, in_channels)).astype(floatX)
    W_np = rng.normal(size=(2, 2, 2, in_channels, out_channels)).astype(floatX)

    np.testing.assert_allclose(
        op(X, W).eval({X: X_np, W: W_np}), correlate_reference(X_np, W_np), atol=ATOL
    )


def test_the_conv_op_takes_a_gradient_matching_finite_differences(rng):
    """Overlapping windows make the gather's pullback a scatter-add, and the bias's is a sum over
    everything but the channel axis. Checked against finite differences for all three inputs, at a
    stride and a dilation that differ per axis, with spatial extents known only at runtime."""
    op = ConvLayer(kernel_size=(2, 2), stride=(2, 1), dilation=(1, 2))
    inputs = [
        rng.normal(size=(2, 7, 7, 2)),
        rng.normal(size=(2, 2, 2, 3)),
        rng.normal(size=(3,)),
    ]

    with pytensor.config.change_flags(floatX="float64"):
        verify_grad(op, inputs, rng=np.random.default_rng(0))


def test_conv1d_pads_with_the_mode_it_is_given():
    """A padding mode reaches `pt.pad`, so an edge-padded input repeats its boundary rather than
    fading to zero. With a summing kernel over a constant input, zero padding would leave each edge
    window short by one, and edge padding leaves none short."""
    X = pt.tensor("X", shape=(None, None, 1))
    X_np = np.ones((1, 6, 1), dtype=floatX)

    edge_padded = Conv1D(
        "c",
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        padding="same",
        padding_mode="edge",
        bias=False,
    )
    edge_padded.W.set_value(np.ones((3, 1, 1), dtype=floatX))

    np.testing.assert_allclose(
        edge_padded(X).eval({X: X_np})[0, :, 0], [3, 3, 3, 3, 3, 3], atol=ATOL
    )


def test_conv1d_bias_is_optional():
    """Dropping the bias drops the parameter as well as the term; an unused one would hand the
    optimizer moment state to carry for a weight that never moves."""
    layer = Conv1D("conv", in_channels=2, out_channels=3, kernel_size=2, bias=False)
    out = layer(pt.tensor("X", shape=(None, None, 2)))

    assert set(collect_trainable_params(out)) == {layer.W}


def test_conv1d_stores_its_kernel_in_the_layout_fans_reads():
    """`fans` reads the trailing two axes of a kernel as its channels and the rest as the receptive
    field, so a kernel stored in any other layout would be drawn at a scale set by the wrong fans."""
    layer = Conv1D("conv", in_channels=8, out_channels=16, kernel_size=5)

    assert layer.W.get_value().shape == (5, 8, 16)


def test_conv1d_forwards_its_initializers_to_its_parameters():
    """Two keyword-only arguments, and one dropped on the floor leaves a parameter silently at its
    default draw."""
    layer = Conv1D(
        "conv",
        in_channels=2,
        out_channels=3,
        kernel_size=2,
        weight_initializer=ZeroInitializer(),
        bias_initializer=OneInitializer(),
    )

    np.testing.assert_array_equal(layer.W.get_value(), np.zeros((2, 2, 3)))
    np.testing.assert_array_equal(layer.b.get_value(), np.ones(3))


def test_conv1d_rejects_an_input_of_the_wrong_rank():
    """A 1-D convolution takes (batch, time, channels); handing it a bare sequence is the natural
    mistake and the op would report it from inside the gather."""
    layer = Conv1D("conv", in_channels=2, out_channels=3, kernel_size=2)

    with pytest.raises(ValueError, match="needs a 3-dimensional input; got a 2-dimensional one"):
        layer(pt.tensor("X", shape=(None, 2)))


def test_conv1d_rejects_a_kernel_wider_than_the_input():
    """A window that does not fit yields no windows at all, and every downstream shape then carries a
    zero axis, so the graph computes an empty answer instead of failing. Caught wherever the input's
    length is known when the graph is built."""
    layer = Conv1D("conv", in_channels=1, out_channels=1, kernel_size=10)

    with pytest.raises(ValueError, match="at least 10 elements along spatial axis 0"):
        layer(pt.tensor("X", shape=(None, 5, 1)))

    # Dilation stretches the span, so a kernel that fits undilated need not fit dilated.
    dilated = Conv1D("conv", in_channels=1, out_channels=1, kernel_size=3, dilation=4)
    with pytest.raises(ValueError, match="at least 9 elements"):
        dilated(pt.tensor("X", shape=(None, 8, 1)))

    # Padding is counted, so the same kernel fits once there is enough of it.
    padded = Conv1D("conv", in_channels=1, out_channels=1, kernel_size=10, padding=3)
    assert padded(pt.tensor("X", shape=(None, 5, 1))).type.shape == (None, 2, 1)


def test_conv1d_rejects_negative_padding():
    """Padding adds elements. A negative amount reaches `pt.pad` as nonsense rather than quietly
    trimming, and the caller almost certainly meant to slice the input."""
    with pytest.raises(ValueError, match="cannot be negative"):
        Conv1D("conv", in_channels=1, out_channels=1, kernel_size=3, padding=-1)


@pytest.mark.parametrize(
    "length, stride", [(10, 3), (11, 2), (11, 3)], ids=["10_stride3", "11_stride2", "11_stride3"]
)
def test_conv1d_same_padding_holds_at_any_stride(length, stride):
    """`same` is a claim about the output length, and torch and keras both define it as
    ceil(input / stride) rather than only holding at unit stride. Only a stride that does not divide
    the length separates the ceiling from the floor."""
    layer = Conv1D(
        "conv", in_channels=1, out_channels=1, kernel_size=3, stride=stride, padding="same"
    )

    out = layer(pt.tensor("X", shape=(None, length, 1)))
    assert out.type.shape == (None, -(-length // stride), 1)


def test_conv1d_strides_and_dilates_like_scipy(rng):
    """Stride and dilation have to survive the trip from the layer's constructor through the op to the
    gather. They differ here, so a layer that swapped the two, or dropped either, would disagree."""
    stride, dilation = 2, 3
    X = pt.tensor("X", shape=(None, None, 3))
    layer = Conv1D(
        "conv", in_channels=3, out_channels=4, kernel_size=3, stride=stride, dilation=dilation
    )

    W_np = rng.normal(size=(3, 3, 4)).astype(floatX)
    b_np = rng.normal(size=(4,)).astype(floatX)
    layer.W.set_value(W_np)
    layer.b.set_value(b_np)
    X_np = rng.normal(size=(2, 20, 3)).astype(floatX)

    np.testing.assert_allclose(
        layer(X).eval({X: X_np}),
        correlate_reference(X_np, W_np, stride=stride, dilation=dilation) + b_np,
        atol=ATOL,
    )


def test_conv1d_rejects_a_per_axis_argument_of_the_wrong_length():
    """`kernel_size`, `stride` and `dilation` each take a scalar or one value per spatial axis, and
    handing a 1-D convolution a pair is the natural mistake when moving code over from 2-D."""
    with pytest.raises(
        ValueError, match="kernel_size must be an int or one value per spatial axis"
    ):
        Conv1D("conv", in_channels=1, out_channels=1, kernel_size=(3, 3))

    with pytest.raises(ValueError, match="stride must be an int"):
        Conv1D("conv", in_channels=1, out_channels=1, kernel_size=3, stride=(1, 1))


@pytest.mark.parametrize(
    "spatial, kernel_size, stride, dilation",
    [((12,), (3,), (5,), (1,)), ((6, 6, 6), (2, 2, 2), (1, 1, 1), (1, 1, 1))],
    ids=["1d_untouched_tail", "3d"],
)
def test_col2im_is_the_adjoint_of_im2col(spatial, kernel_size, stride, dilation, rng):
    """A scatter-add is the adjoint of the gather it reverses, so ``<Im2Col(X), P> == <X, Col2Im(P)>``
    for any ``X`` and ``P``. The untouched-tail case is the one a scatter can get wrong on its own
    terms: a stride that overshoots leaves positions no window reaches, and those have to stay zero
    rather than pick up a neighbor."""
    X = pt.tensor("X", shape=(2, *spatial, 3))
    gathered = Im2Col(kernel_size, stride, dilation)(X)
    patches = pt.tensor("patches", shape=gathered.type.shape)
    scattered = Col2Im(kernel_size, stride, dilation)(patches, *spatial)

    X_np = rng.normal(size=X.type.shape).astype(floatX)
    patches_np = rng.normal(size=patches.type.shape).astype(floatX)
    inner_products = pytensor.function(
        [X, patches], [(gathered * patches).sum(), (X * scattered).sum()]
    )

    np.testing.assert_allclose(*inner_products(X_np, patches_np), rtol=ATOL)


def test_col2im_keeps_a_spatial_extent_it_is_given_statically():
    """The extents arrive as inputs rather than as props, so a known one has to survive as a static
    shape -- otherwise every backward pass loses the shape its forward had."""
    cotangent = pt.tensor("cotangent", shape=(2, 9, 3, 3))
    length = pt.scalar("length", dtype="int64")

    assert Col2Im((3,), (1,), (1,))(cotangent, 11).type.shape == (2, 11, 3)
    assert Col2Im((3,), (1,), (1,))(cotangent, length).type.shape == (2, None, 3)


def test_col2im_gathers_the_cotangent_it_scattered(rng):
    """A scatter-add's pullback is the gather that reverses it, so seeding the output with any
    cotangent has to come back as that cotangent gathered into the windows that reached it."""
    patches = pt.tensor("patches", shape=(2, 9, 3, 3))
    scattered = Col2Im((3,), (1,), (1,))(patches, 11)
    seed = pt.tensor("seed", shape=(2, 11, 3))
    seed_np = rng.normal(size=(2, 11, 3)).astype(floatX)

    pulled_back = pt.grad(cost=None, wrt=patches, known_grads={scattered: seed})
    gathered = Im2Col((3,), (1,), (1,))(seed)

    np.testing.assert_allclose(
        pulled_back.eval({seed: seed_np}), gathered.eval({seed: seed_np}), atol=ATOL
    )


def test_im2col_accepts_an_input_it_cannot_assume_is_contiguous(rng):
    """`TensorType` carries no layout, so a kernel is typed against any layout whatever the data turns
    out to be. One that quietly needs a contiguous buffer does not fall back to `perform` when it does
    not get one -- it fails to compile, for every caller."""
    X_np = rng.normal(size=(2, 3, 11)).astype(floatX)
    X = pt.tensor("X", shape=(2, 3, 11))
    contiguous = pt.tensor("contiguous", shape=(2, 11, 3))
    im2col = Im2Col((3,), (1,), (1,))

    got = pytensor.function([X], im2col(X[:, :, ::-1].transpose(0, 2, 1)))(X_np)
    reference = pytensor.function([contiguous], im2col(contiguous))(
        np.ascontiguousarray(X_np[:, :, ::-1].transpose(0, 2, 1))
    )

    np.testing.assert_allclose(got, reference, atol=ATOL)


def test_the_gather_and_scatter_run_the_same_in_python_as_compiled(rng):
    """`perform` is what runs wherever no backend dispatches the op, and the compiled default never
    reaches it, so the python path is checked against the compiled one. Kernel, stride and dilation
    each differ per axis, so a `perform` that swapped any of them across axes would disagree."""
    geometry = ((2, 3), (2, 1), (1, 2))
    X = pt.tensor("X", shape=(None, None, None, 3))
    patches = pt.tensor("patches", shape=(None, None, None, 2, 3, 3))
    height = pt.scalar("height", dtype="int64")
    width = pt.scalar("width", dtype="int64")
    inputs = [X, patches, height, width]
    outputs = [Im2Col(*geometry)(X), Col2Im(*geometry)(patches, height, width)]

    values = (
        rng.normal(size=(2, 7, 8, 3)).astype(floatX),
        rng.normal(size=(2, 3, 4, 2, 3, 3)).astype(floatX),
        7,
        8,
    )
    in_python = pytensor.function(inputs, outputs, mode="FAST_COMPILE")(*values)
    compiled = pytensor.function(inputs, outputs)(*values)

    for python_value, compiled_value in zip(in_python, compiled):
        np.testing.assert_allclose(python_value, compiled_value, atol=ATOL)


def test_col2im_needs_one_extent_per_spatial_axis():
    """The extents are positional, so a caller passing the wrong number of them would otherwise build
    a node whose rank silently disagrees with the kernel's."""
    patches = pt.tensor("patches", shape=(2, 9, 3, 3))

    with pytest.raises(ValueError, match="needs that many extents"):
        Col2Im((3, 3), (1, 1), (1, 1))(patches, 11)


def test_conv2d_strides_and_dilates_per_axis_like_scipy(rng):
    """Each axis carries its own stride and dilation, and only an asymmetric setting can catch the two
    being swapped or one of them broadcast over both axes. The rectangular kernel separates the two
    spatial axes, so a layer that transposed them would disagree too."""
    stride, dilation = (3, 2), (1, 2)
    X = pt.tensor("X", shape=(None, None, None, 3))
    layer = Conv2D(
        "conv",
        in_channels=3,
        out_channels=4,
        kernel_size=(2, 3),
        stride=stride,
        dilation=dilation,
        bias=False,
    )

    W_np = rng.normal(size=(2, 3, 3, 4)).astype(floatX)
    layer.W.set_value(W_np)
    X_np = rng.normal(size=(2, 14, 16, 3)).astype(floatX)

    np.testing.assert_allclose(
        layer(X).eval({X: X_np}),
        correlate_reference(X_np, W_np, stride=stride, dilation=dilation),
        atol=ATOL,
    )


def test_conv2d_pads_each_axis_by_its_own_amount():
    """An explicit padding takes one amount per spatial axis, applied to both sides of that axis, so
    a layer that broadcast the first amount or swapped the two would change the output's extents."""
    layer = Conv2D("conv", in_channels=2, out_channels=4, kernel_size=3, padding=(1, 2))

    assert layer(pt.tensor("X", shape=(None, 5, 5, 2))).type.shape == (None, 5, 7, 4)


def test_conv2d_rejects_a_padding_it_cannot_read():
    """A misspelled mode and a padding sized for another rank are the natural mistakes, and either
    would otherwise surface from `pt.pad` far from the constructor."""
    with pytest.raises(ValueError, match="padding must be 'valid', 'same', or an explicit number"):
        Conv2D("conv", in_channels=1, out_channels=1, kernel_size=3, padding="full")

    with pytest.raises(ValueError, match="needs one padding amount per axis, but got 3"):
        Conv2D("conv", in_channels=1, out_channels=1, kernel_size=3, padding=(1, 1, 1))


def test_conv2d_same_padding_puts_the_odd_element_after_on_each_axis():
    """At an even extent `same` cannot pad symmetrically, and the extra element goes after rather than
    before. Nothing about the output shape distinguishes the two, so only the values pin it -- and a
    mixed-parity kernel puts the asymmetry on one axis while the other stays symmetric."""
    X = pt.tensor("X", shape=(None, None, None, 1))
    X_np = np.ones((1, 4, 4, 1), dtype=floatX)

    layer = Conv2D(
        "conv", in_channels=1, out_channels=1, kernel_size=(2, 3), padding="same", bias=False
    )
    layer.W.set_value(np.ones((2, 3, 1, 1), dtype=floatX))

    # Summing over a constant input counts how many real elements each window saw. Height pads (0, 1),
    # so only the last row is short; width pads (1, 1), so the first and last columns are.
    np.testing.assert_allclose(
        layer(X).eval({X: X_np})[0, :, :, 0],
        [[4, 6, 6, 4], [4, 6, 6, 4], [4, 6, 6, 4], [2, 3, 3, 2]],
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "spatial, expected",
    [((28, 28), (32, 24, 24, 16)), ((None, 28), (32, None, 24, 16))],
    ids=["static", "one_axis_dynamic"],
)
def test_a_conv_stack_keeps_the_output_shape_it_can_work_out(spatial, expected):
    """A layer downstream of a convolution has to size itself from the graph, so the extents the input
    does know have to survive the op. They reach the output only through reshapes that can fold their
    targets, which a product over a shape slice cannot."""
    X = pt.tensor("X", shape=(32, *spatial, 3))
    first = Conv2D("c1", in_channels=3, out_channels=8, kernel_size=3)(X)
    second = Conv2D("c2", in_channels=8, out_channels=16, kernel_size=3)(first)

    assert second.type.shape == expected


def test_a_convolutional_network_trains_end_to_end(rng):
    """The step the whole plan is for: convolution, pooling, spatial batch norm and a dense head, in
    one graph, learning. Each piece is tested alone elsewhere -- this is the only test that says they
    compose, and that gradients survive every op boundary between them."""
    X = Input("X", shape=(None, 8, 8, 2))
    features = Conv2D("conv", in_channels=2, out_channels=4, kernel_size=3, padding="same")(X)
    normalized = BatchNorm("norm", n_in=4)(ReLU()(features))
    pooled = MaxPool2D("pool", kernel_size=2)(normalized)
    y = Linear("head", n_in=4 * 4 * 4, n_out=1)(Flatten(pooled))

    model = Model(y).initialize(seed=1)
    step = model.compile_train(adam(learning_rate=0.05), SquaredError())

    X_np = rng.normal(size=(32, 8, 8, 2)).astype(floatX)
    y_np = X_np.sum(axis=(1, 2, 3))[:, None].astype(floatX)

    losses = [float(step(X_np, y_np)) for _ in range(50)]
    assert losses[-1] < losses[0] / 5


@pytest.mark.parametrize(
    "wanted, expected",
    [("dX", (True, False)), ("dW", (False, True)), ("both", (True, True))],
    ids=["input_only", "kernel_only", "both"],
)
def test_the_pullback_computes_only_the_gradients_something_reads(wanted, expected):
    """`ConvLayer.pullback` asks for both gradients because only the graph knows which are wanted, and
    only once it is built. A rewrite then drops whichever has no clients -- the input gradient for the
    first convolution of a network, the kernel gradient for a transposed one. That the gradient it
    keeps is unchanged is checked against the op directly, below."""
    X = pt.tensor("X", shape=(4, 24, 3))
    layer = Conv1D("conv", in_channels=3, out_channels=5, kernel_size=3)
    cost = (layer(X) ** 2).sum()
    targets = {
        "dX": [pt.grad(cost, X)],
        "dW": [pt.grad(cost, layer.W)],
        "both": pt.grad(cost, [X, layer.W]),
    }[wanted]

    rewritten = rewrite_graph(targets, include=("canonicalize", "specialize"))
    grad_op = next(
        node.op for node in apply_ancestors(rewritten) if isinstance(node.op, ConvLayerGrad)
    )
    assert (grad_op.compute_dX, grad_op.compute_dW) == expected


@pytest.mark.parametrize("dropped", ["dX", "dW"], ids=["without_dX", "without_dW"])
def test_dropping_one_gradient_leaves_the_other_unchanged(dropped, rng):
    """The rewrite is only safe if a lowered op returns the same numbers as the pair it replaces, so
    this compares them directly rather than trusting that fewer outputs means the same arithmetic."""
    X_np = rng.normal(size=(2, 6, 6, 3)).astype(floatX)
    W_np = rng.normal(size=(3, 3, 3, 4)).astype(floatX)
    V_np = rng.normal(size=(2, 4, 4, 4)).astype(floatX)
    X = pt.tensor("X", shape=X_np.shape)
    W = pt.tensor("W", shape=W_np.shape)
    cotangent = pt.tensor("cotangent", shape=V_np.shape)

    geometry = ((3, 3), (1, 1), (1, 1))
    keeping_dX = dropped == "dW"
    both = ConvLayerGrad(*geometry)(X, W, cotangent)
    alone = ConvLayerGrad(*geometry, compute_dX=keeping_dX, compute_dW=not keeping_dX)(
        X, W, cotangent
    )

    kept = both[0] if keeping_dX else both[1]
    values = pytensor.function([X, W, cotangent], [kept, alone])(X_np, W_np, V_np)
    np.testing.assert_allclose(*values, atol=ATOL)


def test_the_pullback_must_return_some_gradient():
    """Both flags false describes an op with no outputs, which would build and then fail somewhere
    downstream rather than where the mistake was made."""
    with pytest.raises(ValueError, match="must return at least one gradient"):
        ConvLayerGrad((3,), (1,), (1,), compute_dX=False, compute_dW=False)


@pytest.mark.parametrize(
    "flags",
    [{"compute_dW": False}, {"compute_dX": False}, {}],
    ids=["dX_only", "dW_only", "both"],
)
def test_differentiating_a_conv_gradient_leaves_only_dispatchable_ops(flags):
    """A transposed convolution is `ConvLayerGrad` run forward, so training one differentiates through
    it. Inheriting `OpFromGraph`'s pullback would wrap the gather in an anonymous op carrying no props
    and registered against no type, which every backend but numba refuses outright. Each combination of
    flags builds a different set of terms, so each is checked."""
    geometry = ((3, 3), (1, 1), (1, 1))
    X = pt.tensor("X", shape=(2, 6, 6, 3))
    W = pt.tensor("W", shape=(3, 3, 3, 4))
    cotangent = pt.tensor("cotangent", shape=(2, 4, 4, 4))

    outputs = ConvLayerGrad(*geometry, **flags)(X, W, cotangent, return_list=True)
    cost = sum((out**2).sum() for out in outputs)
    gradients = pt.grad(cost, [X, W, cotangent], disconnected_inputs="ignore")
    fn = pytensor.function([X, W, cotangent], gradients)

    convolutions = {ConvLayer, ConvLayerGrad}
    leftover = [
        node.op
        for node in fn.maker.fgraph.apply_nodes
        if isinstance(node.op, OpFromGraph) and type(node.op) not in convolutions
    ]
    assert not leftover, f"undispatchable ops survived the pullback: {leftover}"


@pytest.mark.parametrize(
    "flags",
    [{"compute_dW": False}, {"compute_dX": False}, {}],
    ids=["dX_only", "dW_only", "both"],
)
def test_the_pullback_of_the_pullback_matches_finite_differences(flags, rng):
    """The closed forms the pullback uses are adjoint identities rather than a differentiated graph,
    so they are checked numerically. Every input is perturbed, including the one each output
    ignores. The pullback has no branch on geometry, so one stride and one dilation, each differing
    per axis, stand in for all of them."""
    geometry = ((3, 3), (2, 1), (1, 2))
    X_np = rng.normal(size=(2, 7, 7, 3)).astype(floatX)
    W_np = rng.normal(size=(3, 3, 3, 4)).astype(floatX)
    cotangent_shape = ConvLayer(*geometry)(
        pt.zeros(X_np.shape, dtype=floatX), pt.tensor(shape=W_np.shape)
    ).type.shape
    cotangent_np = rng.normal(size=cotangent_shape).astype(floatX)
    op = ConvLayerGrad(*geometry, **flags)

    def summed_outputs(X, W, cotangent):
        return sum((out**2).sum() for out in op(X, W, cotangent, return_list=True))

    verify_grad(summed_outputs, [X_np, W_np, cotangent_np], rng=np.random.default_rng(0))


def test_col2im_infers_its_shape_from_symbolic_extents():
    """`Col2Im` takes the output's spatial extents as separate scalar inputs, so `input_shapes` holds
    one entry per input rather than only the patches'. A statically known extent is folded away before
    shape inference runs, so only a symbolic one reaches the failure."""
    patches = pt.tensor("patches", shape=(None, 4, 4, 3, 3, 5))
    extent = pt.scalar("extent", dtype="int64")
    out = Col2Im((3, 3), (2, 2), (1, 1))(patches, extent, extent)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shapes = pytensor.function([patches, extent], out.shape)
        assert not [w for w in caught if "infer_shape" in str(w.message)]

    assert list(shapes(np.zeros((2, 4, 4, 3, 3, 5)), 9)) == [2, 9, 9, 5]
