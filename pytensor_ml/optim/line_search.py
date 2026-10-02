from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.compile.builders import OpFromGraph
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import ancestors, truncated_graph_inputs
from pytensor.scan.utils import until
from pytensor.tensor import TensorVariable

from pytensor_ml.optim.lbfgs import flat_dot
from pytensor_ml.pytensorf.rng import find_generators_drawn_from

State = dict[str, TensorVariable]
ValueAndSlope = Callable[[TensorVariable], tuple[TensorVariable, TensorVariable]]


class LineSearch(Protocol):
    """
    A search for a step size along a direction, written as one iteration over a scalar state.

    :func:`search_along` repeats :meth:`step` inside a ``scan`` that stops as soon as it reports it is
    done, and every iteration evaluates the loss and its slope at exactly one trial step.
    """

    max_steps: int

    def init(self, value: TensorVariable, slope: TensorVariable) -> State:
        """The state before the first trial, from the loss and its slope at a step of zero."""
        ...

    def step(
        self,
        state: State,
        value_and_slope: ValueAndSlope,
        value0: TensorVariable,
        slope0: TensorVariable,
        guess: TensorVariable,
    ) -> tuple[State, TensorVariable]:
        """Pick one trial step, evaluate it, and return the next state and whether to stop."""
        ...

    def finalize(self, state: State) -> tuple[TensorVariable, TensorVariable]:
        """The step size to take and whether the search failed, from the state it stopped in."""
        ...


def _cubic_minimizer(a, value_a, slope_a, b, value_b, c, value_c):
    """Minimizer of the cubic through three values and the slope at ``a``, or NaN if it has none."""
    db = b - a
    dc = c - a
    denominator = (db * dc) ** 2 * (db - dc)
    residual_b = value_b - value_a - slope_a * db
    residual_c = value_c - value_a - slope_a * dc
    cubic = (dc**2 * residual_b - db**2 * residual_c) / denominator
    quadratic = (-(dc**3) * residual_b + db**3 * residual_c) / denominator
    return a + (-quadratic + pt.sqrt(quadratic**2 - 3.0 * cubic * slope_a)) / (3.0 * cubic)


def _quadratic_minimizer(a, value_a, slope_a, b, value_b):
    """Minimizer of the quadratic through two values and the slope at ``a``."""
    db = b - a
    curvature = (value_b - value_a - slope_a * db) / db**2
    return a - slope_a / (2.0 * curvature)


_ZOOM_FIELDS = (
    "count",
    "stepsize",
    "value",
    "slope",
    "decrease_error",
    "curvature_error",
    "interval_found",
    "done",
    "failed",
    "low",
    "value_low",
    "slope_low",
    "high",
    "value_high",
    "slope_high",
    "cubic_ref",
    "value_cubic_ref",
    "safe_stepsize",
    "safe_value",
)


@dataclass(frozen=True)
class ZoomLineSearch:
    """
    A line search for a step satisfying the strong Wolfe conditions, by bracketing and then zooming.

    Built by :func:`zoom_line_search`, which documents the parameters. It follows optax's
    ``scale_by_zoom_linesearch``: the bracket grows from the initial guess until it contains a point
    satisfying both conditions, then narrows by safeguarded cubic or quadratic interpolation, falling
    back to bisection.
    """

    max_steps: int
    slope_rtol: float
    curv_rtol: float
    approx_dec_rtol: float | None
    increase_factor: float
    max_learning_rate: float | None
    tol: float
    stepsize_precision: float

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {self.max_steps}.")
        if not 0.0 < self.slope_rtol < self.curv_rtol < 1.0:
            raise ValueError(
                f"The Wolfe conditions need 0 < slope_rtol < curv_rtol < 1, got "
                f"slope_rtol={self.slope_rtol} and curv_rtol={self.curv_rtol}."
            )
        if self.increase_factor <= 1.0:
            raise ValueError(f"increase_factor must exceed 1, got {self.increase_factor}.")

    def init(self, value: TensorVariable, slope: TensorVariable) -> State:
        zero = pt.zeros((), dtype=value.dtype)
        no = pt.constant(np.array(False))
        infinite = pt.constant(np.inf, dtype=value.dtype)
        return {
            "count": pt.constant(0, dtype="int64"),
            "stepsize": zero,
            "value": value,
            "slope": slope,
            "decrease_error": infinite,
            "curvature_error": infinite,
            "interval_found": no,
            "done": no,
            "failed": no,
            "low": zero,
            "value_low": value,
            "slope_low": slope,
            "high": zero,
            "value_high": value,
            "slope_high": slope,
            "cubic_ref": zero,
            "value_cubic_ref": value,
            "safe_stepsize": zero,
            "safe_value": value,
        }

    def _decrease_error(self, stepsize, value, slope, value0, slope0):
        """How far a trial misses sufficient decrease, by Armijo or by the approximate Wolfe test."""
        error = value - value0 - self.slope_rtol * stepsize * slope0
        if self.approx_dec_rtol is not None:
            approximate = pt.maximum(
                slope - (2 * self.slope_rtol - 1.0) * slope0,
                value - value0 - self.approx_dec_rtol * pt.abs(value0),
            )
            error = pt.minimum(approximate, error)
        error = pt.maximum(error, 0.0)
        return pt.where(pt.isnan(error), np.inf, error)

    def _curvature_error(self, slope, slope0):
        """How far a trial misses the strong Wolfe curvature condition."""
        error = pt.maximum(pt.abs(slope) - self.curv_rtol * pt.abs(slope0), 0.0)
        return pt.where(pt.isnan(error), np.inf, error)

    def step(self, state, value_and_slope, value0, slope0, guess):
        where = pt.where
        count = state["count"]
        low, value_low, slope_low = state["low"], state["value_low"], state["slope_low"]
        high, value_high, slope_high = state["high"], state["value_high"], state["slope_high"]

        # Both phases' candidates are cheap scalar arithmetic, so both are formed and the bracket flag
        # picks one; only the loss evaluation that follows is expensive.
        increase_factor = pt.constant(self.increase_factor, dtype=guess.dtype)
        bracketing_trial = where(pt.eq(count, 0), guess, increase_factor * state["stepsize"])
        if self.max_learning_rate is None:
            max_reached = pt.constant(np.array(False))
        else:
            max_learning_rate = pt.constant(self.max_learning_rate, dtype=guess.dtype)
            max_reached = bracketing_trial >= max_learning_rate
            bracketing_trial = pt.minimum(bracketing_trial, max_learning_rate)

        width = pt.abs(high - low)
        left = pt.minimum(high, low)
        right = pt.maximum(high, low)
        cubic = _cubic_minimizer(
            low,
            value_low,
            slope_low,
            high,
            value_high,
            state["cubic_ref"],
            state["value_cubic_ref"],
        )
        use_cubic = (cubic > left + 0.2 * width) & (cubic < right - 0.2 * width)
        quadratic = _quadratic_minimizer(low, value_low, slope_low, high, value_high)
        use_quadratic = (
            ~use_cubic & (quadratic > left + 0.1 * width) & (quadratic < right - 0.1 * width)
        )
        middle = where(use_cubic, cubic, where(use_quadratic, quadratic, (low + high) / 2.0))
        zooming = state["interval_found"]
        stepsize = where(zooming, middle, bracketing_trial).astype(guess.dtype)

        value, slope = value_and_slope(stepsize)
        decrease_error = self._decrease_error(stepsize, value, slope, value0, slope0)
        curvature_error = self._curvature_error(slope, slope0)
        satisfied = pt.maximum(decrease_error, curvature_error) <= self.tol
        sufficient_decrease = decrease_error <= self.tol
        out_of_steps = (count + 1) >= self.max_steps
        trial = (stepsize, value, slope)

        bracketing = {}
        bracketing["safe_stepsize"] = where(sufficient_decrease, stepsize, state["safe_stepsize"])
        bracketing["safe_value"] = where(sufficient_decrease, value, state["safe_value"])
        high_is_trial = (decrease_error > 0.0) | ((value >= state["value"]) & (count > 0))
        low_is_trial = (slope >= 0.0) & ~high_is_trial
        previous = (state["stepsize"], state["value"], state["slope"])
        bracketing["low"], bracketing["value_low"], bracketing["slope_low"] = (
            where(low_is_trial, new, old) for new, old in zip(trial, previous)
        )
        bracketing["high"], bracketing["value_high"], bracketing["slope_high"] = (
            where(low_is_trial, old, new) for new, old in zip(trial, previous)
        )
        bracketing["cubic_ref"] = bracketing["low"]
        bracketing["value_cubic_ref"] = bracketing["value_low"]
        bracketing["interval_found"] = high_is_trial | low_is_trial | satisfied
        bracketing["done"] = satisfied | (max_reached & ~bracketing["interval_found"])
        bracketing["failed"] = out_of_steps & ~bracketing["done"]

        zoom = {}
        improves_safe = sufficient_decrease & (value < state["safe_value"])
        zoom["safe_stepsize"] = where(improves_safe, stepsize, state["safe_stepsize"])
        zoom["safe_value"] = where(improves_safe, value, state["safe_value"])
        zoom["done"] = satisfied
        high_is_middle = (decrease_error > 0.0) | (value >= value_low)
        high_is_low = (slope * (high - low) >= 0.0) & ~high_is_middle
        high_after_middle = [
            where(high_is_middle, new, old)
            for new, old in zip(trial, (high, value_high, slope_high))
        ]
        zoom["high"], zoom["value_high"], zoom["slope_high"] = (
            where(high_is_low, new, old)
            for new, old in zip((low, value_low, slope_low), high_after_middle)
        )
        zoom["low"], zoom["value_low"], zoom["slope_low"] = (
            where(~high_is_middle, new, old) for new, old in zip(trial, (low, value_low, slope_low))
        )
        zoom["cubic_ref"] = where(high_is_middle | high_is_low, high, low)
        zoom["value_cubic_ref"] = where(high_is_middle | high_is_low, value_high, value_low)
        zoom["interval_found"] = state["interval_found"]
        too_narrow = (width <= self.stepsize_precision) & (zoom["safe_stepsize"] > 0.0)
        zoom["failed"] = (out_of_steps | too_narrow) & ~zoom["done"]

        next_state = {
            "count": count + 1,
            "stepsize": stepsize,
            "value": value,
            "slope": slope,
            "decrease_error": decrease_error,
            "curvature_error": curvature_error,
        }
        for field in bracketing:
            next_state[field] = where(zooming, zoom[field], bracketing[field])
        next_state = {
            field: pt.as_tensor_variable(value).astype(state[field].dtype)
            for field, value in next_state.items()
        }
        return next_state, next_state["done"] | next_state["failed"]

    def finalize(self, state):
        # A failed search falls back on the best step that met sufficient decrease, or on not moving at
        # all when the last trial was not even finite.
        fall_back = state["failed"] & (
            (state["safe_stepsize"] > 0.0) | pt.isinf(state["decrease_error"])
        )
        return pt.where(fall_back, state["safe_stepsize"], state["stepsize"]), state["failed"]


def zoom_line_search(
    max_steps: int = 20,
    *,
    slope_rtol: float = 1e-4,
    curv_rtol: float = 0.9,
    approx_dec_rtol: float | None = 1e-6,
    increase_factor: float = 2.0,
    max_learning_rate: float | None = None,
    tol: float = 0.0,
    stepsize_precision: float = 1e-5,
) -> ZoomLineSearch:
    r"""
    A line search for a step satisfying the strong Wolfe conditions.

    A trial step :math:`t` along a direction :math:`d` from :math:`p` is accepted when it decreases the
    loss enough and flattens its slope enough,

    .. math::

        f(p + t d) \le f(p) + c_1 t \nabla f(p)^\top d, \qquad
        |\nabla f(p + t d)^\top d| \le c_2 |\nabla f(p)^\top d|,

    or when it passes the approximate Wolfe test of :cite:t:`hager2005new` in place of the first. The
    search stops at the first accepted trial, so it costs one loss and gradient evaluation per trial and
    a single one when the initial guess is accepted. When ``max_steps`` trials pass without one, it
    takes the best step that met sufficient decrease, or no step at all if the last trial was not
    finite.

    Parameters
    ----------
    max_steps : int
        The most trial steps per search. Default 20.
    slope_rtol : float
        :math:`c_1`, the fraction of the initial slope the loss has to fall by. Default 1e-4.
    curv_rtol : float
        :math:`c_2`, the fraction of the initial slope's magnitude the trial slope has to fall below.
        Default 0.9.
    approx_dec_rtol : float, optional
        Relative tolerance of the approximate Wolfe test, or None to require the Armijo condition
        alone. Default 1e-6.
    increase_factor : float
        Growth of the trial step while the search has not yet bracketed an acceptable one. Default 2.0.
    max_learning_rate : float, optional
        The largest trial step. Default None, which leaves the step unbounded.
    tol : float
        How far a trial may miss either condition and still be accepted. Default 0.0.
    stepsize_precision : float
        The bracket width below which a search that has a safe step stops. Default 1e-5.

    Returns
    -------
    line_search : ZoomLineSearch
        The configured search, for :func:`search_along` or an optimizer's ``line_search`` argument.
    """
    return ZoomLineSearch(
        max_steps=max_steps,
        slope_rtol=slope_rtol,
        curv_rtol=curv_rtol,
        approx_dec_rtol=approx_dec_rtol,
        increase_factor=increase_factor,
        max_learning_rate=max_learning_rate,
        tol=tol,
        stepsize_precision=stepsize_precision,
    )


class LineSearchOp(OpFromGraph):
    """
    One line search as a single node, built by :func:`search_along`.

    Inputs are the parameters, the directions, the loss and slope at the current point, the first trial
    step, and everything else the loss reads; outputs are the step size, whether the search failed, and
    how many trials it took. The inner graph runs ``finish(scan(step, start(...)))``, the ``scan``
    stopping at the first trial the search accepts, and every ``step`` evaluates the loss through one
    ``trial`` node. Each of those pieces is a named ``OpFromGraph``, so a backend that cannot run the
    ``scan`` finds them with :func:`find_piece` in the graph its linker has already rewritten.

    Parameters
    ----------
    inputs : list of Variable
        The inputs described above, in that order.
    outputs : list of Variable
        The step size, the failure flag and the trial count.
    line_search : LineSearch
        The search the inner graph runs.
    n_parameters : int
        How many parameters, and so how many directions, lead the inputs.
    """

    def __init__(
        self,
        inputs: list[Variable],
        outputs: list[Variable],
        *,
        line_search: LineSearch,
        n_parameters: int,
        **kwargs,
    ):
        super().__init__(inputs, outputs, name="line_search", **kwargs)
        self.line_search = line_search
        self.n_parameters = n_parameters

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.line_search == other.line_search
            and self.n_parameters == other.n_parameters
        )

    def __hash__(self):
        return hash((super().__hash__(), self.line_search, self.n_parameters))


# The pieces of a `LineSearchOp`'s inner graph, by name. `start` takes the loss and slope at the current
# point to the first state; `step` takes a state and then every input of the search to the next state and
# whether to stop; `trial` takes a step size, the parameters, the directions and every input after the
# first trial step to the loss and its slope there; `finish` takes the final state to the outputs.
START = "line_search_start"
STEP = "line_search_step"
TRIAL = "line_search_trial"
FINISH = "line_search_finish"


def find_piece(op: LineSearchOp, name: str) -> OpFromGraph:
    """
    Return the piece of ``op``'s inner graph called ``name``, as the linker left it.

    The pieces are looked for in the inner graph itself and in every inner graph nested in it, so a
    piece inside the ``scan`` or inside ``step`` is found too.

    Parameters
    ----------
    op : LineSearchOp
        A search whose inner graph has been rewritten for the backend converting it.
    name : str
        One of ``START``, ``STEP``, ``TRIAL`` and ``FINISH``.

    Returns
    -------
    piece : OpFromGraph
        The piece, with the inner graph the linker rewrote.
    """
    pending = [op.fgraph]
    while pending:
        for node in pending.pop().apply_nodes:
            if isinstance(node.op, OpFromGraph) and node.op.name == name:
                return node.op
            inner = getattr(node.op, "fgraph", None)
            if inner is not None:
                pending.append(inner)
    raise RuntimeError(
        f"The line search's inner graph has no {name!r} piece once rewritten for this backend; a "
        f"rewrite inlined or merged it, and the backend cannot run the search without it."
    )


class SearchResult(NamedTuple):
    """
    What a line search found.

    Attributes
    ----------
    step_size : TensorVariable
        The step to take along the direction.
    failed : TensorVariable
        Whether the search ran out of trials, or narrowed below its precision, without accepting one.
    evaluations : TensorVariable
        How many trial steps it evaluated.
    """

    step_size: TensorVariable
    failed: TensorVariable
    evaluations: TensorVariable


def search_along(
    loss: TensorVariable,
    parameters: Sequence[TensorVariable],
    gradients: Sequence[TensorVariable],
    directions: Sequence[TensorVariable],
    line_search: LineSearch,
    guess: float | TensorVariable = 1.0,
) -> SearchResult:
    """
    Search for a step size along ``directions`` from the current ``parameters``.

    Each trial evaluates ``loss`` and its gradients with every parameter moved to ``p + t * d``, inside a
    ``scan`` that stops at the first trial the search accepts. The loop is one ``OpFromGraph`` node in
    the graph it returns. numba and the default backends run that ``scan`` directly, JAX runs its step
    in a ``lax.while_loop``, and mlx runs every trial its budget allows, holding the state once one is
    accepted.

    Parameters
    ----------
    loss : TensorVariable
        Scalar loss of the parameters. It has to be deterministic: a loss that draws random numbers would
        draw again at every trial, and the trials would not be measuring one function.
    parameters : sequence of TensorVariable
        The variables the trials move.
    gradients : sequence of TensorVariable
        The gradient of ``loss`` with respect to each parameter, at the current point.
    directions : sequence of TensorVariable
        The direction to move each parameter in, one per parameter.
    line_search : LineSearch
        The search to run, for example :func:`zoom_line_search`.
    guess : float or TensorVariable
        The first trial step. Default 1.0.

    Returns
    -------
    result : SearchResult
        The step size, whether the search failed, and how many trials it took.
    """
    generators = find_generators_drawn_from([loss])
    if generators:
        names = ", ".join(str(generator) for generator in generators)
        raise ValueError(
            f"A line search evaluates the loss at several trial points, so a loss that draws random "
            f"numbers would draw again at each one. This loss draws from {names}; remove the randomness "
            f"(dropout, sampling) from the graph the optimizer sees."
        )

    loss_inputs = set(ancestors([loss]))
    unused = [parameter for parameter in parameters if parameter not in loss_inputs]
    if unused:
        names = ", ".join(str(parameter) for parameter in unused)
        raise ValueError(
            f"The loss does not depend on {names}, so a line search has nothing to measure moving it "
            f"by. Leave it out of the parameters searched over."
        )

    dtype = loss.dtype
    value0 = loss
    slope0 = flat_dot(gradients, directions).astype(dtype)
    first_trial = pt.as_tensor_variable(guess).astype(dtype)

    # Each piece is built on stand-ins for what it receives, so it is a graph of its own inputs rather
    # than of the point the search starts from.
    start_value, start_slope, first_guess = value0.type(), slope0.type(), first_trial.type()
    initial = line_search.init(start_value, start_slope)
    fields = list(initial)
    start = OpFromGraph(
        [start_value, start_slope], [initial[field] for field in fields], name=START
    )

    trial_step = first_trial.type()
    moved: dict[Variable, Variable] = {
        parameter: parameter + trial_step.astype(parameter.dtype) * direction
        for parameter, direction in zip(parameters, directions)
    }
    trial_value, *trial_gradients = graph_replace([loss, *gradients], moved, strict=True)
    trial_outputs = [
        trial_value,
        flat_dot(trial_gradients, directions).astype(dtype),  # type: ignore[arg-type]
    ]
    moving = [trial_step, *parameters, *directions]
    others = [
        variable
        for variable in truncated_graph_inputs(trial_outputs, ancestors_to_include=moving)
        if variable not in moving and not isinstance(variable, Constant)
    ]
    trial = OpFromGraph([*moving, *others], trial_outputs, name=TRIAL)

    def value_and_slope(stepsize):
        return tuple(trial(stepsize, *parameters, *directions, *others, return_list=True))

    state = {field: initial[field].type() for field in fields}
    next_state, stop = line_search.step(
        state, value_and_slope, start_value, start_slope, first_guess
    )
    inputs = [*parameters, *directions, value0, slope0, first_trial, *others]
    step = OpFromGraph(
        [*state.values(), *parameters, *directions, start_value, start_slope, first_guess, *others],
        [*(next_state[field] for field in fields), stop],
        name=STEP,
    )

    final = {field: initial[field].type() for field in fields}
    step_size, failed = line_search.finalize(final)
    finish = OpFromGraph(list(final.values()), [step_size, failed, final["count"]], name=FINISH)

    def iterate(*carry_and_inputs):
        *next_carry, stop_here = step(*carry_and_inputs, return_list=True)
        return next_carry, until(stop_here)

    trace = pytensor.scan(
        iterate,
        outputs_info=start(value0, slope0, return_list=True),
        non_sequences=inputs,
        n_steps=line_search.max_steps,
        return_updates=False,
    )
    outputs = finish.make_node(*(values[-1] for values in trace)).outputs

    search = LineSearchOp(inputs, outputs, line_search=line_search, n_parameters=len(parameters))
    return SearchResult(*search.make_node(*inputs).outputs)
