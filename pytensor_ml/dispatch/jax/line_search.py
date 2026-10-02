import jax
import jax.numpy as jnp

from pytensor.link.jax.dispatch import jax_funcify

from pytensor_ml.optim.line_search import TRIAL, LineSearchOp, ZoomLineSearch, find_piece


@jax_funcify.register(LineSearchOp)
def jax_funcify_LineSearchOp(op, node=None, **kwargs):
    """
    Run the search through optax's, which ``lax.while_loop`` drives, since JAX cannot run a ``scan``
    that stops early.

    optax evaluates the loss through the search's own ``trial`` piece, as the linker rewrote it. It
    always starts its search at a step of one, so the direction is scaled by the first trial step on the
    way in and the step it finds is scaled back on the way out, with every tolerance measured in steps
    scaled to match.
    """
    try:
        import optax  # Optional: only a JAX run of a line search needs it
    except ImportError as error:
        raise ImportError(
            "Running a line search on JAX calls optax's, so optax has to be installed: pip install optax."
        ) from error

    search = op.line_search
    if not isinstance(search, ZoomLineSearch):
        raise NotImplementedError(
            f"The JAX backend runs a zoom line search through optax, and has no implementation of "
            f"{type(search).__name__}."
        )
    kwargs.pop("storage_map", None)
    trial = jax_funcify(find_piece(op, TRIAL), **kwargs)

    n_parameters = op.n_parameters
    step_dtype, count_dtype = (
        (node.outputs[0].type.dtype, node.outputs[2].type.dtype)
        if node is not None
        else (None, "int64")
    )

    def line_search(*inputs):
        parameters = tuple(inputs[:n_parameters])
        directions = inputs[n_parameters : 2 * n_parameters]
        value0, _, guess = inputs[2 * n_parameters : 2 * n_parameters + 3]
        others = inputs[2 * n_parameters + 3 :]
        one = jnp.ones((), dtype=value0.dtype)

        def value_fn(point):
            # The trial at a step of one along the way from the current point to `point` is the loss there
            offsets = tuple(there - here for there, here in zip(point, parameters))
            return trial(one, *parameters, *offsets, *others)[0]

        searcher = optax.scale_by_zoom_linesearch(
            max_linesearch_steps=search.max_steps,
            max_learning_rate=(
                None if search.max_learning_rate is None else search.max_learning_rate / guess
            ),
            tol=search.tol,
            increase_factor=search.increase_factor,
            slope_rtol=search.slope_rtol,
            curv_rtol=search.curv_rtol,
            approx_dec_rtol=search.approx_dec_rtol,
            stepsize_precision=search.stepsize_precision / guess,
            initial_guess_strategy="one",
        )
        scaled_directions = tuple(guess * direction for direction in directions)
        _, state = searcher.update(
            scaled_directions,
            searcher.init(parameters),
            parameters,
            value=value0,
            grad=jax.grad(value_fn)(parameters),
            value_fn=value_fn,
        )
        info = state.info
        # optax reports no failure flag. Its errors describe the last trial, which is the step it takes
        # whenever it succeeds, so a last trial that misses either condition means the search failed --
        # except when the search stops at the largest step it may take without having bracketed one,
        # which it counts as done although the conditions still fail there.
        failed = jnp.maximum(info.decrease_error, info.curvature_error) > search.tol
        if search.max_learning_rate is not None:
            failed = failed & (state.learning_rate < search.max_learning_rate / guess)
        step_size = jnp.asarray(state.learning_rate * guess, dtype=step_dtype)
        return step_size, failed, jnp.asarray(info.num_linesearch_steps, dtype=count_dtype)

    return line_search
