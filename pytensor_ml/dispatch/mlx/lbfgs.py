import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify

from pytensor_ml.optim.lbfgs import LBFGSDirection


@mlx_funcify.register(LBFGSDirection)
def mlx_funcify_LBFGSDirection(op, node=None, **kwargs):
    """Run the two-loop recursion as a Python loop over ``mx`` ops, since mlx has no scan."""
    n, m = op.n_parameters, op.memory_size

    def rows(stacks, slot):
        # `count` is traced under mx.compile, so the slot is an mx scalar and the row is gathered rather
        # than indexed from Python.
        return [mx.take(stack, slot, axis=0) for stack in stacks]

    def dot(left, right):
        # Vector matmul is the fastest dot mlx has from 0.32.2 (ml-explore/mlx#3580); before that it ran
        # one threadgroup and was slower than a fused reduction by two orders of magnitude.
        return sum(a.reshape(-1) @ b.reshape(-1) for a, b in zip(left, right))

    def direction(count, gamma, rho, *tensors):
        gradients = tensors[:n]
        S = tensors[n : 2 * n]
        Y = tensors[2 * n :]

        # Every row is gathered once, in ring order (oldest first), and reused by both loops.
        order = [(count + offset) % m for offset in range(m)]
        s_rows = [rows(S, slot) for slot in order]
        y_rows = [rows(Y, slot) for slot in order]
        curvatures = [mx.take(rho, slot) for slot in order]

        vector = list(gradients)
        alphas = [None] * m
        for position in reversed(range(m)):
            alphas[position] = curvatures[position] * dot(s_rows[position], vector)
            vector = [
                v - alphas[position].astype(v.dtype) * y_p
                for v, y_p in zip(vector, y_rows[position])
            ]
        vector = [gamma.astype(v.dtype) * v for v in vector]
        for position in range(m):
            beta = curvatures[position] * dot(y_rows[position], vector)
            vector = [
                v + (alphas[position] - beta).astype(v.dtype) * s_p
                for v, s_p in zip(vector, s_rows[position])
            ]
        return vector[0] if n == 1 else tuple(vector)

    return direction
