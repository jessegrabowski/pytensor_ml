import numpy as np
import pytensor

floatX = pytensor.config.floatX


def dense_inverse_hessian(gamma, pairs, size):
    """The matrix the two-loop recursion multiplies by, built from its definition: BFGS updates from
    ``gamma I`` over ``(s, y)`` pairs oldest first, ``H <- V^T H V + rho s s^T`` with ``V = I - rho y s^T``
    (Nocedal and Wright, equation 7.16)."""
    H = gamma * np.eye(size)
    for s, y in pairs:
        s, y = s.astype(np.float64), y.astype(np.float64)
        rho = 1.0 / (y @ s)
        V = np.eye(size) - rho * np.outer(y, s)
        H = V.T @ H @ V + rho * np.outer(s, s)
    return H


def ring_stacks(pairs, memory_size, count, shapes):
    """Lay chronological flat pairs into per-parameter ring stacks, newest at ``(count - 1) % memory_size``,
    and each slot's curvature ``1 / (y . s)`` into a vector beside them, zero where a slot is empty."""
    S = [np.zeros((memory_size, *shape), dtype=floatX) for shape in shapes]
    Y = [np.zeros((memory_size, *shape), dtype=floatX) for shape in shapes]
    rho = np.zeros(memory_size, dtype=floatX)
    splits = np.cumsum([int(np.prod(shape)) for shape in shapes])[:-1]
    for age, (s, y) in enumerate(reversed(pairs)):
        slot = (count - 1 - age) % memory_size
        for stack, piece in zip(S, np.split(s, splits)):
            stack[slot] = piece.reshape(stack.shape[1:])
        for stack, piece in zip(Y, np.split(y, splits)):
            stack[slot] = piece.reshape(stack.shape[1:])
        rho[slot] = 1.0 / (y.astype(np.float64) @ s.astype(np.float64))
    return S, Y, rho
