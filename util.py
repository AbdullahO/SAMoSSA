import numpy as np
from numpy.typing import NDArray
from itertools import product
from typing import Dict


def lowestMultiple(target: int, base: int) -> int:
    assert base <= target, "Arguments are likely in the wrong order!"
    return int(target // base) * base


def truncatedSVD(matrix: NDArray, rank: int) -> NDArray:
    leftSingularVecs, singularVals, rightSingularVecsT = np.linalg.svd(
        matrix, full_matrices=False
    )
    denoisedMatrix = (
        leftSingularVecs[:, :rank]
        @ np.diag(singularVals[:rank])
        @ rightSingularVecsT[:rank]
    )
    return denoisedMatrix


def donohoRank(matrix: NDArray):
    m, n = matrix.shape
    b = m / n
    omega = 0.56 * b**3 - 0.95 * b**2 + 1.43 + 1.82 * b
    singularVals = np.linalg.svd(matrix, compute_uv=False)
    threshold = omega * np.median(singularVals)
    numSvsNeeded = int(np.sum(singularVals > threshold))
    return max(1, numSvsNeeded)


def energyRank(matrix: NDArray, threshold: float = 0.9):
    sqSingularVals = np.linalg.svd(matrix, compute_uv=False) ** 2
    cumul = sqSingularVals.cumsum() / sqSingularVals.sum()
    return np.sum(cumul < threshold) + 1


def leastSquares(covariates: NDArray, target: NDArray) -> NDArray:
    assert covariates.ndim == 2, "Covariates should be a matrix!"
    assert target.ndim == 1, "Targets should be a vector!"
    assert (
        covariates.shape[0] == target.size
    ), f"Covariates have {covariates.shape[0]} samples but targets have {target.size}!"
    return np.linalg.lstsq(covariates, target, rcond=None)[0]


def learnAR(series: NDArray, arOrder: int) -> NDArray:
    assert series.ndim == 1, "Expected a vector!"
    numSteps = series.size

    XTX = np.empty((arOrder, arOrder))
    XTY = np.empty(arOrder)
    numSamples = numSteps - arOrder
    for i in range(arOrder):
        for j in range(arOrder):
            XTX[i, j] = series[i : i + numSamples] @ series[j : j + numSamples]
        XTY[i] = series[i : i + numSamples] @ series[arOrder:numSteps]
    solution = np.linalg.solve(XTX, XTY)
    return solution


def cartProd(grid: Dict[str, list]):
    keys = grid.keys()
    vals = grid.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))
