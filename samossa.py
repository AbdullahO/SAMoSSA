from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Optional, Union, List
from util import (
    learnAR,
    leastSquares,
    truncatedSVD,
    lowestMultiple,
    donohoRank,
    energyRank,
)


class TimeSeriesModel(ABC):
    @staticmethod
    @abstractmethod
    def updatable() -> bool:
        pass

    @staticmethod
    @abstractmethod
    def oneShot() -> bool:
        pass

    @abstractmethod
    def fit(self, series: NDArray) -> TimeSeriesModel:
        pass

    @abstractmethod
    def update(self, series: NDArray) -> TimeSeriesModel:
        pass

    @abstractmethod
    def predict(self, numSteps: int) -> NDArray:
        pass


class SAMoSSA(TimeSeriesModel):
    def __init__(
        self,
        numSeries: int,
        numCoefs: int,
        rank: Optional[int] = None,
        rankEst: str = "donoho",
        arOrder: Optional[Union[int, List[int]]] = None,
    ) -> None:
        '''
        Args:
            numSeries (int): the dimension of the mutlivairate time series
            numCoefs (int): A parameter for the number of lags used in the linear model 
            rank (Optional[int], optional): The rank used in the HSVT subroutine. Defaults to None. If None, use the rankEst method in rankEst 
            rankEst (str, optional): Method to compute the rank, choose one of  ("donoho", "energy"). Defaults to "donoho".
            arOrder (Optional[Union[int, List[int]]], optional): The order of the AR process. Defaults to None, where the AR process is not learned.
        '''
        super().__init__()
        if rank is None:
            assert rankEst in ("donoho", "energy")
        self.numSeries = numSeries
        self.numCoefs = numCoefs
        self.numPageRows = numCoefs + 1
        self.rank = rank
        self.rankEst = rankEst
        if arOrder is not None:
            self.arOrder = (
                arOrder if isinstance(arOrder, list) else [arOrder] * numSeries
            )
        else:
            self.arOrder = None
        self.maxOrder = max(1, max(self.arOrder)) if self.arOrder is not None else None
        self.fitted = False

    @staticmethod
    def updatable() -> bool:
        return True

    @staticmethod
    def oneShot() -> bool:
        return False

    def _check_dims(self, series: NDArray) -> None:
        assert series.ndim == 2, "Expected T x N matrix!"
        assert (
            series.shape[1] == self.numSeries
        ), f"Expected {self.numSeries} time series!"

    def fit(self, series: NDArray) -> TimeSeriesModel:
        assert not self.fitted, "Model already fitted!"
        self._check_dims(series)

        # Truncate time series to multiple of L, then form page matrix
        numSteps = series.shape[0]
        truncatedSteps = lowestMultiple(numSteps, self.numPageRows)
        self.page = series[:truncatedSteps].reshape(self.numPageRows, -1, order="F")

        # Denoise the page matrix, then fit the betas
        if self.rank is None:
            if self.rankEst == "donoho":
                self.rank = donohoRank(self.page)
            elif self.rankEst == "energy":
                self.rank = energyRank(self.page)
            else:
                assert False, "rankEst must be either (donoho) or (energy)"
        self.denoisedPage = truncatedSVD(self.page, self.rank)
        self.coefs = leastSquares(self.denoisedPage[:-1].T, self.denoisedPage[-1])

        if self.arOrder is not None:
            assert self.maxOrder is not None
            # Recover the stationary process, then fit AR coefficients for each series
            self.arCoefs = np.zeros((self.maxOrder, self.numSeries))
            extractedNoise = series[:truncatedSteps] - self.denoisedPage.reshape(
                truncatedSteps, -1, order="F"
            )
            for i in range(self.numSeries):
                if self.arOrder[i] > 0:
                    self.arCoefs[-self.arOrder[i] :, i] = learnAR(
                        extractedNoise[:, i], self.arOrder[i]
                    )
                else:
                    self.arCoefs[:, i] = 0

        # Store dataset, with extra space for future time steps
        self.history = np.empty((2 * numSteps, self.numSeries))
        self.history[:numSteps] = series
        self.historyLength = numSteps

        self.fitted = True

        return self

    def update(self, series: NDArray) -> TimeSeriesModel:
        assert self.fitted, "Model not yet fitted!"
        self._check_dims(series)
        numSteps = series.shape[0]
        if numSteps + self.historyLength <= self.history.shape[0]:
            # If there is enough space, just store the new data
            self.history[self.historyLength : self.historyLength + numSteps] = series
        else:
            # Allocate more space to store time series
            expandedLength = 2 * (self.historyLength + numSteps)
            oldHistory, self.history = self.history, np.empty(
                (expandedLength, self.numSeries)
            )
            self.history[: self.historyLength] = oldHistory
            self.history[self.historyLength : self.historyLength + numSteps] = series
        self.historyLength += numSteps
        return self

    def predict(self, numSteps: int) -> NDArray:
        '''
        Forecast the next numSteps for all time series

        Args:
            numSteps (int): horizon to be forecasted

        Returns:
            NDArray: the forecast of size (numSeries x numSteps)
        '''
        # Retrieve the most recent values of the time series to do autoregressive prediction
        contextLength = (
            self.numCoefs + self.maxOrder
            if self.maxOrder is not None
            else self.numCoefs
        )

        fForecastWithContext = np.empty((contextLength + numSteps, self.numSeries))
        fForecastWithContext[:contextLength] = self.history[
            self.historyLength - contextLength : self.historyLength
        ]
        for i in range(contextLength, contextLength + numSteps):
            fForecastWithContext[i] = (
                fForecastWithContext[i - self.numCoefs : i].T @ self.coefs
            )
        fForecast = fForecastWithContext[contextLength:]

        if self.arOrder is not None:
            assert self.maxOrder is not None
            yContext = self.history[
                self.historyLength - contextLength : self.historyLength
            ]
            fImputed = np.empty((contextLength, self.numSeries))
            for idx in range(self.numCoefs, contextLength):
                fImputed[idx] = yContext[idx - self.numCoefs : idx].T @ self.coefs
            xForecastWithContext = np.empty((contextLength + numSteps, self.numSeries))
            xForecastWithContext[:contextLength] = yContext - fImputed
            for idx in range(contextLength, contextLength + numSteps):
                xForecastWithContext[idx] = (
                    xForecastWithContext[idx - self.maxOrder : idx] * self.arCoefs
                ).sum(axis=0)
            xForecast = xForecastWithContext[contextLength:]
            fForecast += xForecast

        return fForecast

