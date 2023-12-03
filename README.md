# SAMoSSA
Implementation of the algorithm described in: SAMoSSA: Multivariate Singular Spectrum Analysis with Stochastic Autoregressive Noise, by Abdullah Alomar, Munther Dahleh, Sean Mann, and Devavrat Shah (published in NeurIPS 2023). See the paper in [here](https://arxiv.org/abs/2305.16491). 

# Getting started

The implementation has a fairly straightforward interface.  First, import SAMoSSA using, 
```python
from samossa import SAMoSSA
```

The algorithm takes as input $T$ observations of a multivariate time series with dimension $N$ in the form of a Numpy array. For example:

```python 
# load data and init model
data = np.load("datasets/electricity/electricity.npy")
T, N = data.shape
```
Then to initialize the model, you would need to specify at least two parameters: `numSeries (int)`: the dimension of the multivariate time series; and `L (int)`: the number of lags used in the linear model (check the example,  the code and the paper for guidance about choice of L).

```python
# choose L such that it is on the order of (sqrt(NT)) if N < T, otherwise, choose it to be on of order ~T (must be less than T in this case!)
L = int(np.sqrt(N*T/4))
model = SAMoSSA(N, L, )
```

Then use `.fit()` to train the model, and `.predict(h)` to forecast the next `h` entries in the time series. 

```python
# fit model on all but the last 24 steps
model.fit(data[:-24,:])
# forecast the next 24 steps!
predictions = model.predict(24)
```