# Smooth binary mechanism
An implementation of the **smooth binary mechanism** as described in the paper [A Smooth Binary Mechanism for Efficient Private Continual Observation](https://arxiv.org/abs/2306.09666).

## Files included

This repo consists of two files containing code.

### mechanism_noise_generation.py

Contains implementation of the binary mechanism and the smooth binary mechanism (for $\rho$-zCDP). Both are implemented as generators where the $i^{th}$ iterate provides the noise that either mechanism would output at time $t=i$.

### example.ipynb

A Jupyter Notebook containing a simple example where the noise generators are used to release, under $\rho$-zCDP, all prefix sums on a randomly chosen binary input stream.

## Dependencies

Only numpy is needed for including **mechanism_noise_generation.py**. matplotlib in (addition to jupyter notebook itself) is needed to run **example.ipynb**.
