# dckernel

This package implements the two-sample dcMMD and independence dcHSIC tests which are robust against data corruption, as proposed in our paper [Robust Kernel Hypothesis Testing under Data Corruption](https://arxiv.org/abs/2405.19912), as well as robust variants of the differentially private dpMMD and dpHSIC tests of [Differentially Private Permutation Tests: Applications to Kernel Methods](https://arxiv.org/abs/2310.19043) from the [dpkernel](https://github.com/antoninschrab/dpkernel/) repository.

The implementation is in [JAX](https://jax.readthedocs.io/) which can leverage the architecture of GPUs to provide considerable computational speedups.

The experiments of the paper can be reproduced by running the [notebook](https://github.com/antoninschrab/dckernel-paper/blob/main/experiments.ipynb) from the [dckernel-paper](https://github.com/antoninschrab/dckernel-paper/) repository, which also contains demo code showing how to use the tests.

## Installation

The `dckernel` package can be installed by running:
```bash
pip install git+https://github.com/antoninschrab/dckernel.git
```
which relies on the `jax` and `jaxlib` dependencies.

In order to run the tests on GPUs the `cuda` versions of [JAX](https://jax.readthedocs.io/en/latest/installation.html) should be installed as follows
```bash
pip install -U "jax[cuda12]"
```
Otherwise, the CPU version of JAX can be installed as
```bash
pip install -U "jax[cpu]"
```
This can also be run before installing `dckernel`.

## Examples

**Jax compilation:** The first time the dcmmd or dchsic functions are evaluated, JAX compiles them. 
After compilation, they can fastly be evaluated at any other X and Y of the same shape, and any robustness parameter. 
If the functions are given arrays with new shapes, the functions are compiled again.
For details, check out the demo in the [notebook](https://github.com/antoninschrab/dckernel-paper/blob/main/experiments.ipynb) from the [dckernel-paper](https://github.com/antoninschrab/dckernel-paper/) repository.

### dcMMD

**Two-sample testing:** Given arrays X of shape $(m, d)$ and Y of shape $(n, d)$, our dcMMD test `dcMMD(key, X, Y, robustness)` returns 0 if the samples X and Y are believed to come from the same distribution, or 1 otherwise, and is robust up to the corruption of `robustness` number of samples.

```python
# import modules
>>> import jax.numpy as jnp
>>> from jax import random
>>> from dckernel import dcmmd, human_readable_dict

# generate data for two-sample test
>>> key = random.PRNGKey(0)
>>> subkeys = random.split(key, num=2)
>>> X = random.uniform(subkeys[0], shape=(500, 10))
>>> Y = random.uniform(subkeys[1], shape=(500, 10)) + 1

# run dcMMD test
>>> key, subkey = random.split(key)
>>> output = dcmmd(subkey, X, Y, robustness=50)
>>> output
Array(1, dtype=int32)
>>> output.item()
1
>>> output, dictionary = dcmmd(subkey, X, Y, robustness=50, return_dictionary=True)
>>> output
Array(1, dtype=int32)
>>> human_readable_dict(dictionary)
>>> dictionary
{'Bandwidth': 3.1622776985168457,
 'Kernel gaussian': True,
 'Level': 0.05000000074505806,
 'MMD DC-adjusted quantile': 0.16384649276733398,
 'MMD V-statistic': 1.0293232202529907,
 'MMD quantile': 0.050709404051303864,
 'Number of permutations': 500,
 'Robustness': 40,
 'dcMMD test reject': True}
```

### dcHSIC

**Independence testing:** Given paired arrays X of shape $(n, d_X)$ and Y of shape $(n, d_Y)$, our dcHSIC test `dcHSIC(key, X, Y, robustness)` returns 0 if the paired samples X and Y are believed to be independent, or 1 otherwise, and is robust up to the corruption of `robustness` number of samples.

```python
# import modules
>>> import jax.numpy as jnp
>>> from jax import random
>>> from dckernel import dchsic, human_readable_dict

# generate data for independence test 
>>> key = random.PRNGKey(0)
>>> subkeys = random.split(subkey, num=2)
>>> X = random.uniform(subkeys[0], shape=(1000, 10))
>>> X = X.at[:500].set(X[:500] + 10)
>>> Y = X + 0.01 * random.uniform(subkeys[1], shape=(1000, 10))

# run dcHSIC test
>>> key, subkey = random.split(key)
>>> output = dchsic(subkey, X, Y, robustness=40)
>>> output
Array(1, dtype=int32)
>>> output.item()
1
>>> output, dictionary = dchsic(subkey, X, Y, robustness=40, return_dictionary=True)
>>> output
Array(1, dtype=int32)
>>> human_readable_dict(dictionary)
>>> dictionary
{'Bandwidth X': 3.1622776985168457,
 'Bandwidth Y': 3.1622776985168457,
 'HSIC DC-adjusted quantile': 0.34696316719055176,
 'HSIC V-statistic': 0.4259658455848694,
 'HSIC quantile': 0.027283163741230965,
 'Kernel X gaussian': True,
 'Kernel Y gaussian': True,
 'Level': 0.05000000074505806,
 'Number of permutations': 500,
 'Robustness': 40,
 'dcHSIC test reject': True}
```

## Contact

If you have any issues running our dcMMD and dcHSIC tests, please do not hesitate to contact [Antonin Schrab](https://antoninschrab.github.io).

## Affiliations

Centre for Artificial Intelligence, Department of Computer Science, University College London

Gatsby Computational Neuroscience Unit, University College London

Inria London

## Bibtex

```
@unpublished{schrab2024robust,
title={Robust Kernel Hypothesis Testing under Data Corruption}, 
author={Antonin Schrab and Ilmun Kim},
year={2024},
url = {https://arxiv.org/abs/2405.19912},
eprint={2405.19912},
archivePrefix={arXiv},
primaryClass={stat.ML}
}
```

## License

MIT License (see [LICENSE](LICENSE)).

## Related tests

- [mmdagg](https://github.com/antoninschrab/mmdagg/): MMD Aggregated MMDAgg test 
- [ksdagg](https://github.com/antoninschrab/ksdagg/): KSD Aggregated KSDAgg test
- [agginc](https://github.com/antoninschrab/agginc/): Efficient MMDAggInc HSICAggInc KSDAggInc tests
- [mmdfuse](https://github.com/antoninschrab/mmdfuse/): MMD-Fuse test
- [dpkernel](https://github.com/antoninschrab/dpkernel/): Differentially private dpMMD dpHSIC tests
