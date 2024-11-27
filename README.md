# Adversarial attack

<img src="notebooks/assets/pgd_comparison.png" alt="pgd" width="768" />

This repo contains an implementation of some adversarial attack methods.

## Installation

To install this library, you will need `poetry`. After you have poetry installed, simply run the following command:

```bash
make install
```

For dev installation, run:
```
make install-dev
```

## Methods

The supported methods are:

* Projected Gradient Descent([Madry et al., 2019](https://arxiv.org/abs/1706.06083))
    * Example: [Projected Gradient Descent on ResNet50](./notebooks/pgd.ipynb)