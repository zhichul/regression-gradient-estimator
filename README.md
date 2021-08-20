# 1. Setup/Installation
## 1.1 Method 1
Add `path-to-repo/src` to `PYTHONPATH`.

## 1.2 Method 2
CD into `path-to-repo` and do `pip install -e .` which will install the package `regrad` in place.

# 2. Why does it work?
In short, it turns out that the answer to random design linear regression from Gaussian parameter perturbations `X` around a point `Theta` to its loss `f(X)` of the same Gaussian perturbed parameters gives you the gradient of the expected loss `E[f(X)]`.

[Derivations are Here](https://www.overleaf.com/5496831956djbqzwtrccgb)

# 3. See it in action

`f(x) = round(sin(x))`

<img src="https://github.com/zhichul/regression-gradient-estimator/blob/main/plots/sin.png" alt="f(x) = sin(x)" width="800"/>

`f(x, y) = round(0.1 * (x ** 2 - y ** 2))`

<img src="https://github.com/zhichul/regression-gradient-estimator/blob/main/plots/saddle.png" alt="f(x,y) = x ** 2 - y ** 2" width="800"/>
