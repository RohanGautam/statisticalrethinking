The following are my attempts across various files:

```python
# %%
import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
import jax
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.spatial.distance import cdist
import time
import seaborn as sns
from tinygp import kernels, GaussianProcess

jax.config.update("jax_enable_x64", True)

print(numpyro.__version__)

# %%
df = pd.read_csv("./NYC.csv")
df

# %% [markdown]
# # Normal GP
#

# %%
def standardize(x: pd.Series):
    return ((x - x.mean()) / x.std()).to_numpy()


x = (df["Age"] / 1000).pipe(standardize)
y = df["RSL"].pipe(standardize)
plt.scatter(x, y, alpha=0.6)

# %% [markdown]
# Let's look at sklearn GP first. Note, alpha can represent y uncertainty as seen [here](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html#example-with-noisy-targets). Without that , it will try to fit every data point resulting in some messy visuals.
#

# %%
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

kernel = RBF(length_scale=0.5, length_scale_bounds=(1e-10, 1.0))
gp = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=10, random_state=0, alpha=1e-2
)
gp.fit(x.reshape(-1, 1), y)

# Make predictions on a fine grid
X_test = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_pred, sigma = gp.predict(X_test, return_std=True)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, c="black", label="Observations")
plt.plot(X_test, y_pred, "r-", label="Prediction")
plt.fill_between(
    X_test.ravel(),
    y_pred - 2 * sigma,
    y_pred + 2 * sigma,
    alpha=0.2,
    color="r",
    label="95% confidence interval",
)
plt.xlabel("Standardized Age")
plt.ylabel("Standardized RSL")
plt.legend()
plt.grid(True)
print(gp.kernel_)


# %% [markdown]
# Let's look at numpyro!
#

# %%
def kernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k


def model(X, Y):
    # set uninformative log-normal priors on our three kernel hyperparameters
    var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
    noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
    length = numpyro.sample("kernel_length", dist.LogNormal(0.0, 10.0))

    # compute kernel
    k = kernel(X, X, var, length, noise)

    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=k),
        obs=Y,
    )


k = NUTS(model)
mcmc = MCMC(k, num_warmup=500, num_samples=2000)
mcmc.run(random.PRNGKey(0), X=x, Y=y)

# %%
posterior = mcmc.get_samples()
sns.kdeplot(posterior["kernel_noise"])

# %%
def predict(rng_key, X, Y, X_test, var, length, noise, use_cholesky=True):
    # compute kernels between train and test data, etc.
    k_pp = kernel(X_test, X_test, var, length, noise, include_noise=True)
    k_pX = kernel(X_test, X, var, length, noise, include_noise=False)
    k_XX = kernel(X, X, var, length, noise, include_noise=True)

    # since K_xx is symmetric positive-definite, we can use the more efficient and
    # stable Cholesky decomposition instead of matrix inversion
    if use_cholesky:
        K_xx_cho = jax.scipy.linalg.cho_factor(k_XX)
        K = k_pp - jnp.matmul(k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, k_pX.T))
        mean = jnp.matmul(k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, Y))
    else:
        K_xx_inv = jnp.linalg.inv(k_XX)
        K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y))

    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), 0.0)) * jax.random.normal(
        rng_key, X_test.shape[:1]
    )

    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean, mean + sigma_noise


vmap_args = (
    random.split(random.PRNGKey(0), posterior["kernel_var"].shape[0]),
    posterior["kernel_var"],
    posterior["kernel_length"],
    posterior["kernel_noise"],
)
x_test = jnp.linspace(-2.5, 1.5, num=500)
means, predictions = jax.vmap(
    lambda rng_key, var, length, noise: predict(
        rng_key,
        x,
        y,
        x_test,
        var,
        length,
        noise,
        use_cholesky=True,
    )
)(*vmap_args)


# %%
plt.scatter(x, y, alpha=0.6)
plt.plot(x_test, np.mean(means, axis=0), color="black")
hpdi = numpyro.diagnostics.hpdi(predictions, prob=0.9, axis=0)
plt.fill_between(x_test, hpdi[0], hpdi[1], alpha=0.2, color="black")

# %% [markdown]
# the noise parameter learnt by numpyro was higher than we set when playing with the sklearn GP, hence its overly smooth. we can see this by getting smooth result with sklearn by setting the `alpha` in the gpregressor to tht of the posterior mean for kernel noise.
#

# %% [markdown]
# # Response (y) noise
#

# %%
def standardize_with_uncertainties(x: pd.Series, x_error: pd.Series):
    """Standardize both values and their uncertainties"""
    x_std = x.std()
    x_mean = x.mean()

    # For the main values: (x - mean) / std
    x_standardized = ((x - x_mean) / x_std).to_numpy()

    # For uncertainties: only divide by std (don't subtract mean)
    x_error_standardized = (x_error / x_std).to_numpy()

    return x_standardized, x_error_standardized


x, x_err = standardize_with_uncertainties(df["Age"] / 1000, df["AgeError"] / 1000)
y, y_err = standardize_with_uncertainties(df["RSL"], df["RSLError"])

plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt="o", markersize=1, alpha=0.6);

# %%
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

kernel = RBF(length_scale=0.5, length_scale_bounds=(1e-10, 10.0))
gp = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=10, random_state=0, alpha=y_err
)
gp.fit(x.reshape(-1, 1), y)

# Make predictions on a fine grid
X_test = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_pred, sigma = gp.predict(X_test, return_std=True)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, c="black", label="Observations")
plt.plot(X_test, y_pred, "r-", label="Prediction")
plt.fill_between(
    X_test.ravel(),
    y_pred - 2 * sigma,
    y_pred + 2 * sigma,
    alpha=0.2,
    color="r",
    label="95% confidence interval",
)
plt.xlabel("Standardized Age")
plt.ylabel("Standardized RSL")
plt.legend()
plt.grid(True)
print(gp.kernel_)


# %% [markdown]
# Again, with numpyro. This time, let's use tinyGP to make inference easier
#

# %%
# def kernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
#     deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
#     k = var * jnp.exp(-0.5 * deltaXsq)
#     if include_noise:
#         k += (noise + jitter) * jnp.eye(X.shape[0])
#     return k

true_t = np.linspace(x.min(), x.max(), 100)


def model(X, yerr, Y=None):
    mean = numpyro.sample("mean", dist.Normal(0.0, 5))
    jitter = numpyro.sample("jitter", dist.HalfNormal(5))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5))
    tau = numpyro.sample("tau", dist.HalfNormal(5))
    kernel = sigma**2 * kernels.ExpSquared(tau)

    gp = GaussianProcess(kernel, X, diag=yerr**2 + jitter, mean=mean)
    numpyro.sample("gp", gp.numpyro_dist(), obs=Y)
    if y is not None:
        numpyro.deterministic("pred", gp.condition(y, true_t).gp.loc)


k = NUTS(model)
mcmc = MCMC(k, num_warmup=500, num_samples=2000)
mcmc.run(random.PRNGKey(0), X=x, Y=y, yerr=y_err)

# %%
mcmc.print_summary()

# %%
# get posterior samples
posterior = mcmc.get_samples()
pred = posterior["pred"].block_until_ready()
pred.shape

# %%
plt.scatter(x, y, alpha=0.6)
hpdi = numpyro.diagnostics.hpdi(pred, prob=0.9, axis=0)
plt.fill_between(true_t, hpdi[0], hpdi[1], alpha=0.2, color="black")

# %% [markdown]
# # EIV
#

# %%
def standardize_with_uncertainties(x: pd.Series, x_error: pd.Series):
    """Standardize both values and their uncertainties"""
    x_std = x.std()
    x_mean = x.mean()

    # For the main values: (x - mean) / std
    x_standardized = ((x - x_mean) / x_std).to_numpy()

    # For uncertainties: only divide by std (don't subtract mean)
    x_error_standardized = (x_error / x_std).to_numpy()

    return x_standardized, x_error_standardized


x, x_err = standardize_with_uncertainties(df["Age"] / 1000, df["AgeError"] / 1000)
y, y_err = standardize_with_uncertainties(df["RSL"], df["RSLError"])

plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt="o", markersize=1, alpha=0.6);

# %%
def linear_reg(x, y):
    alpha = numpyro.sample("alpha", dist.Normal(-1.0, 0.5))
    beta = numpyro.sample("beta", dist.Normal(0, 0.5))
    # error in y
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", alpha + beta * x)
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


k = NUTS(linear_reg)
mcmc = MCMC(k, num_warmup=500, num_samples=2000)
mcmc.run(random.PRNGKey(0), x=x, y=y)
mcmc.print_summary()

posterior = mcmc.get_samples()
means = posterior["mu"]
plt.scatter(x, y, alpha=0.6)
plt.plot(x, np.mean(means, axis=0), color="black")
hpdi = numpyro.diagnostics.hpdi(means, prob=0.9, axis=0)
plt.fill_between(x, hpdi[0], hpdi[1], alpha=0.2, color="black")


# %%
# Get posterior samples
posterior = mcmc.get_samples()

# Create ordered x values for smooth plotting
x_ord = np.sort(x)

# For each posterior sample, compute predictions
y_preds_mean = []  # for mean line
y_preds_full = []  # for full predictive distribution
for i in range(len(posterior["alpha"])):
    # Mean prediction
    y_pred_mean = posterior["alpha"][i] + posterior["beta"][i] * x_ord
    y_preds_mean.append(y_pred_mean)

    # Add noise using sigma for full predictive distribution
    y_pred_full = y_pred_mean + random.normal(random.PRNGKey(i)) * posterior["sigma"][i]
    y_preds_full.append(y_pred_full)

y_preds_mean = np.array(y_preds_mean)
y_preds_full = np.array(y_preds_full)

plt.figure(figsize=(10, 6))

# Plot data points
plt.scatter(x, y, alpha=0.6, color="blue", label="Observations")

# Plot mean prediction line
plt.plot(
    x_ord, np.mean(y_preds_mean, axis=0), "k-", linewidth=2, label="Mean prediction"
)

# Add uncertainty bands for mean
mean_hpdi = numpyro.diagnostics.hpdi(y_preds_mean, prob=0.9)
plt.fill_between(
    x_ord, mean_hpdi[0], mean_hpdi[1], color="gray", alpha=0.2, label="90% HPDI (mean)"
)

# Add predictive intervals
pred_hpdi = numpyro.diagnostics.hpdi(y_preds_full, prob=0.9)
plt.fill_between(
    x_ord,
    pred_hpdi[0],
    pred_hpdi[1],
    color="blue",
    alpha=0.1,
    label="90% HPDI (predictive)",
)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# %%
# def linear_reg_eiv(x, y, x_err, y_err):
#     alpha = numpyro.sample("alpha", dist.Normal(-1.0, 0.5))
#     beta = numpyro.sample("beta", dist.Normal(0, 0.5))

#     x_true = numpyro.sample("x_true", dist.Normal(0, 1).expand([len(x)]))
#     numpyro.sample("x_obs", dist.Normal(x_true, x_err), obs=x)

#     sigma = numpyro.sample("sigma", dist.Normal(0.01, 1))
#     mu = numpyro.sample("mu", dist.Normal(alpha + beta * x_true, sigma))
#     numpyro.sample("y", dist.Normal(mu, y_err), obs=y)


def linear_reg_eiv(x, y, x_err, y_err):
    D = jnp.c_[x, y]

    alpha = numpyro.sample("alpha", dist.Normal(-1.0, 0.5))
    beta = numpyro.sample("beta", dist.Normal(0, 0.5))
    x_true = numpyro.sample("x_true", dist.Normal(0, np.sqrt(1000)).expand([len(x)]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))

    def get_cov(x_err_i, y_err_i):
        # add additional microscale variance to y
        cov = jnp.array(
            [
                [x_err_i**2, 0],
                [0, y_err_i**2 + sigma**2],
            ]
        )
        return cov

    cov = jax.vmap(get_cov)(x_err, y_err)

    mu_y = numpyro.deterministic("mu_y", alpha + beta * x_true)
    means = jnp.column_stack([x_true, mu_y])

    # multivariate normal allows correlations/cov between x and y via the kernel
    with numpyro.plate("observation", len(D)):
        numpyro.sample(
            "z",
            dist.MultivariateNormal(means, cov),
            obs=D,
        )


k = NUTS(linear_reg_eiv)
mcmc = MCMC(k, num_warmup=500, num_samples=2000)
mcmc.run(random.PRNGKey(0), x=x, y=y, x_err=x_err, y_err=y_err)
mcmc.print_summary()

posterior = mcmc.get_samples()

# %%
# means = posterior["mu_y"] # this uses x_true which can vary randomly
means = posterior["alpha"][:, None] + posterior["beta"][:, None] * x


s_idx = jnp.argsort(x)
plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt="o", markersize=1, alpha=0.6)
plt.plot(x[s_idx], np.mean(means, axis=0)[s_idx], color="black")
hpdi = numpyro.diagnostics.hpdi(means, prob=0.9, axis=0)
plt.fill_between(x[s_idx], hpdi[0][s_idx], hpdi[1][s_idx], alpha=0.2, color="black")

# %%
# Get the posterior samples
posterior = mcmc.get_samples()

# Create a grid of x values for smooth plotting
x_grid = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)

# For each posterior sample, compute the predicted y AND add random noise from sigma
y_preds_mean = []  # for the mean line
y_preds_full = []  # for the full predictive distribution
for i in range(len(posterior["alpha"])):
    # Mean prediction
    y_pred_mean = posterior["alpha"][i] + posterior["beta"][i] * x_grid
    y_preds_mean.append(y_pred_mean)

    # Add noise using sigma for full predictive distribution
    y_pred_full = y_pred_mean + random.normal(random.PRNGKey(i)) * posterior["sigma"][i]
    y_preds_full.append(y_pred_full)

y_preds_mean = np.array(y_preds_mean)
y_preds_full = np.array(y_preds_full)

plt.figure(figsize=(10, 6))

# Plot data points with error bars
plt.errorbar(
    x, y, xerr=x_err, yerr=y_err, fmt="o", color="blue", alpha=0.3, markersize=3
)

# Plot mean prediction line
plt.plot(
    x_grid, np.mean(y_preds_mean, axis=0), "k-", linewidth=2, label="Mean prediction"
)

# Add uncertainty bands for mean
mean_hpdi = numpyro.diagnostics.hpdi(y_preds_mean, prob=0.9)
plt.fill_between(
    x_grid, mean_hpdi[0], mean_hpdi[1], color="gray", alpha=0.2, label="90% HPDI (mean)"
)

# Add predictive intervals
pred_hpdi = numpyro.diagnostics.hpdi(y_preds_full, prob=0.9)
plt.fill_between(
    x_grid,
    pred_hpdi[0],
    pred_hpdi[1],
    color="blue",
    alpha=0.1,
    label="90% HPDI (predictive)",
)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# %% [markdown]
# main difference when introducing errors in variables here - the hpdi is narrower, because there are more possibilities for x now, within it's uncertainty ranges!
#

# %% [markdown]
# let's now do the same with a GP!
#

# %%
x_test = np.linspace(x.min(), x.max(), 100)


def gp_eiv(x, x_err, y, y_err):
    mean = numpyro.sample("mean", dist.Normal(0.0, 5))
    jitter = numpyro.sample("jitter", dist.HalfNormal(5))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5))
    tau = numpyro.sample("tau", dist.HalfNormal(5))
    kernel = sigma**2 * kernels.ExpSquared(tau)

    gp = GaussianProcess(kernel, x, diag=y_err**2 + jitter, mean=mean)
    numpyro.sample("gp", gp.numpyro_dist(), obs=y)
    if y is not None:
        numpyro.deterministic("pred", gp.condition(y, x_test).gp.loc)


k = NUTS(gp_eiv)
mcmc = MCMC(k, num_warmup=500, num_samples=2000)
mcmc.run(random.PRNGKey(0), x=x, y=y, y_err=y_err, x_err=x_err)
mcmc.print_summary()
# viz
posterior = mcmc.get_samples()
pred = posterior["pred"].block_until_ready()
print(pred.shape)
plt.scatter(x, y, alpha=0.6)
hpdi = numpyro.diagnostics.hpdi(pred, prob=0.9, axis=0)
plt.fill_between(x_test, hpdi[0], hpdi[1], alpha=0.2, color="black")

# %% [markdown]
# # EIV IGP
#

# %%
plt.figure(figsize=(7, 4))
plt.errorbar(
    df["Age"],
    df["RSL"],
    xerr=df["AgeError"],
    yerr=df["RSLError"],
    fmt="o",
    markersize=1,
    alpha=0.6,
);

# %%
# We're not accounting for GIA to begin with
year_of_correction = 2010
convert_bp_to_ce = False
interval = 30  # years
kappa = 1.99


# preprocessing

df["x"] = (1950 - df["Age"]) if convert_bp_to_ce else df["Age"]
df["y"] = df["RSL"]
df["var_x"] = (df["AgeError"] / 1000) ** 2  # x in thousands of years
df["var_y"] = df["RSLError"] ** 2
# variance and precision matrices
V = np.empty((2, 2, len(df)))
P = np.empty((2, 2, len(df)))
for i in range(len(df)):
    V[:, :, i] = np.array(
        [
            [df["var_x"].iloc[i], 0.0],
            [0.0, df["var_y"].iloc[i]],
        ]
    )
    P[:, :, i] = np.linalg.inv(V[:, :, i])

# include error bounds by default
x_max = (df["x"] + df["AgeError"]).max() / 1000
x_min = (df["x"] - df["AgeError"]).min() / 1000
x_grid = np.concatenate(
    [
        [x_min],  # scalar
        np.arange(min(df["x"] / 1000), max(df["x"] / 1000), interval / 1000),
        [x_max],  # scalar
    ]
)
x = (df["x"] / 1000) - min(df["x"] / 1000)
x_star = x_grid - min(df["x"] / 1000)
Ngrid = len(x_star)
# add axis to x_star
Dist = cdist(x_star[:, None], x_star[:, None], metric="euclidean")
D = np.c_[x, df["y"]]

## quadrature init
L = 30
index = np.arange(1, L + 1)
# cosfunc = np.cos(np.pi * (index + 0.5) / L)
cosfunc = np.cos((np.pi * (2 * index - 1)) / (2 * L))
# plt.scatter(range(len(cosfunc)), cosfunc)
quad1 = np.zeros((len(df), Ngrid, L))
quad2 = np.ones((len(df), Ngrid, L))

for j in range(Ngrid):
    for k in range(len(df)):
        quad1[k, j, :] = np.abs((x[k] * cosfunc / 2) + (x[k] / 2) - x_star[j]) ** kappa
        quad2[k, j, :] = (x[k] / 2) * (np.pi / L) * (np.sqrt(1 - cosfunc**2))


V = V.transpose(2, 0, 1)
jags_data = {
    "V": V,
    "P_t": P[:, :].transpose(2, 0, 1),
    "n": len(df),  # c
    "m": Ngrid,  # c
    "P": P,  # c
    "D": D,  # c
    "L": L,  # c
    "ppi": np.pi,  # c
    "cosfunc": cosfunc,  # c
    "Dist": Dist,  # c
    "xstar": x_star,  # c
    "quad1": quad1,  # c
    "quad2": quad2,  # c
    "kappa": kappa,  # c
    "cor.p": 0.2,  # c
}


# %%
V = jnp.array(V)
D = jnp.array(D)
Dist = jnp.array(Dist)
quad1 = jnp.array(quad1)
quad2 = jnp.array(quad2)

# %%
m = Ngrid
K = jnp.zeros((m, m))
# Add small constant to diagonal for numerical stability
K = K.at[jnp.diag_indices(m)].set(1 + 1e-5)
# Off-diagonal elements using powered exponential covariance
i, j = jnp.triu_indices(m, k=1)
triu_values = 0.1 ** ((Dist[i, j] / 0.5) ** 1.5)
K = K.at[i, j].set(triu_values)
K = K.at[j, i].set(triu_values)
plt.imshow(K)
plt.colorbar()

# %%
def eiv_igp(D=D, V=V, m=Ngrid, Dist=Dist, kappa=kappa, quad1=quad1, quad2=quad2):
    # print("hi")
    # x_true = numpyro.sample("x_true", dist.Normal(0, np.sqrt(1e3)).expand([len(D)]))
    x_true = numpyro.sample("x_true", dist.Normal(0, np.sqrt(1e3)).expand([len(D)]))
    beta0 = numpyro.sample("beta0", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0.01, 1))
    p = numpyro.sample("p", dist.Uniform(0, 1))
    tau_g = numpyro.sample("tau_g", dist.Gamma(10.0, 100.0))
    # gp_len = numpyro.sample("gp_len", dist.LogNormal(0, 10))

    # Construct the GP kernel
    K = jnp.zeros((m, m))
    K = K.at[jnp.diag_indices(m)].set(1 + 1e-5)
    i, j = jnp.triu_indices(m, k=1)
    triu_values = p ** ((Dist[i, j]) ** kappa)
    K = K.at[i, j].set(triu_values)
    K = K.at[j, i].set(triu_values)  # mirror to lower triangle

    # precision is just inverse of covariance
    # K_inv = jnp.linalg.inv((1 / tau_g) * K)
    K = numpyro.deterministic("K", K / tau_g)
    w_m = numpyro.sample(
        "w_m",
        dist.MultivariateNormal(jnp.zeros(m), K),
    )

    K_gw = jnp.sum(p**quad1 * quad2, axis=2)
    K_w_inv = jnp.linalg.inv(K)
    w_tilde_m = jnp.matmul(K_gw, jnp.matmul(K_w_inv, w_m))

    mu_y = numpyro.deterministic("mu_y", beta0 + w_tilde_m)

    # if jnp.isnan(K).any():
    #     print("NaN values detected in mu_y")

    mu = jnp.c_[
        x_true,
        mu_y,
    ]
    V_new = V.at[:, 1, 1].set(V[:, 1, 1] + sigma**2)

    with numpyro.plate("observations", len(D)):
        numpyro.sample(
            "z",
            dist.MultivariateNormal(
                mu,
                V_new,
            ),
            obs=D,
        )


pp = Predictive(eiv_igp, num_samples=100)
prior_predictive = pp(random.PRNGKey(0))
print(prior_predictive.keys())
plt.scatter(D[:, 0], D[:, 1], alpha=0.5)
mu = prior_predictive["mu_y"]
s_idx = jnp.argsort(D[:, 0])
for i in range(len(mu)):
    plt.plot(D[:, 0][s_idx], mu[i][s_idx])
    # break

# %%
k = NUTS(
    eiv_igp,
    init_strategy=numpyro.infer.init_to_median,
)

mcmc = MCMC(k, num_warmup=500, num_samples=2000)
mcmc.run(random.PRNGKey(0))
mcmc.print_summary()

posterior = mcmc.get_samples()

# %%
import seaborn as sns

sns.kdeplot(posterior["p"])

# %%
idx = np.random.randint(0, len(posterior["K"]) - 1)
# print(f"Showing covariance matrix for sample {idx}")
plt.imshow(posterior["K"][idx])

# %%
plt.scatter(D[:, 0], D[:, 1], alpha=0.5)
mu = posterior["mu_y"]
s_idx = jnp.argsort(D[:, 0])
for i in range(len(mu)):
    plt.plot(D[:, 0][s_idx], mu[i][s_idx])
    break




```

```python
# %%
import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
import jax
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.spatial.distance import cdist
import time
import seaborn as sns
import pickle
import os

jax.config.update("jax_enable_x64", True)
print(numpyro.__version__)

# %%
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

# %%
print(jax.local_device_count())

# %%
df = pd.read_csv("./NYC.csv")
plt.errorbar(
    df["Age"],
    df["RSL"],
    xerr=df["AgeError"],
    yerr=df["RSLError"],
    fmt="o",
    markersize=1,
    alpha=0.6,
)
plt.title("New york proxy data")
plt.xlabel("Age (CE)")
plt.xlabel("RSL (m)")
plt.show()

# %% [markdown]
# No GIA correction
#

# %%
# import jax.scipy.spatial


# interval = 30  # years
# kappa = 1.99
# L = 30  # number of grid points

# x = jnp.array(df["Age"] / 1000)
# y = jnp.array(df["RSL"])
# var_x = jnp.array(df["AgeError"] / 1000) ** 2
# var_y = jnp.array(df["RSLError"]) ** 2


# def make_cov_matrix(vx, vy):
#     return jnp.array([[vx, 0.0], [0.0, vy]])


# V = jax.vmap(make_cov_matrix)(var_x, var_y)

# # for numerical integration
# x_max = max(x + jnp.sqrt(var_x))
# x_min = min(x - jnp.sqrt(var_x))
# x_grid = jnp.r_[x_min, jnp.arange(min(x), max(x), interval / 1000), x_max]
# x_star = x_grid - min(x)
# Ngrid = len(x_star)
# Dist = jnp.array(cdist(x_star[:, None], x_star[:, None], metric="euclidean"))
# D = np.c_[x - min(x), y]
# index = jnp.arange(1, L + 1)
# cosfunc = jnp.cos((jnp.pi * (2 * index - 1)) / (2 * L))


# quad1 = np.zeros((len(df), Ngrid, L))
# quad2 = np.ones((len(df), Ngrid, L))

# for j in range(Ngrid):
#     for k in range(len(df)):
#         quad1[k, j, :] = np.abs((x[k] * cosfunc / 2) + (x[k] / 2) - x_star[j]) ** kappa
#         quad2[k, j, :] = (x[k] / 2) * (np.pi / L) * (np.sqrt(1 - cosfunc**2))

# quad1 = jnp.array(quad1)
# quad2 = jnp.array(quad2)

# %%
# We're not accounting for GIA to begin with
year_of_correction = 2010
convert_bp_to_ce = False
interval = 30  # years
kappa = 1.99


# preprocessing

df["x"] = (1950 - df["Age"]) if convert_bp_to_ce else df["Age"]
df["y"] = df["RSL"]
df["var_x"] = (df["AgeError"] / 1000) ** 2  # x in thousands of years
df["var_y"] = df["RSLError"] ** 2
# variance and precision matrices
V = np.empty((2, 2, len(df)))
P = np.empty((2, 2, len(df)))
for i in range(len(df)):
    V[:, :, i] = np.array(
        [
            [df["var_x"].iloc[i], 0.0],
            [0.0, df["var_y"].iloc[i]],
        ]
    )
    P[:, :, i] = np.linalg.inv(V[:, :, i])

# include error bounds by default
x_max = (df["x"] + df["AgeError"]).max() / 1000
x_min = (df["x"] - df["AgeError"]).min() / 1000
x_grid = np.concatenate(
    [
        [x_min],  # scalar
        np.arange(min(df["x"] / 1000), max(df["x"] / 1000), interval / 1000),
        [x_max],  # scalar
    ]
)
x = (df["x"] / 1000) - min(df["x"] / 1000)
x_star = x_grid - min(df["x"] / 1000)
Ngrid = len(x_star)
# add axis to x_star
Dist = cdist(x_star[:, None], x_star[:, None], metric="euclidean")
D = np.c_[x, df["y"]]

## quadrature init
L = 30
index = np.arange(1, L + 1)
# cosfunc = np.cos(np.pi * (index + 0.5) / L)
cosfunc = np.cos((np.pi * (2 * index - 1)) / (2 * L))
# plt.scatter(range(len(cosfunc)), cosfunc)
quad1 = np.zeros((len(df), Ngrid, L))
quad2 = np.ones((len(df), Ngrid, L))

for j in range(Ngrid):
    for k in range(len(df)):
        quad1[k, j, :] = np.abs((x[k] * cosfunc / 2) + (x[k] / 2) - x_star[j]) ** kappa
        quad2[k, j, :] = (x[k] / 2) * (np.pi / L) * (np.sqrt(1 - cosfunc**2))

V = V.transpose(2, 0, 1)
V = jnp.array(V)
D = jnp.array(D)
Dist = jnp.array(Dist)
quad1 = jnp.array(quad1)
quad2 = jnp.array(quad2)

# %%
def eiv_igp(D=D, V=V, m=Ngrid, Dist=Dist, kappa=kappa, quad1=quad1, quad2=quad2):
    N = len(V)
    x_true = numpyro.sample("x_true", dist.Normal(0, np.sqrt(1e3)).expand([N]))
    beta0 = numpyro.sample("beta0", dist.Normal(0, 1))
    # sigma.y in the jags code
    sigma = numpyro.sample("sigma", dist.Uniform(0.01, 1))
    p = numpyro.sample("p", dist.Uniform(0, 1))
    tau_g = numpyro.sample("tau_g", dist.Gamma(10.0, 100.0))
    # just to keep track, will be used later
    sigma_g = numpyro.deterministic("sigma_g", jnp.pow(tau_g, -0.5))

    # Construct the GP kernel
    K = jnp.zeros((m, m))
    K = K.at[jnp.diag_indices(m)].set(1 + 1e-5)
    i, j = jnp.triu_indices(m, k=1)
    triu_values = p ** ((Dist[i, j]) ** kappa)
    K = K.at[i, j].set(triu_values)
    K = K.at[j, i].set(triu_values)  # mirror to lower triangle
    K = numpyro.deterministic("K", K / tau_g)
    w_m = numpyro.sample(
        "w_m",
        dist.MultivariateNormal(jnp.zeros(m), K),
    )

    K_gw = jnp.sum(p**quad1 * quad2, axis=2)
    K_w_inv = numpyro.deterministic("K_w_inv", jnp.linalg.inv(K))
    w_tilde_m = jnp.matmul(K_gw, jnp.matmul(K_w_inv, w_m))

    mu = jnp.c_[
        x_true,
        beta0 + w_tilde_m,
    ]
    mu = numpyro.deterministic("mu", mu)
    V_new = V.at[:, 1, 1].set(V[:, 1, 1] + sigma**2)

    with numpyro.plate("observations", N):
        numpyro.sample(
            "z",
            dist.MultivariateNormal(
                mu,
                V_new,
            ),
            obs=D,
        )


# %%
rng_key = random.PRNGKey(0)

# %%
rng_key, rng_key_ = random.split(rng_key)
pp = Predictive(eiv_igp, num_samples=100)
prior_predictive = pp(rng_key_)
print(prior_predictive.keys())
plt.scatter(D[:, 0], D[:, 1], alpha=0.5)
# print(prior_predictive["mu"].shape)
mu = prior_predictive["mu"][:, :, 1]
s_idx = jnp.argsort(D[:, 0])
for i in range(len(mu)):
    plt.plot(D[:, 0][s_idx], mu[i][s_idx])

# %%
numpyro.set_host_device_count(2)
rng_key, rng_key_ = random.split(rng_key)

k = NUTS(
    eiv_igp,
    init_strategy=numpyro.infer.init_to_median,
)

mcmc = MCMC(
    k,
    num_warmup=5_000,
    num_samples=20_000,
    thinning=10,
    # num_chains=2,
    # chain_method="parallel",
)
mcmc.run(rng_key_)
mcmc.print_summary()

# posterior = mcmc.get_samples()


with open("mcmc_results.pkl", "wb") as f:
    pickle.dump(mcmc, f)

# %%
# import pickle

# with open("mcmc_results.pkl", "wb") as f:
#     pickle.dump(mcmc, f)


# %%
# with open("mcmc_results.pkl", "rb") as f:
#     mcmc: MCMC = pickle.load(f)

# posterior = mcmc.get_samples()
# mcmc.print_summary()
# print(posterior.keys())


# %%
posterior = mcmc.get_samples()

# %%
sns.kdeplot(posterior["beta0"])

# %%
plt.scatter(D[:, 0], D[:, 1], alpha=0.5)
mu = posterior["mu"][:, :, 1]
s_idx = jnp.argsort(D[:, 0])

plt.plot(D[:, 0][s_idx], mu.mean(axis=0)[s_idx])

hpdi = numpyro.diagnostics.hpdi(mu, prob=0.9, axis=0)
plt.fill_between(x, hpdi[0], hpdi[1], alpha=0.2, color="black")

# %%
rng_key, rng_key_ = random.split(rng_key)
predictive = Predictive(eiv_igp, posterior)(rng_key_, D=None)

# %%
plt.scatter(D[:, 0], D[:, 1], alpha=0.5)
yhat = predictive["z"][:, :, 1]
plt.plot(D[:, 0][s_idx], mu.mean(axis=0)[s_idx])
hpdi = numpyro.diagnostics.hpdi(yhat, prob=0.9, axis=0)
plt.fill_between(
    D[:, 0][s_idx], hpdi[0][s_idx], hpdi[1][s_idx], alpha=0.2, color="black"
)
# print(yhat.shape)
# for i in range(100):
#     plt.plot(D[:, 0][s_idx], yhat[i][s_idx], c="b", alpha=0.1)

# %%
quad1.shape

# %%
# n_iter = posterior["beta0"].shape[0]
# Kgw = np.zeros((n_iter, Ngrid, Ngrid))
# K = np.zeros((n_iter, Ngrid, Ngrid))
# Kwinv = np.zeros((n_iter, Ngrid, Ngrid))


quad1n = np.zeros((Ngrid, Ngrid, L))
quad2n = np.ones((Ngrid, Ngrid, L))
for j in range(Ngrid):
    for k in range(Ngrid):
        quad1n[k, j, :] = np.abs((x[k] * cosfunc / 2) + (x[k] / 2) - x_star[j]) ** kappa
        quad2n[k, j, :] = (x[k] / 2) * (np.pi / L) * (np.sqrt(1 - cosfunc**2))


# for i in range(n_iter):
#     for j in range(Ngrid):
#         for k in range(Ngrid):
#             Kgw[i, j, k] = np.sum(posterior["p"][i] ** quad1[j, k, :] * quad2[j, k, :])

# broadcasting+vmap is badass
Kgw = np.sum(
    posterior["p"][:, None, None, None] ** quad1n[None, :, :, :]
    * quad2n[None, :, :, :],
    axis=-1,
)
K = posterior["p"][:, None, None] ** (Dist[None, :, :] * kappa)

Kwinv = jax.vmap(jnp.linalg.inv)(K)

# Get predictions using jax operations
pred = posterior["beta0"][:, None] + jnp.einsum(
    "ijk,ikl,il->ij", Kgw, Kwinv, posterior["w_m"]
)


# %%
# Convert predictions to long format DataFrame
pred_df = pd.DataFrame(pred, columns=x_star)
pred_long = pd.melt(
    pred_df, value_vars=pred_df.columns, var_name="year", value_name="value"
)
pred_long["year"] = pred_long["year"].astype(float)

# sl estimates

# Group by year and calculate summary statistics
sl_estimates = (
    pred_long.groupby("year")
    .agg(
        {
            "value": [
                "mean",
                lambda x: np.mean(x) - 2 * np.std(x),
                lambda x: np.mean(x) + 2 * np.std(x),
            ]
        }
    )
    .reset_index()
)

# Rename columns
sl_estimates.columns = ["year", "SL_est", "SL_lwr", "SL_upr"]


# %%
# Plot SL estimates
plt.figure(figsize=(10, 6))
plt.plot(sl_estimates["year"], sl_estimates["SL_est"], "b-", label="Estimate")
plt.fill_between(
    sl_estimates["year"],
    sl_estimates["SL_lwr"],
    sl_estimates["SL_upr"],
    color="b",
    alpha=0.5,
)


plt.xlabel("Year")
plt.ylabel("Sea Level (m)")
plt.title("Sea Level Estimates")
plt.legend()
plt.grid(True, alpha=0.3)


# %%
dydt = posterior["w_m"]
mean_dydt = jnp.mean(dydt, axis=1)
sd_rate = np.std(mean_dydt)
u95 = np.quantile(mean_dydt, 0.975)
l95 = np.quantile(mean_dydt, 0.025)

# mean_dydt.shape


# %%
# Get predictions
pred = posterior["beta0"][:, None] + np.einsum(
    "ijk,ikl,il->ij", Kgw, Kwinv, posterior["w_m"]
)

# Calculate mean and confidence intervals for predictions
sl_est = np.mean(pred, axis=0)
sl_sd = np.std(pred, axis=0)
sl_lwr = sl_est - 2 * sl_sd
sl_upr = sl_est + 2 * sl_sd

# Calculate mean and confidence intervals for rates
rate_est = np.mean(dydt, axis=0)
rate_sd = np.std(dydt, axis=0)
rate_lwr = rate_est - 2 * rate_sd
rate_upr = rate_est + 2 * rate_sd

# Create plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot SL estimates
ax1.plot(x_star, sl_est, "b-", label="Estimate")
ax1.fill_between(x_star, sl_lwr, sl_upr, color="b", alpha=0.2, label="95% CI")
# ax1.scatter(D[:, 0], D[:, 1], c="k", s=20, alpha=0.5, label="Observations")
# ax1.fill_between(D[:, 0], D[:, 1] - 2 * , y_obs + 2 * sigma_y, color="k", alpha=0.1)
ax1.set_xlabel("Year")
ax1.set_ylabel("Sea Level (m)")
ax1.set_title("Sea Level Estimates")
ax1.legend()

# Plot rates
ax2.plot(x_star, rate_est, "r-", label="Rate Estimate")
ax2.fill_between(x_star, rate_lwr, rate_upr, color="r", alpha=0.2, label="95% CI")
ax2.set_xlabel("Year")
ax2.set_ylabel("Rate (mm/yr)")
ax2.set_title("Sea Level Rates")
ax2.legend()

plt.tight_layout()
plt.show()

# Print mean rate and confidence interval
print(f"Mean rate: {np.mean(mean_dydt):.2f} mm/yr")
print(f"95% CI: [{l95:.2f}, {u95:.2f}] mm/yr")


# %%
posterior = mcmc.get_samples()


# %% [markdown]
# # SVI
#

# %%
model = eiv_igp
guide = numpyro.infer.autoguide.AutoDiagonalNormal(
    model, init_loc_fn=numpyro.infer.autoguide.init_to_median
)
optimizer = numpyro.optim.Adam(step_size=1e-2)
svi = numpyro.infer.SVI(model, guide, optimizer, loss=numpyro.infer.Trace_ELBO())


svi_result = svi.run(random.PRNGKey(0), 10000)

# %%
svi_result.params.keys()

# %%
plt.plot(svi_result.losses)
plt.yscale("log")


# %%
num_samples = 1000
rng_key = random.PRNGKey(1)
params = guide.sample_posterior(rng_key, svi_result.params, sample_shape=(num_samples,))


# %%
plt.scatter(D[:, 0], D[:, 1])
mu = params["mu"][:, :, 1]
s_idx = np.argsort(D)
print(mu.shape)
# plt.plot(D[:, 0][s_idx], np.mean(mu, axis=0)[s_idx])


```

```python
# %% [markdown]
# # some research:
#
# - can use the jax odeint(rungekatta) instead of chebyshev-gauss quadrature for better differentiability
# - can use tinygp for kernels?
# - let's try 3 things here. GP, EIV GP, and EIV IGP
#

# %%
import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
import jax
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.spatial.distance import cdist
import time
import seaborn as sns
from tinygp import kernels, GaussianProcess

jax.config.update("jax_enable_x64", True)

print(numpyro.__version__)

# %%
df = pd.read_csv("./NYC.csv")
df

# %%
def standardize_with_uncertainties(x: pd.Series, x_error: pd.Series):
    """Standardize both values and their uncertainties"""
    x_std = x.std()
    x_mean = x.mean()

    # For the main values: (x - mean) / std
    x_standardized = ((x - x_mean) / x_std).to_numpy()

    # For uncertainties: only divide by std (don't subtract mean)
    x_error_standardized = (x_error / x_std).to_numpy()

    return x_standardized, x_error_standardized


x, x_err = standardize_with_uncertainties(df["Age"] / 1000, df["AgeError"] / 1000)
y, y_err = standardize_with_uncertainties(df["RSL"], df["RSLError"])


# %%
plt.scatter(x, y)

# %% [markdown]
# ## GP
#

# %%
true_t = np.linspace(x.min(), x.max(), 100)


def model(x, y, y_err):
    mean = numpyro.sample("mean", dist.Normal(0.0, 1))
    jitter = numpyro.sample("jitter", dist.HalfNormal(1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    tau = numpyro.sample("tau", dist.HalfNormal(1))
    # distance = numpyro.sample("dist", dist.HalfNormal(0.2))
    kernel = sigma**2 * kernels.ExpSquared(tau)

    gp = GaussianProcess(kernel, x, mean=mean)
    # numpyro.sample("gp", gp.numpyro_dist(), obs=y)
    numpyro.sample("gp", gp.numpyro_dist())
    # Always compute predictions first
    # pred_gp = GaussianProcess(kernel, true_t, mean=mean)
    # numpyro.deterministic("pred", pred_gp.numpyro_dist())
    # if y is not None:
    #     numpyro.deterministic("pred", gp.condition(y, true_t).gp.loc)


# Sample from prior predictive
prior = Predictive(model, num_samples=100)
prior_samples = prior(random.PRNGKey(1), x=true_t, y=None, y_err=y_err)
prior_pred = prior_samples["gp"]
print(prior_pred.shape)

# Plot prior predictive samples
plt.figure(figsize=(10, 6))
for i in range(100):
    plt.plot(true_t, prior_pred[i], color="blue", alpha=0.1)
plt.scatter(x, y, alpha=0.6, color="black")
plt.title("Prior Predictive Samples")
plt.show()


# %%
mcmc = MCMC(NUTS(model), num_warmup=100, num_samples=200)
mcmc.run(random.PRNGKey(0), x=x, y=y, y_err=y_err)
mcmc.print_summary()

# %%
posterior = mcmc.get_samples()
pred = posterior["pred"].block_until_ready()
plt.scatter(x, y, alpha=0.6)
hpdi = numpyro.diagnostics.hpdi(pred, prob=0.9, axis=0)
plt.fill_between(true_t, hpdi[0], hpdi[1], alpha=0.2, color="black")

# %%
import numpy as np
import matplotlib.pyplot as plt

random = np.random.default_rng(42)

t = np.sort(
    np.append(
        random.uniform(0, 3.8, 28),
        random.uniform(5.5, 10, 18),
    )
)
yerr = random.uniform(0.08, 0.22, len(t))
y = (
    0.2 * (t - 5)
    + np.sin(3 * t + 0.1 * (t - 5) ** 2)
    + yerr * random.normal(size=len(t))
)

true_t = np.linspace(0, 10, 100)
true_y = 0.2 * (true_t - 5) + np.sin(3 * true_t + 0.1 * (true_t - 5) ** 2)

plt.plot(true_t, true_y, "k", lw=1.5, alpha=0.3)
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.xlabel("x [day]")
plt.ylabel("y [ppm]")
plt.xlim(0, 10)
plt.ylim(-2.5, 2.5)
_ = plt.title("simulated data")
plt.show()

prior_sigma = 5.0


def numpyro_model(t, yerr, y=None):
    mean = numpyro.sample("mean", dist.Normal(0.0, prior_sigma))
    jitter = numpyro.sample("jitter", dist.HalfNormal(prior_sigma))

    sigma1 = numpyro.sample("sigma1", dist.HalfNormal(prior_sigma))
    rho1 = numpyro.sample("rho1", dist.HalfNormal(prior_sigma))
    tau = numpyro.sample("tau", dist.HalfNormal(prior_sigma))
    kernel1 = sigma1**2 * kernels.ExpSquared(tau) * kernels.Cosine(rho1)

    sigma2 = numpyro.sample("sigma2", dist.HalfNormal(prior_sigma))
    rho2 = numpyro.sample("rho2", dist.HalfNormal(prior_sigma))
    kernel2 = sigma2**2 * kernels.Matern32(rho2)

    kernel = kernel1 + kernel2
    # we can specify mean function here
    gp = GaussianProcess(kernel, t, diag=yerr**2 + jitter, mean=mean)
    numpyro.sample("gp", gp.numpyro_dist(), obs=y)

    if y is not None:
        numpyro.deterministic("pred", gp.condition(y, true_t).gp.loc)


nuts_kernel = NUTS(numpyro_model, dense_mass=True, target_accept_prob=0.9)
mcmc = MCMC(
    nuts_kernel,
    num_warmup=1000,
    num_samples=1000,
    num_chains=1,
    # progress_bar=False,
)
rng_key = jax.random.PRNGKey(34923)

# %%
mcmc.run(rng_key, t, yerr, y=y)
samples = mcmc.get_samples()
pred = samples["pred"].block_until_ready()

# %%
q = np.percentile(pred, [5, 50, 95], axis=0)
plt.fill_between(true_t, q[0], q[2], color="C0", alpha=0.5, label="inference")
plt.plot(true_t, q[1], color="C0", lw=2)
plt.plot(true_t, true_y, "k", lw=1.5, alpha=0.3, label="truth")

plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.xlabel("x [day]")
plt.ylabel("y [ppm]")
plt.xlim(0, 10)
plt.ylim(-2.5, 2.5)
plt.legend()

# %% [markdown]
#

# %%
x, x_err = standardize_with_uncertainties(df["Age"] / 1000, df["AgeError"] / 1000)
y, y_err = standardize_with_uncertainties(df["RSL"], df["RSLError"])
test_x = np.linspace(x.min(), x.max(), 100)


def gp_model(x, y, y_err, test_x):
    # jitter = numpyro.sample(
    #     "jitter", dist.HalfNormal(1)
    # )  # similar to microscale variance mentioned in paper
    scale = numpyro.sample("sigma", dist.HalfNormal(1))
    kernel = kernels.ExpSquared(scale)
    gp = GaussianProcess(kernel, x, diag=y_err**2)
    # gp = GaussianProcess(kernel, x, diag=1e-1)
    numpyro.sample("gp", gp.numpyro_dist(), obs=y)

    if y is not None:
        conditioned_gp = gp.condition(y, test_x).gp
        numpyro.deterministic("pred", conditioned_gp.loc)
        numpyro.deterministic("sd", jnp.sqrt(conditioned_gp.variance))


mcmc = MCMC(
    NUTS(gp_model, dense_mass=True, target_accept_prob=0.9),
    num_warmup=1000,
    num_samples=1000,
)
mcmc.run(jax.random.PRNGKey(0), x=x, y=y, y_err=y_err, test_x=test_x)
mcmc.print_summary()

# %%
samples = mcmc.get_samples()
# predictive = Predictive(gp_model, samples)
# posterior_predictive = predictive(
#     jax.random.PRNGKey(0), x=x, y=y, y_err=y_err, test_x=test_x
# )
pred = samples["pred"]
# sd = posterior_predictive["sd"]
pred.shape

# %%
pred_mean = np.mean(pred, axis=0)
# total_sd = np.sqrt(np.mean(sd**2, axis=0))  # combining GP uncertainty
total_sd = np.std(pred, axis=0)  # emprical sd

# Plot mean and uncertainty bands
plt.figure(figsize=(10, 6))

# Plot data points
plt.scatter(x, y, color="black", alpha=0.6, label="Data")

# Plot mean prediction
plt.plot(test_x, pred_mean, "C0", label="Mean prediction")

# Plot ±2 standard deviation bands (95% confidence interval)
plt.fill_between(
    test_x,
    pred_mean - 2 * total_sd,
    pred_mean + 2 * total_sd,
    color="C0",
    alpha=0.3,
    label="2σ uncertainty",
)

plt.legend()
plt.xlabel("Standardized Age")
plt.ylabel("Standardized RSL")

# %%
import tinygp


class PoweredExponential(tinygp.kernels.Kernel):
    p: jax.Array  # length scale parameter
    kappa: jax.Array  # power parameter (between 0 and 2)

    def evaluate(self, X1, X2):
        # D = jnp.sqrt((X1 - X2) ** 2)
        D = jnp.abs((X1 - X2))
        return self.p ** (D**self.kappa)


# %%
def eiv_gp_model(x, y, x_err, y_err, test_x):
    jitter = numpyro.sample(
        "jitter", dist.HalfNormal(1)
    )  # similar to microscale variance mentioned in paper
    # scale = numpyro.sample("sigma", dist.HalfNormal(1))
    p = numpyro.sample("p", dist.Uniform(0, 1))
    # kernel = kernels.ExpSquared(scale)
    kernel = PoweredExponential(p=p, kappa=1.99)

    chi = numpyro.sample(
        "chi", dist.Normal(jnp.mean(x), jnp.std(x)), sample_shape=(len(x),)
    )

    gp = GaussianProcess(kernel, chi, diag=y_err**2 + jitter)

    numpyro.sample("gp", gp.numpyro_dist(), obs=y)
    numpyro.sample("x_obs", dist.Normal(chi, x_err), obs=x)

    if y is not None:
        conditioned_gp = gp.condition(y, test_x).gp
        numpyro.deterministic("pred", conditioned_gp.loc)


mcmc = MCMC(
    NUTS(eiv_gp_model, dense_mass=True, target_accept_prob=0.9),
    num_warmup=1000,
    num_samples=1000,
)
mcmc.run(jax.random.PRNGKey(0), x=x, y=y, x_err=x_err, y_err=y_err, test_x=test_x)
mcmc.print_summary()

# %%
samples = mcmc.get_samples()
pred = samples["pred"]
pred_mean = np.mean(pred, axis=0)
total_sd = np.std(pred, axis=0)  # emprical sd

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="black", alpha=0.6, label="Data")
plt.plot(test_x, pred_mean, "C0", label="Mean prediction")
plt.fill_between(
    test_x,
    pred_mean - 2 * total_sd,
    pred_mean + 2 * total_sd,
    color="C0",
    alpha=0.3,
    label="2sigma uncertainty",
)

plt.legend()
plt.xlabel("Standardized Age")
plt.ylabel("Standardized RSL")

# %%
from jax.experimental.ode import odeint

integral_bounds = ((x - x_err).min(), (x + x_err).max())
integral_bounds

# %%
def eiv_igp_model(x, y, x_err, y_err, test_x):
    jitter = numpyro.sample(
        "jitter", dist.HalfNormal(1)
    )  # similar to microscale variance mentioned in paper
    scale = numpyro.sample("sigma", dist.HalfNormal(1))
    kernel = kernels.ExpSquared(scale)

    chi = numpyro.sample(
        "chi", dist.Normal(jnp.mean(x), jnp.std(x)), sample_shape=(len(x),)
    )

    gp = GaussianProcess(kernel, chi, diag=y_err**2 + jitter)

    numpyro.sample("x_obs", dist.Normal(chi, x_err), obs=x)
    numpyro.sample("gp", gp.numpyro_dist(), obs=y)

    if y is not None:
        conditioned_gp = gp.condition(y, test_x).gp
        numpyro.deterministic("pred", conditioned_gp.loc)


mcmc = MCMC(
    NUTS(eiv_igp_model, dense_mass=True, target_accept_prob=0.9),
    num_warmup=1000,
    num_samples=1000,
)
mcmc.run(jax.random.PRNGKey(0), x=x, y=y, x_err=x_err, y_err=y_err, test_x=test_x)
mcmc.print_summary()

# %%
from diffrax import diffeqsolve, ODETerm, Dopri5
import jax.numpy as jnp


def f(t, y, args):
    return -y


term = ODETerm(f)
solver = Dopri5()
y0 = jnp.array([2.0, 3.0])
solution = diffeqsolve(term, solver, t0=0, t1=1, dt0=0.1, y0=y0)
solution

```
