# Bayesian Conditional Transformation Models in Liesel

[![pre-commit](https://github.com/liesel-devs/liesel-ctm/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/liesel-devs/liesel-ctm/actions/workflows/pre-commit.yml)
[![pytest](https://github.com/liesel-devs/liesel-ctm/actions/workflows/pytest.yml/badge.svg)](https://github.com/liesel-devs/liesel-ctm/actions/workflows/pytest.yml)
[![pytest-cov](tests/coverage.svg)](https://github.com/liesel-devs/liesel-ctm/actions/workflows/pytest.yml)

`bctm` is a Python library for building Bayesian Conditional
Transformation Models and sampling from their parameter’s posterior
distribution via MCMC. It is built on top of the probabilistic
programming framework.

For more on Liesel, see

- [The Liesel documentation](https://docs.liesel-project.org/en/latest/)
- [The Liesel GitHub repository](https://github.com/liesel-devs/liesel)

For more on Bayesian Conditional Transformation Models, see

- [Carlan, Kneib & Klein (2022). Bayesian Conditional Transformation
  Models](https://arxiv.org/abs/2012.11016)
- [Carlan & Kneib (2022). Bayesian discrete conditional transformation
  models](https://journals.sagepub.com/doi/10.1177/1471082X221114177)

## Example usage

For a start, we import the relevant packages.

``` python
import numpy as np
import liesel.model as lsl
import liesel.goose as gs
import liesel_bctm as bctm
```

### Data generation

Next, we generate some data. We do that by defining a data-generating
function - this function will become useful for comparing our model
performance to the actual distribution later.

``` python
rng = np.random.default_rng(seed=3)

def dgf(x, n):
    """Data-generating function."""
    return rng.normal(loc=1. + 5*np.cos(x), scale=2., size=n)

n = 300
x = np.sort(rng.uniform(-2, 2, size=n))

y = dgf(x, n)
data = dict(y=y, x=x)
```

Let’s have a look at a scatter plot for these data. The solid blue line
shows the expected value.

![](README_files/figure-commonmark/Plot%20data-1.png)

### Model building

Now, we can build a model:

``` python
ctmb = (
    bctm.CTMBuilder(data)
    .add_intercept()
    .add_trafo("y", nparam=15, a=2.0, b=0.5, name="y")
    .add_pspline("x", nparam=15, a=2.0, b=0.5, name="x")
    .add_response("y")
)

model = ctmb.build_model()
```

Let’s have a look at the model graph:

``` python
lsl.plot_vars(model)
```

![](README_files/figure-commonmark/Model%20graph-1.png)

### MCMC sampling

With `bctm.ctm_mcmc`, it’s easy to set up an
[EngineBuilder](https://docs.liesel-project.org/en/latest/generated/liesel.goose.builder.EngineBuilder.html)
with predefined MCMC kernels for our model parameters. We just need to
define the warmup duration and the number of desired posterior samples,
then we are good to go.

``` python
nchains = 2
nsamples = 1000
eb = bctm.ctm_mcmc(model, seed=1, num_chains=nchains)
eb.positions_included += ["z"]
eb.set_duration(warmup_duration=500, posterior_duration=nsamples)

engine = eb.build()
engine.sample_all_epochs()
```

    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 75 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 6, 5 / 75 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 5, 1 / 25 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 5, 4 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 14, 9 / 100 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 21, 10 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_02: 4, 5 / 50 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Finished warmup
    liesel.goose.engine - INFO - Starting epoch: POSTERIOR, 1000 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch

``` python

results = engine.get_results()
samples = results.get_posterior_samples()
```

### Posterior predictive conditional distribution

To interpret our model, we now define some fixed covariate values for
which we are going to evaluate the conditional distribution. We also
define an even grid of response values to use in the evaluations. The
function `bctm.grid` helps us to create arrays of fitting length.

``` python
yvals = np.linspace(np.min(y), np.max(y), 100)
xvals = np.linspace(np.min(x), np.max(x), 6).round(2)[1:-1]
ygrid, xgrid = bctm.grid(yvals, xvals)
```

To get a sense of how well our model captures the true distribution of
the response even for complex data-generating mechanism, `bctm` offers
the helper function `bctm.samle_dgf`. This function gives us a
data-frame with our desired number of samples, each for a grid of
covariate values. We just need to enter the data-generating function
that we defined above.

``` python
dgf_samples = bctm.sample_dgf(dgf, 5000, x=xvals)
dgf_samples
```

           index    x         y
    0          0 -1.2  4.344596
    1          1 -1.2  4.224118
    2          2 -1.2  4.365094
    3          3 -1.2  4.452642
    4          4 -1.2  0.833007
    ...      ...  ...       ...
    19995   4995  1.2  3.966420
    19996   4996  1.2  2.106471
    19997   4997  1.2  3.516566
    19998   4998  1.2 -0.466490
    19999   4999  1.2  0.156236

    [20000 rows x 3 columns]

#### Uncertainty visualization via quantiles

Next, we evaluate the transformation density for our fixed covariate
values. The function `bctm.dist_df` does that for us. It evaluates the
transformation density for each individual MCMC sample and returns a
summary data-frame that contains the mean and some quantiles of our
choice (default: 0.1, 0.5, 0.9).

``` python
dist_df = bctm.cdist_quantiles(samples, ctmb, y=ygrid, x=xgrid)
dist_df
```

           id summary    x          y       pdf       cdf  log_prob
    0       0    mean -1.2  -5.504508  0.000277  0.000238 -8.787865
    1       1    mean -1.2  -5.341085  0.000338  0.000288 -8.529080
    2       2    mean -1.2  -5.177663  0.000413  0.000349 -8.277538
    3       3    mean -1.2  -5.014240  0.000504  0.000424 -8.032559
    4       4    mean -1.2  -4.850818  0.000614  0.000515 -7.793633
    ...   ...     ...  ...        ...       ...       ...       ...
    1595  395    q0.9  1.2  10.020624  0.000360  0.999993 -7.929464
    1596  396    q0.9  1.2  10.184047  0.000263  0.999996 -8.242940
    1597  397    q0.9  1.2  10.347469  0.000194  0.999998 -8.545481
    1598  398    q0.9  1.2  10.510891  0.000143  0.999999 -8.854554
    1599  399    q0.9  1.2  10.674314  0.000103  0.999999 -9.180962

    [1600 rows x 7 columns]

Now it is time to plot the estimated distribution. For this part, I like
to switch to R, because `ggplot2` is just amazing. I load `reticulate`
here, too, because that lets us access Python objects from within R.

``` r
library(reticulate)
library(tidyverse)
```

    ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.1 ──

    ✔ tibble  3.1.7     ✔ dplyr   1.0.9
    ✔ tidyr   1.2.0     ✔ stringr 1.5.0
    ✔ readr   2.1.2     ✔ forcats 0.5.1
    ✔ purrr   1.0.1     

    ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ✖ dplyr::filter() masks stats::filter()
    ✖ dplyr::lag()    masks stats::lag()

For easy handling during plotting, I transform the data to wide format.

``` r
dist_df <- py$dist_df

dist_df_wide <- dist_df |>
    pivot_wider(
        names_from = "summary",
        values_from = c("pdf", "cdf", "log_prob")
    )
```

Next, it is time for our plot:

``` r
dist_df_wide |>
    ggplot() +
    geom_ribbon(aes(x=y, ymin=`pdf_q0.1`, ymax=`pdf_q0.9`),
                fill = "#56B4E9",
                alpha = 0.5) +
    geom_density(data = py$dgf_samples,
                 aes(y),
                 linetype = "dotted",
                 adjust=2) +
    geom_line(aes(y, `pdf_q0.5`)) +
    facet_wrap(~x, labeller = label_both) +
    labs(
        x = "Response",
        y = "Density",
        title = "Posterior estimates of the response distribution",
        subtitle = "Shaded area shows 0.1 - 0.9 quantiles.\n
        Dotted line is KDE of samples from the data-generating function."
    )
```

![](README_files/figure-commonmark/Plot%20cPDF%20with%20quantiles-1.png)

#### Uncertainty visualization via random samples

``` python
rdist_df = bctm.cdist_rsample(samples, ctmb, size=100, y=ygrid, x=xgrid)
rdist_df
```

                pdf       cdf   log_prob          y    x  id
    0      0.000221  0.000153  -8.417691  -5.504508 -1.2   0
    1      0.000276  0.000194  -8.193705  -5.341085 -1.2   0
    2      0.000347  0.000244  -7.966990  -5.177663 -1.2   0
    3      0.000436  0.000308  -7.736731  -5.014240 -1.2   0
    4      0.000552  0.000389  -7.502257  -4.850818 -1.2   0
    ...         ...       ...        ...        ...  ...  ..
    39995  0.000157  0.999954  -8.761975  10.020624  1.2  99
    39996  0.000092  0.999974  -9.295465  10.184047  1.2  99
    39997  0.000053  0.999985  -9.854688  10.347469  1.2  99
    39998  0.000029  0.999992 -10.432300  10.510891  1.2  99
    39999  0.000016  0.999995 -11.020738  10.674314  1.2  99

    [40000 rows x 6 columns]

``` r
py$rdist_df |>
    ggplot() +
    geom_line(aes(y, pdf, group = id), alpha = 0.15, color = "#56B4E9") +
    geom_line(data = dist_df_wide, aes(y, `pdf_q0.5`)) +
    geom_density(data = py$dgf_samples, aes(y), linetype = "dotted", adjust=2) +
    facet_wrap(~x, labeller = label_both) +
        labs(
        x = "Response",
        y = "Density",
        title = "Posterior estimates of the response distribution",
        subtitle = "Blue: 100 samples from the posterior predictions. Solid: Median.\n
        Dotted line is KDE of samples from the data-generating function."
    )
```

![](README_files/figure-commonmark/Plot%20cPDF%20with%20samples-1.png)

### Plot the transformation function

#### Transformation part

``` python
tf_quantiles = bctm.partial_ctrans_quantiles(samples, ctmb, y=yvals)
tf_samples = bctm.partial_ctrans_rsample(samples, ctmb, y=yvals, size=50)
```

``` r
tf_wide <- py$tf_quantiles |> pivot_wider(
    names_from = "summary", values_from = c("value", "value_d")
)

tf_wide |>
    ggplot() +
    geom_line(
        data = py$tf_samples,
        aes(y, value, group = id),
        color = "#56B4E9",
        alpha = 0.2
    ) +
    geom_line(aes(y, `value_q0.5`)) +
    labs(
        x = "Response",
        y = expression(h[0](y)),
        title = "Conditional transformation function",
        subtitle = "Black: Median, Blue: 50 random samples from the posterior"
    )
```

![](README_files/figure-commonmark/Plot%20h0(y)-1.png)

#### Location shift part

``` python
xvals = np.linspace(np.min(x), np.max(x), 100)
tf_quantiles = bctm.partial_ctrans_quantiles(samples, ctmb, x=xvals)
tf_samples = bctm.partial_ctrans_rsample(samples, ctmb, x=xvals, size=50)
```

``` r
tf_wide <- py$tf_quantiles |>
    pivot_wider(names_from = "summary",
                values_from = c("value"),
                names_prefix="value_")

tf_wide |>
    ggplot() +
    geom_line(
        data = py$tf_samples,
        aes(x, value, group = id),
        color = "#56B4E9",
        alpha = 0.2
    ) +
    geom_line(aes(x, `value_q0.5`)) +
    labs(
        x = "Response",
        y = expression(h[1](x)),
        title = "Conditional transformation function",
        subtitle = "Black: Median, Blue: 50 random samples from the posterior"
    )
```

![](README_files/figure-commonmark/Plot%20h1(x)-1.png)

### Diagnostics

#### Distribution of the latent variable

``` python
z = bctm.ConditionalPredictions(samples, ctmb).ctrans().mean(axis=(0,1))
```

``` r
kde_plot <- ggplot() +
    stat_function(fun = dnorm, linetype = "dotted") +
    geom_density(aes(py$z), adjust = 1) +
    labs(x = "h(y|x)", title = "Kernel density estimator for h(y|x)",
         subtitle = "Dotted line: True Standard Normal distribution") +
    NULL

qq_plot <- ggplot() +
    aes(sample = py$z, color="h(y|x)") +
    stat_qq() +
    stat_qq(aes(sample = py$y, color="y"), alpha=0.6) +
    stat_qq_line(color="black") +
    labs(x = "Theoretical quantile", y="Empirical quantile",
         title = "QQ-plot for latent variable h(y|x) and y")

ggpubr::ggarrange(qq_plot, kde_plot, ncol = 2, common.legend = TRUE, legend = "top")
```

![](README_files/figure-commonmark/Plot%20estimated%20density%20of%20latent%20variable-1.png)

#### rhat

``` python
summary = gs.Summary(results)
summary_df = summary._param_df().reset_index()
error_df = summary._error_df().reset_index()
error_df = error_df.astype({"count": "int32", "relative": "float32"})
```

``` r
py$summary_df |>
    filter(parameter != "z") |>
    group_by(parameter) |>
    mutate(index = seq(0, n()-1)) |>
    mutate(parameter_i = paste0(parameter, "[", sprintf("%02d", index), "]")) |>
    relocate(parameter_i) |>
    ggplot() +
    geom_point(aes(parameter_i, rhat)) +
    geom_line(aes(parameter_i, rhat, group=1), alpha = 0.5) +
    geom_hline(yintercept = 1.01, linetype = "dotted") +
    coord_flip() +
    NULL
```

![](README_files/figure-commonmark/Plot%20rhat-1.png)

#### Effective sample size

``` r
py$summary_df |>
    filter(parameter != "z") |>
    group_by(parameter) |>
    mutate(index = seq(0, n()-1)) |>
    mutate(parameter_i = paste0(parameter, "[", sprintf("%02d", index), "]")) |>
    relocate(parameter_i) |>
    ggplot() +
    aes(parameter_i, ess_bulk) +
    # aes(parameter_i, ess_bulk, color=dgf) +
    geom_point() +
    geom_line(aes(group = 1), alpha = 0.5) +
    geom_hline(yintercept = py$nchains*py$nsamples, linetype = "dotted") +
    coord_flip() +
    # facet_wrap(~dgf, nrow = 4) +
    labs(title = "Effective sample sizes (ESS Bulk)",
         subtitle = "Dotted line: Actual number of MCMC draws.") +
    NULL
```

![](README_files/figure-commonmark/Plot%20ess_bulk-1.png)

``` r
py$summary_df |>
    filter(parameter != "z") |>
    group_by(parameter) |>
    mutate(index = seq(0, n()-1)) |>
    mutate(parameter_i = paste0(parameter, "[", sprintf("%02d", index), "]")) |>
    relocate(parameter_i) |>
    ggplot() +
    aes(parameter_i, ess_tail) +
    geom_point() +
    geom_line(aes(group = 1), alpha = 0.5) +
    geom_hline(yintercept = py$nchains*py$nsamples, linetype = "dotted") +
    coord_flip() +
    # facet_wrap(~dgf, nrow = 4) +
    labs(title = "Effective sample sizes (ESS Tail)",
         subtitle = "Dotted line: Actual number of MCMC draws.") +
    NULL
```

![](README_files/figure-commonmark/Plot%20ess_tail-1.png)

#### Transition errors

``` r
py$error_df |>
    ggplot() +
    aes(error_msg, relative) +
    geom_bar(aes(fill = error_msg), stat = "identity",
             position = position_dodge2(preserve = "single")) +
    facet_wrap(~phase) +
    theme(legend.position = "top") +
    labs(title = "Error Summary") +
    NULL
```

![](README_files/figure-commonmark/Plot%20error%20summary-1.png)

#### Variance parameter plots

``` python
gs.plot_param(results, "y_igvar")
```

![](README_files/figure-commonmark/Diagnostic%20plots%20for%20variance%20of%20trafo-1.png)

``` python
gs.plot_param(results, "x_igvar")
```

![](README_files/figure-commonmark/Diagnostic%20plots%20for%20variance%20of%20pspline-3.png)
