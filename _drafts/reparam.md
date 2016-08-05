---
layout: post
title:  "Reparameterizing a Probabilistic Graphical Model"
date:   2016-08-04 14:38:39 -0500
categories: mcmc
---

Let's say we have some random "data" variable $$X$$ with probability density
function
 $$f_X(x\mid\theta)$$ that depends on some parameters
$$\theta$$. In Bayesian statistics, we want to make inferences about $$f_{\Theta \mid X}(\theta \mid x)$$, the posterior distribution
of the parameters given the data $$x$$.
To do this we have to assume that $$\theta$$ are distributed according to some prior density $$f_\Theta(\theta)$$

Then, we can compute the posterior distribution for $$\theta$$ using Bayes' Theorem:

$$f_\Theta(\theta \mid X) = \frac{1}{Z} f_X(x \mid \theta)f_\Theta(\theta)$$

where $$Z$$ is the normalizing constant $$Z = \int_\Theta f_X(x \mid \theta)f_\Theta(\theta)d\theta$$

For all but the simplest models, it is not possible to compute $$Z$$ analytically, and so numerical
integration techniques such as MCMC are used to estimate the posterior instead.

What if we want to use a different parameterization for our model?
Specifically, let's assume $$\theta = h(\phi)$$, where the parameters $$\theta$$ 
are simply a deterministic function $$h$$ of some alternative parameters $$\phi$$ 
(equivalently, we could assume $$\phi = g(\theta)$$ such that $$g(\theta) = h^{-1}(\theta)$$).

Then, we can recompute $$Z$$ in terms of $$\phi$$ using the 
[*change of variables formula*](https://en.wikipedia.org/wiki/Integration_by_substitution#Substitution_for_multiple_variables)
 in the integral.

$$\begin{align}
Z &= \int_\Theta f_X(x \mid \theta)f_\Theta(\theta)d\theta\\
&= \int_\Phi f_X(x \mid h(\phi))f_\Theta(h(\phi))\left\vert \frac{d}{d\phi} h(\phi)\right \vert d\phi\\
\end{align}$$

Where the term 
$$J_h(\phi) = \left\vert \frac{d}{d\phi} h(\phi)\right \vert$$ is the
[absolute value of the Jacobian determinant](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)
(or simply "the Jacobian") of the transformation $$h(\phi)$$, 
and is equal to the change in volume that occurs near $$\phi$$ when transforming coordinates 
in $$\Phi$$ parameter space to $$\Theta$$-space. 
Expanding/contracting the volume near $$\phi$$ also decreases/increases the density at that 
point by a factor equal to the Jacobian. Thus, the Jacobian is needed to 
ensure that $$Z$$, the total mass (volume x density) of the posterior distribution, remains constant 
when changing to a new parameterization.

Next, we can see that the implicit prior for $$\phi$$ is
 the familiar [*formula for the density of a transformed random variable*](https://en.wikipedia.org/wiki/Random_variable#Functions_of_random_variables)

$$\begin{align}
f_\Phi(\phi) &= f_\Theta(h(\phi))\left\vert \frac{d}{d\phi} h(\phi)\right \vert\\
\end{align}
$$

which is just the prior for $$\theta$$ 
multiplied by the Jacobian

$$f_\Phi(\phi) = f_\Theta(\theta)J_h(\phi)$$

Then, the posterior distribution for $$\phi$$ can then be computed in the same way as for $$\theta$$

$$f_{\Phi\mid X}(\phi\mid x) = \frac{1}{Z}f_X(x \mid h(\phi))f_\Phi(\phi)$$.


As an example, let's assume $$X$$ is normally distributed with 
unknown mean $$\mu$$ and known standard deviation $$\sigma$$. Using an exponential
prior for the mean $$f_\mu(\mu) = e^{-\mu}$$, the posterior distribution
for $$\mu$$ is

$$f_\mu(\mu \mid x) = 
\frac{1}{Z}\left\{\frac{1}{\sqrt{2\sigma^2 \pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}\right\}e^{-\mu}$$

Now suppose our data $$X$$ has been log-transformed from some other dataset $$Y$$, i.e. $$X = \ln Y$$.
Thus, $$Y$$ is log-normally distributed with mean $$\lambda = e^{\mu + \sigma^2/2}$$.
In order to make inferences about $$Y$$, we can reparameterize our model in terms of $$\lambda$$.
In this case, $$\mu$$ can be expressed in terms of $$\lambda$$ using the transformation $$\mu = h(\lambda)$$, 
where $$h(\lambda) = \ln \lambda - \sigma^2/2$$.

Thus, the Jacobian is 

$$J_h(\lambda) = \left\vert\frac{d}{d\lambda}h(\lambda)\right\vert 
=\frac{1}{\lambda}
$$

and therefore the implicit prior density for $$\lambda$$ is

$$f_\Lambda(\lambda) = f_\mu(h(\lambda))J_h(\lambda) = \frac{e^{\sigma^2/2}}{\lambda^2} \propto \frac{1}{\lambda^2}$$

Now we can rewrite our posterior in terms of $$\lambda$$, ensuring that $$Z$$ remains unchanged

$$f_{\Lambda\mid X}(\lambda \mid x) = 
\frac{1}{Z}
\left\{
\frac{1}{\sqrt{2\sigma^2 \pi}}
\exp{\left(-\frac{(x-\ln\lambda+\sigma^2/2)^2}{2\sigma^2}\right)}
\right\}
\frac{1}{\lambda^2}
$$

Alternatively, we could specify a prior directly for $$\lambda$$. Let's assume 
$$\lambda \sim \text{Gamma}(\alpha,\beta)$$ such that $$f_\Lambda(\lambda) \propto \lambda^{\alpha -1}e^{-\beta \lambda}$$, and then determine what is the implicit prior
for $$\mu$$

$$\begin{align}
f_\mu(\mu) &= \frac{f_\Lambda(\lambda)}{J_h(\lambda)}\\
&= \frac{f_\Lambda(h^{-1}(\mu))}{J_h(h^{-1}(\mu))}\\
&\propto \exp\left(\alpha(\mu+\sigma^2/2)-\beta e^{\mu+\sigma^2/2}\right)
\end{align}
$$

$$h(\lambda,\mu) = (\delta,\tau) = \left(\lambda - \mu, \frac{\mu}{\lambda}\right)$$


$$h^{-1}(\delta,\tau) = (\lambda,\mu) = \left(\frac{\delta}{1-\tau}, \frac{\tau}{1-\tau}\right)$$

$$\begin{align}
J_h(\lambda,\mu) &=
\begin{vmatrix}
\frac{\partial\delta}{d\lambda} & \frac{\partial\delta}{d\mu} \\
\frac{\partial\tau}{d\lambda} & \frac{\partial\tau}{d\mu}
\end{vmatrix}\\

&= \begin{vmatrix}
1 & -1 \\
\frac{-\mu}{\lambda^2} & \frac{1}{\lambda}
\end{vmatrix}\\

&= \left\vert\frac{\lambda - \mu}{\lambda^2}\right\vert
\end{align}
$$

$$\begin{align}
J_{h^{-1}}(\delta,\tau) &=
\begin{vmatrix}
\frac{\partial\lambda}{d\delta} & \frac{\partial\lambda}{d\tau} \\
\frac{\partial\mu}{d\delta} & \frac{\partial\mu}{d\tau}
\end{vmatrix}\\

&= \begin{vmatrix}
\frac{1}{1-\tau} & \frac{\delta}{(1-\tau)^2} \\
0 & \frac{1}{(1-\tau)^2}
\end{vmatrix}\\

&= \left\vert\frac{\delta}{(1-\tau)^2}\right\vert\\

&= \frac{1}{J_h(\lambda,\mu)}
\end{align}
$$

In the context of a graphical model, nodes can be either deterministic or stochastic.
During an MCMC, we propose changes to the parameters of the model, which are represented using
stochastic nodes. However, we may also wish to propose an update to a deterministic node,
in which case the change of variables induced by the deterministic node function requires
that we include the Jacobian in the proposal acceptance ratio.

In order to be able to compute the Jacobian determinant of a function, 
it must be a *diffeomorphism*,
which is a differentiable bijection whose inverse is also differentiable.
Many deterministic nodes with the same number of input parameters as output values will satisfy
this requirement.

Therefore, in order to reparameterize a graphical model, 
we need to define a deterministic node representing a diffeomorphism $$h(\theta)$$ that will serve as the
transformation for our parameters. 
All of the inputs to this node must be stochastic nodes with computable probability densities.


