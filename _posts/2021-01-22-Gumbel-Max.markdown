---
title: On the Gumbel-Max Trick
date: 2021-01-22 11:12:00 -0400
categories: [Explanations]
tags: [academic]     # TAG names should always be lowercase
---

<script src="//yihui.org/js/math-code.js"></script>
<!-- Just one possible MathJax CDN below. You may use others. -->
<script async
  src="//mathjax.rstudio.com/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


Softmax's slicker sibling.

***

## **Motivation**

I've recently been playing around with a few nature-inspired metaheuristic algorithms (think genetic algorithms, simulated annealing, etc.) In such settings, an algorithm iterates through a candidate solution search space with the objective of converging to an optimal solution (for some sense of the word optimal, i.e. characterized by the algorithm's loss function). If executed with well-chosen hyperparameters (especially temporal ones), such algorithms are able to initially avoid locally optimal solutions, while gradually honing in on globally optimal solutions by making more and more precise -- and careful -- decisions.

Concretely, an algorithm in this class usually iterates through the candidate space by probabilistically choosing the next candidate(s) and performing some set of action(s) on them. Speaking in abstract, for a genetic algorithm, this manifests in the form of chromosome sampling and crossover, as well as mutation. As for stimulated annealing, a single next candidate $s^*$ is sampled, and the action is simply whether or not to transition to this next candidate from the current candidate $ s $.

The probability of sampling the next candidates are derived from the algorithm-specific loss function. In my case, I've come across a specific instance of a genetic algorithm with a loss function that scores solutions by $ \log(p) $ -- that is, by the log of the probability that the solution is chosen.

Given this choice of metric, the question becomes **how to best sample from a distribution parametrized by log-probabilities.**


## **Good Old Fashioned Exp-Norm, i.e. Softmax**

So we have a distribution of log-probabilities $ x_k $. The standard way to sample from this distribution is to use the **softmax function** (the same one you might have heard of for classification in neural networks). A more descriptive synonym for softmax is the **normalized exponential function**. As the name implies, we will exponentiate and transform the log-probabilities $ x_k $ to $ \exp(x_k) $. Recall that by basic probability axioms, the probabilities all events need to sum to 1. We guarantee this by normalizing each transformed probability. Ultimately, we have the following transformation allowing us to sample from the distribution:

$$ x_k \mapsto \exp(x_k) \mapsto \frac{1}{S} \exp(x_k) \text{ where } S = \sum_{k=1}^{N} \exp(x_k) \text{ for number of categories $N$}$$

That is, the probability of sampling item $ k $ from $ N $ items is:

$$ \pi_k = P(K = k) = \frac{1}{S} \exp(x_k) $$

Critically, the $ x_k $ are unconstrained in $ \mathbb{R} $, but the $ \pi_k $ lie on the probability simplex (i.e. $\forall k$, $\pi_k \geq 0$, and $\sum \pi_k = 1$), as desired.

## **The Gumbel-Max Trick**

Interestingly, the following formulation is equivalent to the softmax function:

$$ \pi_k = \underset{ k \in \{1, \dots, N \} } {\text{argmax}} (x_k + z_k) \text{ where }  z_k \underset{\text{i.i.d.}}{\sim} \text{Gumbel}(0,1) $$

There are multiple benefits to using the Gumbel-Max Trick. Most saliently:
1. It operates **primarily in log-space**, thereby avoiding potentially nasty numerical over/under-flow errors and unexpected/incorrect sampling behavior.
2. It entirely **bypasses the need for marginalization** (i.e. exp-sum), which can be expensive for a large number of categories.

Definitely pretty slick! We'll see next the secret sauce behind this black magic.

## **Derivation**

First, recall that for a random variable $ X \sim \text{Gumbel}(0,1)$, we have the following PDF and CDF:

$$
\begin{align}
f(x) &= e^{-x + e^{-x}} \\
F(x) &= e^{-e^{-x}}
\end{align} $$

Now, let $ M = \underset{ k \in \{1, \dots, N \} } {\text{argmax}} (x_k + z_k) \text{ where }  z_k \underset{\text{i.i.d.}}{\sim} \text{Gumbel}(0,1) $, as defined above.

Define $ u_k = \log(\pi_k) + z_k = \log(e^{x_k}) - \log(\sum e^{x_k}) + z_k = \log(e^{x_k}) - \log(1) = x_k + z_k $ for notational convenience. Critically, the randomness in $ u_k$ comes from $z_k$, whereas the $\log(\pi_k) = x_k $  are known. We then have that:

$$
\begin{align} P(M = k) &= P(u_k \geq u_i, \, \forall i \neq k)  \\
                       &= \int_{-\infty}^{\infty}  P(u_k \geq u_i, \, \forall i \neq k | u_k) P(u_k) ~du_k \\
                       &= \int_{-\infty}^{\infty}  \prod_{i \neq k} P(u_k \geq u_i | u_k) P(u_k) ~du_k \\
                       &= \int_{-\infty}^{\infty}  \prod_{i \neq k} P(z_i \leq u_k - \log(\pi_i) | u_k) P(u_k) ~du_k \\
                       &= \int_{-\infty}^{\infty}  \prod_{i \neq k} [e^{-e^{\log(\pi_i) - u_k}}] P(z_k) ~du_k \\
                       &= \int_{-\infty}^{\infty}  \prod_{i \neq k} [e^{-e^{\log(\pi_i) - u_k}}] P(u_k - \log(\pi_k)) ~du_k \\
                       &= \int_{-\infty}^{\infty}  \prod_{i \neq k} [e^{-e^{\log(\pi_i) - u_k}}] e^{-(u_k - \log(\pi_k)) + e^{-{(u_k - \log(\pi_k))}}} ~du_k \\
                       &= \int_{-\infty}^{\infty} [e^{-\sum_{i \neq k} e^{\log(\pi_i) - u_k}}] \cdot \pi_k e^{-u_k + \pi_ke^{-u_k}} ~du_k \\
                       &= \int_{-\infty}^{\infty} [e^{-\sum_{i \neq k} \pi_ie^{- u_k}}] \cdot \pi_k e^{-u_k + \pi_ke^{-u_k}} ~du_k \\
                       &= \pi_k \int_{-\infty}^{\infty} e^{-u_k - e^{-u_k}(\sum_{i \neq k} \pi_i + \pi_k )} ~du_k \\
                       &= \pi_k \int_{-\infty}^{\infty} e^{-u_k - e^{-u_k}} ~du_k \\
                       &= \pi_k \int_{-\infty}^{\infty} e^{-z_k - e^{-z_k}} ~dz_k \\
                       &= \pi_k = \frac{1}{S} \exp(x_k)
\end{align} $$

Thus, using Gumbel-Max Trick is indeed equivalent to using the softmax function.
