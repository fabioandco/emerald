---
title: Testing investment strategies accross markets
---

In the following lines I propose a robustenss test for systematic investment strategies that consists in testing an investment signal across different markets. 
Robustness tests are a key step of a healthy research process: they increase the burden of criteria that investments strategies have to pass before being deployed live and in this way they can reduce the likelihood of finding false investment strategies. 
In this post I will provide some evidence that testing an investment signal accross multiple markets makes sense also from a stastical standpoint.

### Intro 

Researchers might have the tendency to test multiple hypothesis and only present the successful ones (this is known as selection bias). On the top of that, testing many hyptohesis under different settings can easilily lead to false discoveries (a strategy which appears successful in the backtest sample just because of noise in the data and not because of a true pattern that is exploited by the strategy).

A valid research process should therefore entail a batch of robustness tests once the researcher individuates a potential profitable investment strategy with strong economic underpinnings. Testing the investment signal across different markets consistues one of these robustess tests. In the next lines I show why this is a good idea from a statistical standpoint. In fact testing a signal on different markets reduces the probability of picking up false strategies (i.e. false positives).

### Sharpe Ratio, t-statistic and the probability of false discoveries

Given an investment strategy:
$$
\begin{align}
r_{t,strategy} & = r_{t,market} \ x \ signal_{t-1,strategy}\\
\end{align}
$$

Its goodness is usually evaluated through the Sharpe Ratio (SR):
$$
\begin{align}
SR & = \frac{\mu}{\sigma}
\end{align}
$$
where $\mu = E(r_{t,strategy})$ and $\sigma = \sqrt{VAR(r_{t,strategy})}$. 

From a statistical standpoint, the level of significance is usually evaluated by referring to the t-statistic. Values choosen can be 1.645, 1.96, or 2.58 which corresponds respectively to the level of significance of 10%, 5%, or 1%. 
A t-statitic greater than the pre-specified levels leads the researcher to refuse the null hpyothesis in favor of the alternative. In the opposite case the null hypothesis is accepted. The bigger the value of the t-statistic, the easier for the resarcher (and the higher her/his confidence) to reject the null hypothesis.
I remind that the level of significance associated to the t-statistic corresponds to the probability of incorrectly rejecting the null hypothesis when the null is true (i.e. false positive). 

What is the implication? This means that by assuming normally distributed returns, under the null hypothesis (no predictability existing), by testing 100 strategies, respectively 10,5,1 will present a t-stat higher than respectively 1.64,1.96,2.58 merely due to chance. In these 10,5,1 cases the researcher would be incorrectly induced to reject the null hypothesis in favor of predictability. In other words, the more strategies we test, the more strategies with a significant t-statistic we will encounter.

This reasoning can also be translated to the Sharpe Ratio. The Sharpe Ratio is directly related to the $t-statistic$, as  $t-statistic = \sqrt{n}SR$. Similarly as before, the higher the SR, the higher the likelihood of rejecting the null hyptothesis of non-predictability. In other words, the higher the SR, the smaller the probability that our SR is purely due to chance (noise in the data), but reflects istead true predictability that our investment signal captures. 

### Probability of false discoveries when testing on different markets

A solution to reduce false positives is to test the same signal on more (uncorrelated) markets. The markets to select are the ones that shows similar dynamics and from where we might expect similar predictability to be picked up by the signal. An example are the currency and bonds markets, where factors like growth/inflation affect both asset classes (with opposite sign). Then, the probability of having such a SR or bigger  under the null hypothesis (no-predictabiliy) in both markets, which I term here as the probability of having a false positive, is given by the product:

$$
\begin{align}
P(false \ positive) & = (1-P(t<\sqrt{n} \ SR_{1})) \times \  (1-P(t<\sqrt{n} \ SR_{2}))
\end{align}
$$

where $SR_1$, $SR_2$ are the sharpe ratios of the investment strategy implemented on the two uncorrelated markets. Since we are multiplying two probabilities, the product of the single probabilities will be less than each of the single probabilities. Intuitively this result means that if we test the same signal on two or more markets, and if we find positive and sizeable SRs, we are even less likely to stumble upon a false discovery.

### An example:

As an example, assume a strategy with a SR of 0.49, backtested on 12 years of history:

<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
tex2jax: {
inlineMath: [['$','$'], ['\\(','\\)']],
processEscapes: true},
jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
TeX: {
extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
equationNumbers: {
autoNumber: "AMS"
}
}
});
</script>

<pre><code class="python">
from scipy.stats import t
import numpy as np

years_of_backtest = 12
n = 12 * years_of_backtest

SR_1 = 0.49 #Assuming annualized Sharpe Ratio is 0.49

probability_false_positive =1-t.cdf(SR_1*np.sqrt(n/12),n,0,1) #n are the degress of freedom
print("P(false positive)={}".format(np.round(probability_false_positive,3)))
>>> P(false positive)=0.046
</code></pre>

In the 4.6% of cases we wrongly reject the null hypothesis of 0 SR in favor of a profitable investment strategy. 

If we now test the same investment strategy on a second market and we find a SR of 0.49 again, the probability of finding a signal with such a SR on both markets under the null hypothesis would be only 0.2% compared to the 4.6% when testing on a single market.

<pre><code class="python">
SR_2 = SR_1

market_1 = 1-t.cdf(SR_1*np.sqrt(n/12),n,0,1) #n are the degress of freedom
market_2 = 1-t.cdf(SR_2*np.sqrt(n/12),n,0,1) #n are the degress of freedom

probability_false_positive = market_1 * market_2
print("P(false positive)={}".format(np.round(probability_false_positive,3)))
>>> P(false positive)=0.002
</code></pre>
