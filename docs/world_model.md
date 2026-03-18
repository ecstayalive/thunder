# World Model

## Variational Inference
$$
\begin{align}
\ln p(o) &= ln \int p(o|s) p(s) ds \\
&= \ln \left[ \int \frac{p(o,s) q(s)}{q(s)}ds \right] \\
&= \ln \left[ \mathbb{E}_{q(s)}\left[ \frac{p(o,s)}{q(s)} \right] \right]\\
&\ge \mathbb{E}_{q(s)} \left[ \ln(\frac{p(o,s)}{q(s)}) \right]
\end{align}
$$

$$
\begin{align}
\mathbb{E}_{q(s)} \left[ \ln(\frac{p(o,s)}{q(s)}) \right] &= \mathbb{E}_{q(s)} \left[ \ln(\frac{p(o|s) p(s)}{q(s)}) \right]  \\
&= \mathbb{E}_{q(s)} \left[ \ln(\frac{p(s|o) p(o)}{q(s)}) \right]
\end{align}
$$

$$
\begin{align}
\mathbb{E}_{q(s)} \left[ \ln(\frac{p(o|s) p(s)}{q(s)}) \right] &= -D_{KL}(q(s) || p(s)) + \mathbb{E}_{q(s)} \left[ \ln(p(o|s)) \right]
\end{align}
$$

$$
\begin{align}
\mathbb{E}_{q(s)} \left[ \ln(\frac{p(s|o) p(o)}{q(s)}) \right] &= -D_{KL}(q(s) || p(s|o)) + \mathbb{E}_{q(s)} \left[ \ln(p(o)) \right] \\
&= -D_{KL}(q(s)||p(s|o)) + \int q(s) \ln(p(o)) ds \\
&= -D_{KL}(q(s)||p(s|o)) + \ln(p(o))
\end{align}
$$

$$
\begin{align}
-D_{KL}(q(s)||p(s|o)) + \ln p(o) = -D_{KL}(q(s) || p(s)) + \mathbb{E}_{q(s)} \left[ \ln(p(o|s)) \right]
\end{align}
$$

$$
\begin{align}
\ln p(o) &= -D_{KL}(q(s) || p(s)) + \mathbb{E}_{q(s)} \left[ \ln(p(o|s)) \right] + D_{KL}(q(s)||p(s|o)) \\
&= -F + D_{KL}(q(s)||p(s|o))
\end {align}
$$

We believe observations arise from the interaction of a set of latent variables $s$. After collecting observational data, we aim to construct a distribution $q(s)$ that approximates the true posterior distribution $p(s|o)$ as closely as possible, meaning minimize $D_{KL}(q(s)||p(s|o))$. Within a dataset, the probability $p(o)$ is constant, thus, when the KL divergence is minimised, the problem transforms into one of minimising free energy $F=D_{KL}(q(s) || p(s)) - \mathbb{E}_{q(s)} \left[ \ln(p(o|s)) \right]$. This is variational inference: converting a difficult-to-optimise problem into one that can be optimised.

The question is how we should construct the distribution $q(s)$. Since we have obtained the observation data, we can use this data for construction. In the above formula, $q(s)$ becomes $q(s|o)$, which is the method proposed in the [VAE](https://arxiv.org/abs/1312.6114) paper.

$$
\begin{align}
\ln p(o) &= -D_{KL}(q(s|o) || p(s)) + \mathbb{E}_{q(s|o)} \left[ \ln(p(o|s)) \right] + D_{KL}(q(s|o)||p(s|o)) \\
&= ELBO + D_{KL}(q(s|o)||p(s|o))
\end {align}
$$

## Activate Inference(SSM,World Model)

Someone believes that biological brains operate on the same principles as variational inference. By observing the world, the brain continuously infers the underlying true state $s$ based on historical observations $o_{\le t}$. Based on this state, organisms select actions that influence the observations we obtain, making them align more closely with our expectations. At this point, $\ln p(o)$ in above formula becomes in $\ln p(o_{1:T}|a_{1:T})$.

$$
\begin{align}
\ln p(\tilde{o}|\tilde{a}) &= \ln \int p(\tilde{o}, \tilde{s} | \tilde{a}) d\tilde{s} \\
&= \ln \left[ \int \frac{p(\tilde{o}, \tilde{s} | \tilde{a}) q(\tilde{s}|\tilde{o}, \tilde{a})}{q(\tilde{s}|\tilde{o}, \tilde{a})} d\tilde{s} \right] \\
&= \ln \left[ \mathbb{E}_{q(\tilde{s}|\tilde{o}, \tilde{a})}\left[ \frac{p(\tilde{o}, \tilde{s} | \tilde{a})}{q(\tilde{s}|\tilde{o}, \tilde{a})} \right] \right]\\
&\ge \mathbb{E}_{q(\tilde{s}|\tilde{o}, \tilde{a})} \left[ \ln\left(\frac{p(\tilde{o}, \tilde{s} | \tilde{a})}{q(\tilde{s}|\tilde{o}, \tilde{a})}\right) \right] \quad \text{(Jensen's Inequality)} \\
&= \mathbb{E}_{q(s_{1:T}|o_{\le 1:T}, a_{< 1:T})} \left[ \ln \frac{p(o_{1:T}, s_{1:T} | a_{1:T})}{q(s_{1:T} | o_{\le 1:T}, a_{< 1:T})} \right] \\
&= \mathbb{E}_{q} \left[ \ln \frac{\left(\prod_{t=1}^{T} p(o_t | s_t)\right) \left(\prod_{t=1}^{T} p(s_t | s_{t-1}, a_{t-1})\right)}{\prod_{t=1}^{T} q(s_t | a_{\le {t-1}}, o_{\le t})} \right] \\
&= \mathbb{E}_{q} \left[ \sum_{t=1}^{T} \ln p(o_t | s_t) + \sum_{t=1}^{T} \ln p(s_t | s_{t-1}, a_{t-1}) - \sum_{t=1}^{T} \ln q(s_t | a_{\le {t-1}}, o_{\le t}) \right] \\
&= \sum_{t=1}^{T} \mathbb{E}_{q} \left[ \ln p(o_t | s_t) + \ln \frac{p(s_t | s_{t-1}, a_{t-1})}{q(s_t | a_{\le {t-1}}, o_{\le t})} \right]\\
&= \sum_{t=1}^{T} \Bigg[ \underbrace{\mathbb{E}_{q(s_t|o_{\le t}, a_{<t})} \ln p(o_t | s_t)}_{\text{reconstruction loss}} \\
&\quad - \underbrace{\mathbb{E}_{q(s_{t-1}|o_{\le {t-1}}, a_{<{t-1}})} \Bigg[ D_{KL} \bigg( q(s_t | a_{\le {t-1}}, o_{\le t}), p(s_t | s_{t-1}, a_{t-1}) \bigg) \Bigg]}_{\text{epistemic value}} \Bigg] \\
&= ELBO
\end{align}
$$

The above is the derivation content from the [Planet](https://arxiv.org/abs/1811.04551) paper. Among these, $q(s_{t}|o_{\le t}, a_{< t})$ is referred to as the **`Representation Model`**, whose function is to derive the true state determining the current observation from the sequence of previous trajectories. We can construct the **`Representation Model`** $q(s_{t}|o_{\le t})$ solely from the observation or just construct $q(s_{t}|f)$ from some feature extracted from observation, and its selection is diverse—any distribution of $s$ will suffice. $p(s_t|s_{t-1}, a_{t-1})$ is the **`Transition Model`**, reflecting the actual state transition process of the world. In fact, we cannot obtain the true $p(s_t|s_{t-1}, a_{t-1})$. For example, in [VAE](https://arxiv.org/abs/1312.6114), we directly assume $p(s)$ follows a standard normal distribution. In **`Dreamer`**, we use a neural network. This is the problem: we have to estimate the current state using the **`Representation Model`**, then perform **`path integration`** with the network to fit these states. At the same time, the **`Representation Model`** must not only ensure the estimated state is accurate (capable of fully reconstructing the observation) but also enable the **`Transition Model`** to perform **`path integration`** more effectively. Usually, transition model is a energy distribution.

$$
\begin{align}
p(s_t|s_{t-1}, a_{t-1}) = \frac{e^{-D(s_t, s_{t-1}, a_{t-1})}}{\int e^{-D(s_t, s_{t-1}, a_{t-1})}ds_t}
\end{align}
$$

### Latent Manifold World Model (Abstract World Model)

If we derive the result from a different perspective, we can obtain a form that does not involve reconstructing observations.

$$ p(o_{1:T}, s_{1:T} | a_{1:T}) = \prod_{t=1}^T \underbrace{p(o_t | s_t)}_{\text{Generation Model}} \underbrace{p(s_t | s_{t-1}, a_{t-1})}_{\text{Transition Model}} $$
*(Note: For $t=1$, this is typically interpreted as the prior $p(s_1)$, i.e., $s_0$ and $a_0$ are either empty or have already been sampled as constants.)*

$$
\max_{\phi, \theta} \sum_k^K I(S_t; Y_{t+k} | a_{t:t+k-1}) - \beta I(S_t; H_t|S_{t-1}, A_{t-1})
$$
<!--
当$\ln p(o_{1:T} | a_{1:T})$在真实的联合分布$p(o_{1:T}, s_{1:T}, a_{1:T})$取期望,我们可以得到$\mathbb{E}_p [\ln p(o_{1:T} | a_{1:T})]$.
根据概率的边缘化公式和贝叶斯定理，对于任意隐变量 $s_{1:T}$，有：
$$ p(o_{1:T} | a_{1:T}) = \frac{p(o_{1:T}, s_{1:T} | a_{1:T})}{p(s_{1:T} | o_{1:T}, a_{1:T})} $$
两边取对数，并在真实的联合分布 $p(o_{1:T}, s_{1:T}, a_{1:T})$ 下取期望：
$$ \mathbb{E}_p [\ln p(o_{1:T} | a_{1:T})] = \mathbb{E}_p \left[ \ln \frac{p(o_{1:T}, s_{1:T} | a_{1:T})}{p(s_{1:T} | o_{1:T}, a_{1:T})} \right] $$
$$ = \mathbb{E}_p \Big[ \ln p(o_{1:T}, s_{1:T} | a_{1:T}) - \ln p(s_{1:T} | o_{1:T}, a_{1:T}) \Big] $$
*(这实际上是当变分后验 $q$ 等于真实后验 $p$ 时，ELBO（证据下界）变为严格等式的形式)*

接下来, 根据图模型分解联合概率和后验概率
1. **分解联合概率**：
   $$ \ln p(o_{1:T}, s_{1:T} | a_{1:T}) = \sum_{t=1}^T \Big( \ln p(o_t | s_t) + \ln p(s_t | s_{t-1}, a_{t-1}) \Big) $$

2. **分解后验概率**：
   $$ \ln p(s_{1:T} | o_{1:T}, a_{1:T}) = \sum_{t=1}^T \ln p(s_t | s_{<t}, o_{1:T}, a_{1:T}) $$
   为了简便，我们定义 $C_t = \{s_{<t}, o_{1:T}, a_{1:T}\}$，所以这一项就是 $\sum_{t=1}^T \ln p(s_t | C_t)$。

将这两个分解代回第一步的等式中：
$$ \mathbb{E}_p [\ln p(o_{1:T} | a_{1:T})] = \mathbb{E}_p \left[ \sum_{t=1}^T \Big( \ln p(o_t | s_t) + \ln p(s_t | s_{t-1}, a_{t-1}) - \ln p(s_t | C_t) \Big) \right] $$

将期望对数概率转化为“条件熵”.在信息论中，条件熵的定义为 $H(X|Y) = -\mathbb{E}_{p(x,y)}[\ln p(x|y)]$。我们将期望算子 $\mathbb{E}_p$ 分配到求和号内的每一项，可以得到：
1. $\mathbb{E}_p[\ln p(o_t | s_t)] = -H(O_t | S_t)$
2. $\mathbb{E}_p[\ln p(s_t | s_{t-1}, a_{t-1})] = -H(S_t | S_{t-1}, A_{t-1})$
3. $\mathbb{E}_p[\ln p(s_t | C_t)] = -H(S_t | C_t)$

替换进去后，原式变为：
$$ \mathbb{E}_p [\ln p(o_{1:T} | a_{1:T})] = \sum_{t=1}^T \Big( -H(O_t | S_t) - H(S_t | S_{t-1}, A_{t-1}) + H(S_t | C_t) \Big) $$

互信息的经典等价公式为：
$I(X; Y) = H(X) - H(X|Y) \implies -H(X|Y) = I(X;Y) - H(X)$
或 $I(X; Y) = H(Y) - H(Y|X) \implies -H(Y|X) = I(X;Y) - H(Y)$

我们对第三步中的三个条件熵分别应用这个公式：
1. **第一个项：**
   $$ -H(O_t | S_t) = I(S_t; O_t) - H(O_t) $$
2. **第二个项：**
   $$ -H(S_t | S_{t-1}, A_{t-1}) = I(S_t; S_{t-1}, A_{t-1}) - H(S_t) $$
3. **第三个项**（注意前面有个加号）：
   $$ +H(S_t | C_t) = -\Big( -H(S_t | C_t) \Big) = -\Big( I(S_t; C_t) - H(S_t) \Big) = -I(S_t; C_t) + H(S_t) $$

将第四步展开的项全部代入第三步的求和公式中：
$$ \sum_{t=1}^T \Big[ \underbrace{I(S_t; O_t) - H(O_t)}_{\text{项 1}} + \underbrace{I(S_t; S_{t-1}, A_{t-1}) - H(S_t)}_{\text{项 2}} \underbrace{- I(S_t; C_t) + H(S_t)}_{\text{项 3}} \Big] $$

你一定会发现一个极其精妙的地方：**项 2 产生了一个 $-H(S_t)$，而项 3 产生了一个 $+H(S_t)$。两者完美抵消！**

抵消后重新整理各项顺序，我们得到了：
$$ \sum_{t=1}^T \Big[ I(S_t; O_t) + I(S_t; S_{t-1}, A_{t-1}) - I(S_t; C_t) - H(O_t) \Big] $$
各项具有如下的意义:

1. **局部空间对齐 ($I(S_t; O_t)$)**：表示隐状态 $S_t$ 必须包含关于当前观测 $O_t$ 的足够信息（重构能力）。
2. **局部时序对齐 ($I(S_t; S_{t-1}, A_{t-1})$)**：表示隐状态必须满足马尔可夫转移规律，当前状态应该能从前一状态和动作预测出来（动态预测能力）。
3. **全局信息瓶颈 ($- I(S_t; S_{<t}, O_{1:T}, A_{1:T})$)**：这是一个“惩罚项”。它要求隐状态 $S_t$ 在满足局部预测的同时，**不要**过度记忆整个全局上下文（包括未来的和遥远过去的观测）。这强制表征模型提取出真正的马尔可夫隐状态，从而避免过拟合。
4. **常数项 ($- H(O_t)$)**：这是环境观测的内在熵（数据的随机性），它独立于我们的表征模型，因此在优化过程中是一个常数。

由上述公式,我们认为一个世界模型至少包括两个部分,一个部分是表征模型($p(s_t|o_t)$或者$p(s_t|o_{\le t})$),一个部分是转移模型($p(s_{t+1}|s_t, a_t)$).$s_t$我们认为是观测的表征,一个抽象的概念. 这两个模型都是由神经网络表示,从数据中学习.但是一个问题是,我们无法在真实的联合分布中进行采样,我们只能在一个真实的环境中采样得到$(o_{1:T}, a_{1:T})$. 如果我们想使用上述推导公式进行网络优化,该怎么办? -->

From the above formulation, we argue that a world model must comprise two models: a representation model and a transition model.  The representation model is $p(s_t|o_t)$ or $p(s_t|o_{\le t})$, while the transition model is $p(s_{t+1}|s_t, a_t)$. $p(o_t|s_t)$ is the generative model. Typically, we consider $s_t$ to be a representation of $o_t$, an abstract expression, or a set of variables that truly determine $o_t$. Our goal is to construct a world model in the more abstract, information-dense $S$ space.

An abstract world model usually consists two components:
1. A learnable encoding function $\varphi: O \rightarrow S$. $\varphi$ extract observation into a low-dimensional abstract state space $S \subseteq R^d$, where $d \ll n$.
2. A learnable transition function $\tau: S \times A \rightarrow S$. $\tau$ describes the transition dynamics $T$ within the abstract state space.

## A Unified Policy

### Learning with Activate Inference
According to active inference, creatures infer actions based on the minimisation of expected free energy. The below is the original free energy formula.

$$
G(\pi, s_0) = \sum_{t=0}^{N} \left[ \mathbb{E}_{q(o_{t+1}|s_{t+1}) q(s_{t+1}|s_t, a_t)} \left[ \ln \frac{q(s_{t+1}|o_{\le {t+1}}, a_{\le t})}{q(s_{t+1}|s_t, a_t)} \right] + \mathbb{E}_{q(r_{t+1}|s_{t+1}) q(s_{t+1}|s_t, a_t)} \ln p(r_{t+1}) \right]
$$

$$
\begin{align}
-G(π) &\triangleq D_{KL} \left[ q(o,s|\pi) || p(o,s) \right] \\
&= \mathbb{E}_{q(o, s|\pi)} \left[ ln\frac{q(o, s|\pi)}{p(o, s)} \right] \\
&=  \mathbb{E}_{q(o, s|\pi)} \left[ \ln\left[q(o, s|\pi)\right] - \ln \left[ p(o, s) \right]  \right] \\
&= \mathbb{E}_{q(o, s|\pi)} \left[ \ln \left[ q(s|\pi) \right] - \ln \left[ p(o, s) \right] +\ln\left[q(o|s,\pi)\right] \right]\\
&= \mathbb{E}_{q(o, s|\pi)} \left[ \ln \left[ q(s|\pi) \right] - \ln \left[ p(o) \right] - \ln \left[ p(s|o) \right] +\ln\left[q(o|s,\pi)\right] \right]
\end{align}
$$


$$
G(\pi, s_0) = \sum_{t=0}^{N} \left[ \mathbb{E}_{q(o_{t+1}|s_{t+1}) q(s_{t+1}|s_t, a_t)} \left[ \ln \frac{q(s_{t+1}|o_{\le {t+1}}, a_{\le t})}{q(s_{t+1}|s_t, a_t)} \right] + \mathbb{E}_{q(r_{t+1}|s_{t+1}) q(s_{t+1}|s_t, a_t)} \ln p(r_{t+1}) \right]
$$

We note that when this problem is extended to an infinite time domain, the objective becomes:
$$
G(\pi, s_0) = V_{\pi}(s_0) = \sum_{t=0}^{\infty} \left[ \gamma^{t} \mathbb{E}_{s_{t}(t\ge1), a_{t}} \left[ g(s_{t}, a_{t}) \right] \right] \\
\begin{align}
g(s_{t}, a_{t}) &= \mathbb{E}_{q(o_{t+1}|s_{t+1}) q(s_{t+1}|s_t, a_t)} \left[ \ln \frac{q(s_{t+1}|o_{\le {t+1}}, a_{\le t})}{q(s_{t+1}|s_t, a_t)} \right] + \mathbb{E}_{q(r_{t+1}|s_{t+1}) q(s_{t+1}|s_t, a_t)} \ln p(r_{t+1}) \\
&= \mathbb{E}_{q(o_{t+1}|s_t, a_t)} \left[ \ln \frac{q(s_{t+1}|o_{\le {t+1}}, a_{\le t})}{q(s_{t+1}|s_t, a_t)} \right] + \mathbb{E}_{q(r_{t+1}|s_t, a_t)} \ln p(r_{t+1})
\end{align}
$$

In this context, $G(\pi, s_0)$ comprises two components: one is epistemic value, which drives the agent to search for actions yielding the greatest internal state information gain. The other component is extrinsic value (reward), incentivising the agent to perform specific tasks to obtain greater rewards. $p$ denotes the expected reward probability distribution. Clearly, $G$ is a policy-dependent function, similar to the value function $V$, requiring sampling and fitting using the current policy $\pi_{now}$. Of course, we can also train it using the [SAC](https://arxiv.org/abs/1801.01290) approach.

$$
G_{\pi}(s_0, a_0) = Q_{\pi}(s_0, a_0) = g(s_0, a_0) + \mathbb{E}_{q(s_1|s_0, a_0)}[V_{\pi}(s_1)]
$$

The employment of the Q-function enables the avoidance of the reconstruction of observations, thereby accelerating the training speed. Consequently, a Q-function-based training method will be introduced here.


It is imperative that an alternative approach to optimisation is adopted. In active inference, the reinforcement learning optimisation problem is transformed into a inference problem. In other words, the preceding optimisation problem – namely, the action to be taken in order to maximise the expected cumulative reward – is thus rendered: knowing the existence of a successful state that maximises the expected cumulative reward, what action is most likely to have been taken? This view has also been used in [MPO](https://arxiv.org/abs/1806.06920)

However, relying solely on $G(\pi, s_0)$ does not appear to achieve learning as efficient as that of biological organisms. I suspect one reason is that, although we select actions based on free energy, we do not actively seek higher-value states. One example is that when undertaking tasks, we contemplate the future target state and then derive a reasonable course of action based on the current state. For state $s_t$ and our generative model, we believe it can reach any state after a finite number of time steps. One objective of the actor is to find a suitable path along this trajectory. Particularly, if we possess a structured world model, it seems unreasonable not to actively seek out states within this structured space(We use RSSM currently).One question is how we should incorporate this cost into our objective function? And how should we design this architecture?
