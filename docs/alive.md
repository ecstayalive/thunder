# Alive

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
&= -D_{KL}(q(s)||p(s|o)) + \int p(s) \ln(p(o)) ds \\
&= -D_{KL}(q(s)||p(s|o)) + \ln(p(o))
\end{align}
$$

$$
\begin{gather}
-D_{KL}(q(s)||p(s|o)) + \ln p(o) = -D_{KL}(q(s) || p(s)) + \mathbb{E}_{q(s)} \left[ \ln(p(o|s)) \right] \\
\begin{align}
\ln p(o) &= -D_{KL}(q(s) || p(s)) + \mathbb{E}_{q(s)} \left[ \ln(p(o|s)) \right] + D_{KL}(q(s)||p(s|o)) \\
&= -F + D_{KL}(q(s)||p(s|o))
\end {align}
\end{gather}
$$

We wish for the state estimated from observations to resemble our prior estimation state as closely as possible, meaning $D_{KL}(q(s)||p(s|o))$ should be minimised. Within a dataset, the probability $p(o)$ is constant; thus, when the KL divergence is minimised, the problem transforms into one of minimising free energy $F=D_{KL}(q(s) || p(s)) - \mathbb{E}_{q(s)} \left[ \ln(p(o|s)) \right]$. This is variational inference: converting a difficult-to-optimise problem into one that can be optimised.


## Activate Inference
$$
\begin{align}
-G(π) &\triangleq D_{KL} \left[ q(o,s|\pi) || p(o,s) \right] \\
&= \mathbb{E}_{q(o, s|\pi)} \left[ ln\frac{q(o, s|\pi)}{p(o, s)} \right] \\
&=  \mathbb{E}_{q(o, s|\pi)} \left[ \ln\left[q(o, s|\pi)\right] - \ln \left[ p(o, s) \right]  \right] \\
&= \mathbb{E}_{q(o, s|\pi)} \left[ \ln \left[ q(s|\pi) \right] - \ln \left[ p(o, s) \right] +\ln\left[q(o|s,\pi)\right] \right]\\
&= \mathbb{E}_{q(o, s|\pi)} \left[ \ln \left[ q(s|\pi) \right] - \ln \left[ p(o) \right] - \ln \left[ p(s|o) \right] +\ln\left[q(o|s,\pi)\right] \right]
\end{align}
$$

According to active inference, creatures infer actions based on the minimisation of expected free energy. The below is the original free energy formula.
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

## GPT5-Answer
Great—let’s turn your idea into a precise, math-first framework. I’ll use a standard latent world model and show (A) how to search in latent space, including how to turn rewards into a desired latent-state distribution and plan a path to it; and (B) how IL, IRL, and RL become concrete learning procedures inside this world-model agent.

---

## Notation (compact)

* Observations $o_t$, actions $a_t$, rewards $r_t$.
* Latent state $z_t \in \mathcal{Z}$.
* Encoder $q_\phi(z_t \mid h_t)$ with history $h_t=(o_{\le t},a_{<t})$ (or RSSM belief $b_t$).
* Latent dynamics $p_\theta(z_{t+1}\mid z_t,a_t)$.
* Reward model $\hat r_\theta(z_t,a_t)$ (optional if rewards are observed).
* Policy $\pi_\psi(a_t\mid z_t,g)$ (optionally conditioned on a latent goal $g$).
* Value/soft-Q: $V_\psi(z)$, $Q_\psi(z,a)$.
* Discount $\gamma\in(0,1)$, temperature $\alpha>0$.

---

## A) Searching in latent space

### A1) From rewards to a **desired latent-state distribution**

A control-as-inference view introduces “optimality” variables $O_t$ with
$$
p(O_t=1\mid z_t,a_t) \propto \exp\big(\tfrac{1}{\alpha} r(z_t,a_t)\big)
$$
The **posterior** over trajectories given optimality is
$$
p(\tau \mid O_{0:T}{=}1) \propto p(z_0) \prod_{t=0}^{T-1} p_\theta(z_{t+1} \mid z_t,a_t) \pi(a_t \mid z_t) \exp \big(\tfrac{1}{\alpha}r(z_t,a_t)\big)
$$
Marginalizing this posterior defines the **optimal visitation distribution** over latent states,
$$
\rho^*(z) = \sum_{t} \mathbb{P}(z_t{=}z\mid O_{0:T}{=}1)
$$
which you can **estimate** in practice in two equivalent ways:

1. **Soft value route.** Define the soft value and Q by
   $$
   \begin{aligned}
   V(z) &= \alpha \log \int \exp \Big(\tfrac{1}{\alpha} Q(z,a)\Big) da \\
   Q(z,a) &= r(z,a) + \gamma \mathbb{E}_{z' \sim p_\theta(\cdot\mid z,a)}[V(z')]
   \end{aligned}
   $$
   The induced **energy** $E(z) \equiv -V(z)$ gives an **unnormalized** density
   $$
   \tilde p^*(z) \propto \exp \big(\tfrac{1}{\alpha}V(z)\big)\quad\Rightarrow\quad
   p^*(z)=\frac{1}{Z}\exp \big(\tfrac{1}{\alpha}V(z)\big)
   $$
   which behaves like your “expected high-value latent distribution.”

2. **Occupancy-matching route.** Learn an energy $E_\eta(z)$ so that
   $$
   p^*_\eta(z) \propto \exp(-E_\eta(z))
   $$
   fits **high-return** state samples vs. replay negatives via contrastive/NCE:
   $$
   \min_\eta \mathbb{E}_{z\sim \text{top-return}}[\log \sigma(-E_\eta(z))]
   +\mathbb{E}_{z\sim \text{replay}}[\log \sigma(+E_\eta(z))]
   $$
   You can also weight positives by Monte-Carlo returns or soft advantages.

Either way, you obtain a **target** latent distribution (p^*(z)) peaking on high-value regions.

---

### A2) Constructing an action **path** from current state to the target distribution

Let the **goal** be either:

* a sampled **goal latent** $g \sim p^*(z)$, or
* the full **distribution** $p_g(z)\equiv p^*(z)$.

Two practical planners in latent space:

#### (i) Terminal-matching MPC (waypoint/goal reaching)

Pick horizon $H$. Optimize an action sequence $a_{0:H-1}$ to minimize the divergence between the predicted terminal distribution and the goal:
$$
\min_{a_{0:H-1}} \underbrace{D \left(p_\theta(z_H \mid z_0,a_{0:H-1}) \big|\big| p_g(z)\right)}_{\text{reach the goal distribution}} - \lambda\sum_{t=0}^{H-1} \mathbb{E}[r(z_t,a_t)]
$$
subject to $z_{t+1}\sim p_\theta(\cdot\mid z_t,a_t)$.
If $p_\theta$ is Gaussian in latent space, this KL becomes analytic; otherwise do sampling-based CEM/shooting with reparameterized rollouts of the world model.

#### (ii) Soft-return planning with terminal attraction

$$
\max_{a_{0:H-1}}
\mathbb{E}\Big[\sum_{t=0}^{H-1}\gamma^t r(z_t,a_t)
+\gamma^H\underbrace{U(z_H; p_g)}_{\text{terminal utility}}\Big]
$$
where $U(z_H;p_g) \equiv \tfrac{1}{\alpha}V(z_H)$ or $-D(\delta_{z_H}||p_g)$.
This recovers TD-MPC/Dreamer-style shooting but **pulls** the terminal latent toward $p_g$.

#### Executing with a policy (goal-conditioning)

At run time, either execute the first action of the optimized sequence (MPC), or condition a **goal-conditioned policy** on a (receding-horizon) goal:
$$
a_t \sim \pi_\psi(a\mid z_t,g),\quad g\sim\text{Planner}(z_t;p^*,p_\theta).
$$
Train $\pi_\psi$ with imagined rollouts, with **HER-style relabeling** where terminal/waypoint latents from the planner are treated as goals.

---

## B) Learning the full algorithm (world model + control)

We separate (1) **world-model learning**, and (2) **control learning** (IL / IRL / RL). All learning happens in latent space.

### B1) World-model learning (RSSM-style ELBO with rewards)

Learn $\theta,\phi$ from environment data $\mathcal{D}$ (and later from imagined rollouts):
$$
\mathcal{L}_{\text{WM}}(\theta,\phi)=
\sum_{t}\underbrace{\mathbb{E}
_{q_\phi}[\log p_\theta(o_t\mid z_t)]}_{\text{reconstruction}}
+\lambda_r \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(r_t\mid z_t,a_t)]}_{\text{reward head}} - \beta \underbrace{\mathrm{KL}\big(q_\phi(z_t \mid h_t) || p_\theta(z_t\mid z_{t-1},a_{t-1})\big)}_{\text{dynamics regularization}}
$$
(Use the deterministic+stochastic RSSM state; include value head if desired.)

---

### B2) Control learning: three unified procedures

#### (i) **Reinforcement Learning** inside the world model (soft actor-critic, latent)

Critic (soft-Q) loss:
$$
\mathcal{L}_{Q}=\mathbb{E}_{(z,a,r,z')}\Big[
\big(Q_\psi(z,a)-\underbrace{r+\gamma \mathbb{E}_{a'\sim \pi_\psi}[Q^-_\psi(z',a')-\alpha \log \pi_\psi(a'\mid z')] }_{y}\big)^2
\Big]
$$
Actor loss:
$$
\mathcal{L}_{\pi}=\mathbb{E}_{z\sim \text{rollouts},a\sim \pi_\psi}\big[\alpha\log \pi_\psi(a\mid z) - Q_\psi(z,a)\big]
$$
with $z,z'$ sampled from **imagined** transitions $p_\theta(z'|z,a)$ (plus real buffer).
Optional **goal-conditioning**: use $Q(z,a,g)$, $V(z,g)$, and provide planner-chosen $g$.

#### (ii) **Imitation Learning** in latent space

**Behavior Cloning (BC):**
$$
\min_{\pi_\psi} \mathbb{E}_{(z_t,a_t)\sim \tau_E}\big[-\log \pi_\psi(a_t\mid z_t)\big]
$$
**IQ-style single-Q IL** (soft RL as inference from demos):
Learn $Q$ to satisfy soft Bellman with **expert** TD targets:
$$
Q(z_t,a_t)\approx r_E(z_t,a_t) + \gamma\mathbb{E}_{z'|z_t,a_t} V(z'),\quad
V(z)=\alpha\log\int \exp\big(\tfrac{1}{\alpha}Q(z,a)\big)da
$$
where $r_E$ is **implicit** (no learned reward): the loss pushes $Q$ up on expert pairs and enforces soft Bellman consistency; the **actor** is $\pi(a|z)\propto \exp(\tfrac{1}{\alpha}Q(z,a))$.
This learns a control policy **purely from demos** yet is algebraically the same backbone as soft RL.

**Occupancy matching** (ValueDICE-style, latent): learn a critic-like $f_\omega(z,a)$ such that
$$
\min_\omega \mathbb{E}_{(z,a)\sim \rho*\pi}\big[\exp(f_\omega(z,a))\big] - \mathbb{E}_{(z,a)\sim \rho_E}\big[f*\omega(z,a)\big]
$$
and improve $\pi$ to increase $\mathbb{E}_{\rho_\pi}[f_\omega]$ subject to Bellman constraints in the world model.

#### (iii) **Inverse RL** (maximum-entropy IRL, latent)

Learn a reward $r_\varphi(z,a)$ that makes expert trajectories likely under soft-optimal control:
$$
\max_{\varphi} \underbrace{\mathbb{E}_{\tau\sim \mathcal{D}_E}\Big[\sum_t r_\varphi(z_t,a_t)\Big]}_{\text{fit experts}} - \underbrace{\log Z_\varphi}_{\text{partition}}
$$
where $Z*\varphi$ is computed by **soft value iteration** in the learned dynamics:
$$
  Q_\varphi(z,a)=r_\varphi(z,a)+\gamma\mathbb{E}_{z'|z,a}[V_\varphi(z')],\quad
  V_\varphi(z)=\alpha\log \int \exp\big(\tfrac{1}{\alpha}Q_\varphi(z,a)\big)da
$$
  Then **RL step**: improve $\pi$ with SAC-style losses using $r_\varphi$.
  This realizes “IRL → reward → RL” entirely **inside** the world model.

---

## C) Putting it together as a single loop

1. **Collect** seed data in the real env → update world model $(\theta,\phi)$ with $\mathcal{L}_{\text{WM}}$.
2. **Choose a control regime** (BC / IQ-IL / IRL / RL):

   * **BC/IL**: train $\pi$ (and possibly $Q,V$) on demos in latent.
   * **IRL**: learn $r_\varphi$ via soft VI in latent; then improve $\pi$.
   * **RL**: train $Q,V,\pi$ with SAC-style losses on imagined + real rollouts.
3. **Planner/Search** in latent (every step or intermittently):

   * Build $p^*(z)$ via $V$ (or via contrastive energy).
   * Sample a goal $g\sim p^*$ or keep the full $p_g$.
   * Optimize $a_{0:H-1}$ with one of the two objectives in **A2** to get a waypoint/terminal $z_H$ or just the first action.
   * Execute $a_0$, or condition $\pi(a\mid z,g)$.
4. **Update** replay buffers (real and imagined), repeat 1–3.

---

## D) Practical tips / knobs that matter

* **Uncertainty/bias control:** use ensembles or value-disagreement penalties in planning; cap horizon $H$.
* **Latent goals:** learn a goal encoder $g=f_\kappa(z)$ and a distance $d(g,g')$; set $U(z_H;p_g)=-d(f_\kappa(z_H),g)$.
* **Stability:** softly tie the planner’s terminal utility to $V(z)$ so search follows learned value ridges rather than raw reconstruction features.
* **Unified training:** even under IL/IRL, always keep the SAC-style critics and value heads consistent with the soft Bellman in the learned dynamics; this makes later RL fine-tuning seamless.

---

If you’d like, I can turn this into a short “theory → pseudocode” note (ELBO, soft-VI, SAC losses, goal-conditioned MPC objective) you can drop straight into a paper’s Methods section.
