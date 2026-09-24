# Pilot -> plan -> certify: frozen mathematical specification

This is a finite-menu planner for bounded IID anchor vectors with even IID
siblings. The menu, nuisance law, rho, loss, candidates, final safety delta,
planning delta, power target, support bounds, and maximum budget are fixed
before pilot observations. An independent design pilot chooses centering
constants and KL tangency points; an estimation pilot constructs simultaneous
confidence bounds for all scalar means, variances and third absolute moments
used by the complete menu. A final block is independent of both pilot blocks.

## Valid planning witnesses

1. Theorem 4: substitute upper mean/variance bounds into its sufficient
conditions. In its adaptive sensitivity increment substitute a lower V and
upper Var(Q); the increment decreases with V. Use upper V for the population
envelope. Never infer V=0 from an empirical zero.

2. Berry--Esseen: let a frozen final endpoint have the deterministic majorant
bar(Y)+b_N + sum_i[sqrt(2 t_i/N) s_i + 7 R_i t_i/(3(N-1))].
For generic fixed-dual DRO Y=M+Q/(4 eta), b=rho eta. For paired KL, use the
tangent of the concave function sqrt(rho min(Vcap,H_c(q))) at a design-pilot
q0. For separate KL, add the two tangents and use Y=Mdelta+a_C Q_C+a_F Q_F.
The tangent is a global upper bound, not a delta-method approximation.

Let muY_U, sigmaY_U, sigmaY_L, T3Y_U and sigma_i_U be simultaneous pilot
bounds. For beta_s>0, set
epsilon_BE = .56 T3Y_U / (sigmaY_L^3 sqrt(N)),
q = 1-(1-p)/J + d beta_s + epsilon_BE.
If q<1 and
muY_U+b_N + sum_i[sqrt(2 t_i/N)(sigma_i_U+
R_i sqrt(2 log(1/beta_s)/(N-1)))+7 R_i t_i/(3(N-1))]
+ Phi^{-1}(q) sigmaY_U/sqrt(N) < 0,
then the endpoint is negative with probability at least 1-(1-p)/J.
This follows from sample-SD concentration and Berry--Esseen with a union
bound; no independence between the sample mean and SD is assumed. Use only
p>=1/2, so the normal quantile and its variance multiplier are nonnegative.
A zero variance lower bound disables this witness; Theorem 4 remains valid.

The planner selects the lowest N feasible through either witness, then the
lowest KN, then frozen deterministic method/ledger tie breakers. It returns
INFEASIBLE if the finite menu contains no witnessed plan. No outcomes are
collected or declared certified on that branch. Generic eta is selected
before the fresh final block, so final safety does not pay a search penalty
for pilot-only choices. All final candidate decisions still pay J.

## Three distinct probabilities

For every pilot history, including a confidence-set failure, the chosen
final procedure is safe with probability >=1-delta (conditional on pilot).
With pilot probability >=1-delta_plan, every returned feasible plan has
conditional final family-certification power >=p. On this good-pilot event
an infeasible result has no power promise. If feasibility holds almost
surely, unconditional power >=(1-delta_plan)p; generally it is at least
p P(good pilot AND feasible). The theorem does not promise feasibility.

## What near-oracle means

The oracle is explicitly the minimum menu budget satisfying these SAME
deterministic sufficient witnesses with true population moments and limiting
design tangencies. It is not the Gaussian forecast or the actual minimum
power threshold. For a fixed finite menu, nonzero variances for enabled BE
witnesses, consistent pilot moments, and strict separation of every menu
inequality from zero and q from one, estimated witnesses converge uniformly.
Thus the selected budget equals this sufficient-witness oracle eventually
in probability (and the ratio tends to one). Feasibility must hold at the
oracle and tangent limits must be continuous (exclude cap kinks).
For an isolated continuous-budget root with derivative bounded away from
zero, local witness error epsilon gives budget error O(epsilon/|F'(N*)|),
plus grid rounding. This is a local statement, not a global optimality claim.

## Local information benchmark

In the frozen one-parameter subclass A~Bernoulli(theta), all conditional
factual/nuisance distributions are theta-independent. For theta1 with
Psi=-gamma and theta0 with Psi=0, a safe procedure of (unconditional) power
q>delta and total independent-anchor stopping time T obeys
E_theta1[T] >= kl(q,delta)/kl(theta1,theta0).
Proof: likelihood chain rule gives KL(full stopped transcript)<=E[T] kl;
binary data processing for the event 'certify' gives the lower bound.
It includes pilot anchors and charged aborted plans. K nuisance siblings
carry no extra information about theta once A is known. Observed losses
cannot be more informative than the latent anchor transcript. This is a
two-point local benchmark, not global minimax optimality. As gamma->0 it
scales 2 theta0(1-theta0) Psi'(theta0)^2 kl(q,delta)/gamma^2.

## Validation integrity

Implementation and this specification precede new controlled outcomes.
Prospective simulation means new held-out controlled laws, pilot-only budget
locks, then exactly the locked N fresh IID anchors per final replication.
Known support enclosures are supplied, but law probabilities and population
moments are not planner inputs. This extra known-range assumption and pilot
cost must be reported. No new expensive model experiment precedes proof.
