# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: CHME5137
#     language: python
#     name: chme5137
# ---

# %% [markdown]
# # Rabbits and foxes, by kinetic Monte Carlo
#
# **CHME 5137, Lecture 10, Tuesday October 13, 2026.** Started in class, finished for
# homework.
#
# *Written for 2026 by Claude Code (Claude Opus 5.5) from Prof. West's KMC question of
# earlier years, and checked by Prof. West. The problem is from Chapter 1 of H. Scott
# Fogler's "Essentials of Chemical Reaction Engineering".*
#
# ---
#
# Same farm, same rate constants: 400 rabbits and 200 foxes at the start, and
#
# $$\frac{dR}{dt} = k_1 R - k_2 R F \qquad\qquad \frac{dF}{dt} = k_3 R F - k_4 F$$
#
# with $k_1 = 0.015$ day$^{-1}$, $k_2 = 0.00004$ day$^{-1}$ foxes$^{-1}$,
# $k_3 = 0.0004$ day$^{-1}$ rabbits$^{-1}$ and $k_4 = 0.04$ day$^{-1}$.
#
# In the `solve_ivp` homework you found that the foxes fall to **under two** in each
# trough. The ODEs don't mind: 1.6 foxes carry on and breed. Real foxes can't be
# fractional, and two foxes can easily both die before they meet a rabbit. So now we
# treat every rabbit and fox as an individual, and every birth and death as a random
# event. That is **kinetic Monte Carlo** (KMC).
#
# ### The events
#
# There are four events. Each has a rate, in events per day, which changes as the
# populations change:
#
# | Event | Rate | Effect |
# |---|---|---|
# | a rabbit is born | $k_1 R$ | $R \to R + 1$ |
# | a rabbit is eaten | $k_2 R F$ | $R \to R - 1$ |
# | a fox is born | $k_3 R F$ | $F \to F + 1$ |
# | a fox dies | $k_4 F$ | $F \to F - 1$ |
#
# The rates are easy to calculate, so we use the **rejection-free** algorithm, which you
# worked out in class from
# [Wikipedia's Kinetic Monte Carlo page](https://en.wikipedia.org/wiki/Kinetic_Monte_Carlo).
# Writing it down in your own words is the first part of task 1.
#
# ### How to work
#
# - **Commit as you go.** After each numbered task works, Restart & Run All, then
#   stage, commit with a message that says what you did, and push. At least one commit
#   per task. Your commit history is part of what gets looked at. **Commit the `.ipynb`
#   and its paired `.py` together**: the `.py` is the one whose diffs you can read on
#   GitHub. If saving the notebook doesn't update the `.py`, see *Setting up Jupytext* on
#   Canvas.
# - **Answer the questions in words**, in the markdown cells marked *Answer*.
#   Double-click a cell to edit it.
# - **Random numbers:** use a NumPy generator, `rng = np.random.default_rng(seed)`,
#   not the legacy `np.random.*` functions. Give it a seed, so that Restart & Run All
#   gives the same answers every time. Check what interval each function returns,
#   and what its arguments mean, before you trust it.
# - **Let bugs shout.** If a population goes negative or an event isn't one of the
#   four, `raise` an error. Don't hide it with `max(0, F)`: that turns a bug you'd
#   notice into a wrong answer you won't.

# %%
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

k1 = 0.015  # 1/day
k2 = 0.00004  # 1/day/fox
k3 = 0.0004  # 1/day/rabbit
k4 = 0.04  # 1/day

R0, F0 = 400, 200  # rabbits and foxes at t = 0
t_end = 600  # days

# %% [markdown]
# The deterministic answer, for comparison. This is the `solve_ivp` homework's
# solution, at a tight tolerance.

# %%
def dydt(t, y):
    R, F = y
    return [k1 * R - k2 * R * F, k3 * R * F - k4 * F]

ode = solve_ivp(dydt, (0, t_end), [R0, F0], rtol=1e-10, atol=1e-8, dense_output=True)
t_ode = np.linspace(0, t_end, 3001)
R_ode, F_ode = ode.sol(t_ode)

plt.plot(t_ode, R_ode, label="rabbits (ODE)")
plt.plot(t_ode, F_ode, label="foxes (ODE)")
plt.xlabel("time (days)")
plt.ylabel("population")
plt.legend()
plt.show()

# %% [markdown]
# ## 1. One trajectory
#
# **(a) The algorithm, in your own words.** Before you write any code, write the
# rejection-free algorithm as numbered steps, for rabbits and foxes: what each of
# Wikipedia's symbols ($k$, $r_{ki}$, $R_{ki}$, $Q_k$, $u$, $u'$) becomes here, how the
# next event is chosen, and how far the clock moves. Then work the **first step by
# hand**, from 400 rabbits and 200 foxes, with $u = 0.5$ and $u' = 0.3$: which event
# happens, and how many minutes pass?

# %% [markdown]
# *Answer:*  I'll just define variables first (mostly for my own sake) and then go stepwise after that... A state **k** would be the given number of rabbits and foxes at a given time. The transition rates **r_ki** would be the rates for each event (total of four: being fox birth, fox death, rabbit birth, rabbit death). The cumulative function **R_ki** is the summation of all the individual rates up to and including a given point "i". This is related to the total sum of all rates (fully going to the end), which would be **Q_k** (sum of all the rates, think of numberline shown on blackboard in class). **u** is a random number generated between zero and one, which is used to choose which event happens (points to which event on the numberline), with **u'** being used for the next time step (deltaT = ln(1/u') / Q_k).
#
# Stepwise starting from the top: 
#
# 1. time = 0, get an initial state k (400 rabbits, 200 foxes)
# 2. get starting rates **r_ki** from 1 to 4: (k_1*R, k_2*R*F, k_3*R*F, k_4*F) resulting in (0.015*400) , (0.00004*200*400) , (0.0004*200*400) , (0.04*200)
# 3. **R_k1** = (0.015*400), **R_k2** = (0.00004*200*400) + **R_k1**, **R_k3** = (0.0004*200*400) + **R_k2**, **R_k4** = (0.04*200) + **R_k3** and thus **Q_k** = **R_k4** = 49.2 events/day
# 4. A uniform random number **u** is drawn from (0,1] and then is multiplied by **Q_k**, that result is then the event that lands on (numberline) is carried out
# 5. a new random uniform number is drawn **u'**, and the time step is calculated by ln(1/u') / (Q_k). this is then added with t_current = t + deltaT
# 6. This is then repeated (loop back to step 2), with new rates being calclated based on the new environment since the last event was carried out
#
#
# working out by hand:
#
# **R_k1** = 6
# **R_k2** = 9.2
# **R_k3** = 41.2
# **R_k4** = 49.2
# **Q_k** = 49.2 (see above, events/day)
#
# if **u** = 0.5, then **Q_k** * (u = 0.5) = 24.6, which is **R_k2** (9.2) < 24.6 < **R_k3** (41.2), so the event is that a fox is born (event 3). So the next state is (400,201)
#
# with **u'** = 0.3, the time step is = ln(1/0.3) / 49.2 = ~0.0245 days which is 35.28 minutes
#

# %% [markdown]
# **(b) The function.** Write
#
# ```python
# def kmc(R0, F0, t_end, rng):
#     ...
#     return times, rabbits, foxes   # three NumPy arrays, one entry per event
# ```
#
# that runs your algorithm from $t = 0$ until the clock passes `t_end`. If the total
# rate is ever zero, nothing more can happen, so stop there.
#
# Run it once. How many events were there, and how long did it take? Plot rabbits and
# foxes against time, on top of the ODE solution. (`plt.step(times, foxes, where="post")`
# draws a population that jumps, rather than one that slopes.) Run it a few more times
# with different seeds. How does a KMC trajectory differ from the ODE solution?

# %%
rng = np.random.default_rng(67)
def kmc(R0, F0, t_end, rng):
    t=0.0
    R,F = R0, F0
    times = [t]
    rabbits = [R]
    foxes = [F]
    while t < t_end:
        r = np.array([k1*R, k2*R*F, k3*R*F, k4*F])
        R_tot = np.cumsum(r)
        Q = R_tot[-1]
        if Q == 0:
            break
        u = rng.random()
        i = np.searchsorted(R_tot, u*Q)
        if i == 0:
            R+= 1
        elif i ==1:
            R -= 1
        elif i ==2:
            F += 1
        elif i ==3:
            F -= 1
        else:
            raise ValueError(f"unknown event index {i}")
        if R < 0 or F < 0:
            raise ValueError(f"negative population: R={R}, F={F}, t={t}")
        u2 = rng.random()
        t += (np.log(1/u2)) / Q
        times.append(t)
        rabbits.append(R)
        foxes.append(F)
    return np.array(times), np.array(rabbits), np.array(foxes)

start = time.perf_counter()
times, rabbits, foxes = kmc(R0, F0, t_end, rng)
print(len(times) - 1, "events in", round(time.perf_counter() - start, 2), "s")

plt.figure()
plt.plot(t_ode, R_ode, label="rabbits (ODE)")
plt.plot(t_ode, F_ode, label="foxes (ODE)")
plt.xlabel("time (days)")
plt.ylabel("population")

plt.step(times, rabbits, where="post", label="rabbits (KMC)")
plt.step(times, foxes, where="post", label="foxes (KMC)")
plt.legend()


# %% [markdown]
# *Answer:* The KMC differs from the ODE after the about day 200, where the KMC never has a rebound in the fox population (presumably the r_foxborn decreases and it is not a statistically likely event that they will get out of an extinction vortex situation within the populations, therefore rabbit population increases exponentially with no predator there). There were 23462 events in 0.24s (see above in output line)

# %% [markdown]
# ## 2. Many trajectories, and where they end up
#
# Plot the foxes from 20 trajectories on one graph (thin lines, `lw=0.5`), with the ODE
# solution on top.
#
# - What do the trajectories have in common, and where do they spread apart?
# - What happens to a trajectory once the foxes reach zero? Once the rabbits reach
#   zero? Why can't it ever recover?
# - Where in the cycle do the foxes usually die out, and why there? (Your answer to
#   task 5 of the `solve_ivp` homework is relevant.)

# %%
rng = np.random.default_rng(67)
plt.figure()
for n in range(20):
    times, rabbits, foxes = kmc(R0, F0, t_end, rng)
    plt.step(times, foxes, where="post", lw=0.5)
plt.plot(t_ode, F_ode, label='ODE', lw=2)
plt.xlabel('time (days')
plt.ylabel('population')
plt.legend()
plt.title('Foxes')

plt.figure()
for n in range(20):
    times, rabbits, foxes = kmc(R0, F0, t_end, rng)
    plt.step(times, rabbits, where="post", lw=0.5)
plt.plot(t_ode, R_ode, label = 'ODE', lw = 2)
plt.xlabel('time (days')
plt.ylabel('population')
plt.legend()
plt.title('Rabbits')


# %% [markdown]
# *Answer:* The trajectories start out very similar to one another at times < ~250 days. There is a large differential after that due to small numbers of foxes being present, so extinction becomes the most likely outcome, and the fox population rebounding is less likely event. The trajectory at F=0 or R=0 will never rebound, because in the equations for new animals, if either are zero, then the r_newfox = 0, and r_newrabbit = 0, meaning this equation takes into account that you cannot create an animal from nothing (fox birth = k3RF, fox death = k4F, rabbits death = k2FR all become zero when F = 0, but not rabbit birth, which is k1R and not coupled to fox populations resulting in exponential rise of rabbit population when no foxes). The foxes are most likely to die out in the fox "trough" at around the same amount of time as listed before (~250 - 300 days), where the fox_min is ~1.59 foxes. With such minimal fox(es) left, a short run of deaths with no births wipes them all out of the picture

# %% [markdown]
# ## 3. The second fox peak
#
# The ODE's foxes peak at about day 32, and again at about day 428, at about 2804
# foxes each time. When is the second peak in a KMC trajectory, and how high?
#
# A noisy trajectory has lots of tiny local maxima, so we need a definition. Use this
# one: **the largest number of foxes after day 200, counted as a peak only if it is
# above 100 foxes and isn't within 5 days of either end of the window** (day 200 or
# day 600). Write a function `second_peak(times, foxes)` that returns `(time, height)`,
# or `None` if there isn't one.
#
# Run 100 trajectories. In what fraction is there a second peak? For those that have
# one, give the mean and median of its time and of its height, and the interquartile
# range (25th to 75th percentile, `np.percentile`) of each. Plot histograms.
#
# How do these compare with the ODE's second peak? If they differ, why? And what does
# the definition get wrong? Find a trajectory where it picks something you wouldn't
# call "the second peak", if you can.

# %%

# %% [markdown]
# *Answer:*

# %% [markdown]
# ## 4. How many runs are enough?
#
# Every number in task 3 came from random runs, so it has its own uncertainty. Repeat
# task 3 with 100, 300, 1000 and 3000 trajectories. (Time it first, and work out how
# long 3000 will take before you start.)
#
# - How do the mean second-peak time and height change with the number of runs?
# - Work out the **standard error** of each mean, $s / \sqrt{n}$, where $s$ is the
#   standard deviation and $n$ the number of trajectories with a peak. How does it
#   scale with $n$?
# - Give the expected time and height of the second peak, and their interquartile
#   ranges, to a precision you can justify. Say how you justified it.

# %%

# %% [markdown]
# *Answer:*

# %% [markdown]
# ## 5. How likely is it that the foxes die out?
#
# From your largest set of runs, what fraction have **no foxes left at day 600**?
# That fraction $p$ estimates a probability. Its standard error is
# $\sqrt{p(1-p)/N}$ for $N$ runs.
#
# - Give the probability that the foxes die out before 600 days, with its uncertainty.
# - How many runs would you need to know it to within ±1 percentage point? To within
#   ±0.1?
# - Plot a histogram of the times at which the foxes die out. When does it usually
#   happen?

# %%

# %% [markdown]
# *Answer:*

# %% [markdown]
# ## 6. Your project
#
# One paragraph. Is there any part of your project where the numbers are small
# enough, or the events discrete and random enough, that a continuum model (an ODE or
# a PDE) could mislead you, the way the ODE does about the foxes? Think about
# molecules on a surface, cells, bubbles, particles, defects, failures. If there is,
# what would the "events" and their rates be? If there isn't, why is a continuum model
# good enough?

# %% [markdown]
# *Answer:*

# %% [markdown]
# ## Stretch problems
#
# **(a) Faster.** Time one trajectory. Then find where the time goes (`%prun`, or
# time the pieces). Try Python's `random.random()` in place of `rng.random()`,
# appending to lists versus filling a preallocated array, and recording only every
# 100th event. How much faster can you make it?
#
# **(b) Bigger, on Explorer.** The `.py` paired with this notebook is a script.
# Make a copy that runs 10,000 trajectories and saves the second-peak times and heights
# and the extinction times with `np.savetxt`, and run it as a batch job on Explorer,
# as in Lecture 9. Do your answers to tasks 4 and 5 hold up?
#
# **(c) Is the random clock worth it?** Wikipedia notes that advancing the clock by
# $\Delta t = 1/Q_k$ every step, instead of an exponentially distributed $\Delta t$, gives
# "the same *average* time scale" (and that note is marked *citation needed*). Test it.
# Make a copy of your `kmc` function that uses $\Delta t = 1/Q$, and run both versions
# many times. Compare the probability that the foxes die out by day 600, the fraction of
# runs with a second peak, and the mean time and height of that peak, each with its
# standard error. Which differences are bigger than their uncertainty? How many runs did
# you need before you could tell? (Several thousand of each; (a) and (b) help.) Why might
# some quantities change and others not?

# %%

# %% [markdown]
# ---
#
# ### When you're done
#
# Restart & Run All. Check that every cell ran, in order, with no errors. Commit and
# push. Then check on GitHub that your fork shows your latest commit, and that the
# notebook displays with its plots.

# %%
