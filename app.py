import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mc_analysis import load_paired_data, make_lagged_pairs, fit_ols, monte_carlo

st.set_page_config(page_title="GEOG 524 | Monte Carlo weather", layout="wide")
st.title("Monte Carlo sampling, regression and temporal dependence")
st.caption("Fredericton and Saint John daily mean temperatures · May–August 2019 · Environment and Climate Change Canada source files")

with st.sidebar:
    st.header("Experiment controls")
    lag = st.slider("Time lag (days)", min_value=-30, max_value=30, value=0,
                    help="Positive values pair Fredericton on day t with Saint John on day t−lag. This changes the scientific relationship being fitted; it does not remove autocorrelation.")
    sample_size = st.slider("Paired days per experiment", 5, 100, 30, 1)
    repetitions = st.slider("Monte Carlo repetitions", 100, 2000, 500, 100)
    scheme = st.radio("Resampling scheme", ["IID day resampling", "Contiguous block resampling"])
    block_length = st.slider("Block length (days)", 2, 30, 7, 1,
                             disabled=(scheme == "IID day resampling"))
    intercept = st.checkbox("Fit an intercept", value=True)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

raw = load_paired_data()
paired = make_lagged_pairs(raw, lag)
if len(paired) < 10:
    st.error("Too few complete pairs at this lag. Choose a smaller absolute lag.")
    st.stop()

full_model = fit_ols(paired, intercept=intercept)
slope_full = float(np.asarray(full_model.params)[-1])
sim = monte_carlo(paired, sample_size, repetitions, block_length, scheme, intercept, int(seed))
if sim.empty:
    st.error("No successful model fits. Try a larger sample size.")
    st.stop()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Complete paired days", f"{len(paired)}")
c2.metric("Full-record slope", f"{slope_full:.3f}")
c3.metric("Monte Carlo mean slope", f"{sim.slope.mean():.3f}")
c4.metric("Monte Carlo slope SD", f"{sim.slope.std():.3f}")

st.subheader("1 · The paired observations")
fig, ax = plt.subplots(figsize=(7,4))
ax.scatter(paired["STJ_T"], paired["FRD_T"], alpha=.65)
x = np.linspace(paired.STJ_T.min(), paired.STJ_T.max(), 100)
b0 = float(np.asarray(full_model.params)[0]) if intercept else 0
ax.plot(x, b0+slope_full*x)
ax.set_xlabel("Saint John mean temperature (°C)")
ax.set_ylabel("Fredericton mean temperature (°C)")
ax.set_title(f"Regression at time lag {lag:+d} day(s)")
ax.grid(alpha=.2)
st.pyplot(fig); plt.close(fig)

st.subheader("2 · Sampling distribution of the estimated slope")
fig, ax = plt.subplots(figsize=(8,4))
ax.hist(sim.slope, bins=30, edgecolor="black", alpha=.8)
ax.axvline(slope_full, linestyle="--", linewidth=2, label="Full-record slope")
ax.axvline(sim.slope.mean(), linestyle=":", linewidth=2, label="Monte Carlo mean")
ax.set_xlabel("Estimated slope"); ax.set_ylabel("Number of simulations")
ax.legend(); ax.grid(alpha=.2)
st.pyplot(fig); plt.close(fig)

qlo,qhi = sim.slope.quantile([.025,.975])
st.write(f"Central 95% of simulated slopes: **{qlo:.3f} to {qhi:.3f}**. This is a Monte Carlo resampling interval for this exercise, not automatically a formal confidence interval.")

st.subheader("3 · How estimates vary across repeated samples")
fig, ax = plt.subplots(figsize=(8,3.5))
ax.plot(sim.slope.to_numpy(), linewidth=.8)
ax.axhline(slope_full, linestyle="--", label="Full-record slope")
ax.set_xlabel("Simulation"); ax.set_ylabel("Estimated slope")
ax.legend(); ax.grid(alpha=.2)
st.pyplot(fig); plt.close(fig)

with st.expander("Interpretation and important cautions", expanded=True):
    st.markdown("""
    - **Lag changes the question.** At lag 0, same-day temperatures are paired. At a positive lag, Fredericton on day *t* is paired with Saint John on an earlier day. A different slope may reflect the physical timing of weather systems, not a statistical correction.
    - **IID resampling assumes independent, exchangeable rows.** Daily temperatures commonly have persistence: nearby days are more alike than distant days. Resampling individual days destroys that local sequence.
    - **Block resampling keeps short runs together.** It can better retain local temporal structure, but results depend on block length and this short one-season record. It is a teaching demonstration, not a definitive inferential procedure.
    - **Autocorrelation and regression.** Serially correlated errors do not necessarily bias an OLS slope when the conditional mean model is correctly specified and regressors are exogenous, but they invalidate the usual independent-error standard-error calculations and can make conventional tests and intervals misleading. Dependence in the predictor, omitted seasonal structure, or lagged relationships can also change the estimated slope.
    - **One year is not a climatology.** The May–August 2019 subset is used because it follows the supplied teaching example. Conclusions should not be generalized to all years or seasons.
    """)
st.caption("Source: Environment and Climate Change Canada daily climate CSV records supplied with the course archive. Missing paired days are omitted after date alignment.")
