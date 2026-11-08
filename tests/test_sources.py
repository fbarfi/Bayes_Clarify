"""Draws from other samplers: tables, NumPyro-like and CmdStanPy-like objects."""

import numpy as np
import pandas as pd
import pytest

import bayes_clarify as bc


def _table(one_based=True, dots=False):
    rng = np.random.default_rng(0)
    S = 50
    beta = rng.normal(size=(S, 3))
    gam = rng.normal(size=(S, 2, 2))
    o = 1 if one_based else 0
    cols = {".chain": np.repeat([1, 2], S // 2), ".iteration": np.arange(S), ".draw": np.arange(S),
            "lp__": rng.normal(size=S), "b_Intercept": rng.normal(size=S),
            "r_country[USA,Intercept]": rng.normal(size=S)}
    for j in range(3):
        cols[f"beta.{j + o}" if dots else f"beta[{j + o}]"] = beta[:, j]
    for i in range(2):
        for k in range(2):
            cols[f"gam.{i + o}.{k + o}" if dots else f"gam[{i + o},{k + o}]"] = gam[:, i, k]
    return pd.DataFrame(cols), beta, gam


@pytest.mark.parametrize("one_based,dots", [(True, False), (False, False), (True, True)])
def test_table_gathers_indexed_columns(one_based, dots):
    df, beta, gam = _table(one_based, dots)
    d = bc.draws_from_table(df)
    assert np.allclose(d["beta"], beta) and np.allclose(d["gam"], gam)
    assert d["b_Intercept"].shape == (50,)
    assert "r_country[USA,Intercept]" in d                 # non-numeric index: scalar
    assert not any(k.startswith(".") or k == "lp__" for k in d)


def test_table_from_csv_with_comments(tmp_path):
    df, beta, _ = _table()
    p = tmp_path / "draws.csv"
    p.write_text("# Stan comment\n" + df.to_csv(index=False))
    d = bc.draws_from_table(p, var_names=["beta"])
    assert list(d) == ["beta"] and np.allclose(d["beta"], beta)


def test_table_rejects_gaps():
    df, _, _ = _table()
    with pytest.raises(ValueError):
        bc.draws_from_table(df.drop(columns=["beta[2]"]))


def test_numpyro_like():
    class MCMC:
        def get_samples(self, group_by_chain=False):
            return {"a": np.ones(10), "b": np.zeros((10, 2))}
    d = bc.draws_from_numpyro(MCMC())
    assert d["b"].shape == (10, 2)
    assert set(bc.draws_from_numpyro({"a": np.ones(3)}, ["a"])) == {"a"}


def test_cmdstanpy_like():
    class Fit:
        def stan_variables(self):
            return {"a": np.ones(8), "b": np.zeros((8, 3))}

        def stan_variable(self, name):
            return self.stan_variables()[name]
    assert bc.draws_from_cmdstanpy(Fit())["b"].shape == (8, 3)
    assert list(bc.draws_from_cmdstanpy(Fit(), ["a"])) == ["a"]


def test_table_draws_feed_a_posterior():
    rng = np.random.default_rng(1)
    S, N = 200, 300
    df = pd.DataFrame({"a": rng.normal(-0.5, .1, S), "b[1]": rng.normal(.8, .1, S)})
    data = pd.DataFrame({"x": rng.normal(size=N)})
    post = bc.Posterior.from_linear(bc.draws_from_table(df), data, coef="b",
                                    intercept="a", columns=["x"], link="logit")
    p = 1 / (1 + np.exp(-(df["a"].to_numpy()[:, None] + df["b[1]"].to_numpy()[:, None] * data["x"].to_numpy())))
    hand = (p * (1 - p) * df["b[1]"].to_numpy()[:, None]).mean(1)
    assert np.allclose(bc.sim_ame(post, "x").values[:, 0], hand, atol=1e-6)
