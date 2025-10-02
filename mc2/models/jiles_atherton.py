import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from mc2.models.RNN import GRU, GRUwLinear

# Adjoint method for estimating Jiles-Atherton hysteresis
# model parameters


# On physical aspects of the Jiles-Atherton hysteresis models
def softclip(x, limit=1e8):
    return limit * jnp.tanh(x / limit)


def softplus(x, eps=1e-6):
    return jnp.log1p(jnp.exp(x)) + eps


class JilesAthertonStatic(eqx.Module):
    Ms_param: jax.Array
    a_param: jax.Array
    alpha_param: jax.Array
    k_param: jax.Array
    c_param: jax.Array
    mu_0: float = 4e-7 * jnp.pi
    tau: float = 1 / 16e6

    @property
    def a(self):
        return 2000 * jax.nn.sigmoid(self.a_param)

    @property
    def alpha(self):
        return 1e-2 * jax.nn.sigmoid(self.alpha_param)

    @property
    def k(self):
        return 1000.0 * jax.nn.sigmoid(self.k_param)

    @property
    def c(self):
        return jax.nn.sigmoid(self.c_param)

    @property
    def Ms(self):
        return 6e6 * jax.nn.sigmoid(self.Ms_param)

    def __init__(self, key, **kwargs):
        super().__init__(**kwargs)
        k_key, alpha_key, c_key, Ms_key, a_key = jax.random.split(key, 5)
        self.k_param = jax.random.uniform(k_key, ()) * 0.05 + 0.5
        self.c_param = jax.random.uniform(c_key, ()) * 0.05 + 0.5
        self.a_param = jax.random.uniform(a_key, ()) * 0.05 + 0.5
        self.Ms_param = jax.random.uniform(Ms_key, ()) * 0.05 + 0.5
        self.alpha_param = jax.random.uniform(alpha_key, ()) * 0.001 + 0.0002

    def coth(self, x):
        return 1 / jnp.tanh(x)

    def coth_stable(self, x):
        eps = 1e-7
        x = jnp.where(jnp.abs(x) < eps, eps * jnp.sign(x), x)
        return 1 / jnp.tanh(x)

    # Updated dM_dH function
    def dM_dH(self, H, M, dB_dt):
        H_e = H + self.alpha * M
        M_an = self.Ms * (self.coth_stable(H_e / self.a) - self.a / H_e)
        delta_m = 0.5 * (1 + jnp.sign((M_an - M) * dB_dt))

        dM_an_dH_e = self.Ms / self.a * (1 - (self.coth_stable(H_e / self.a)) ** 2 + (self.a / H_e) ** 2)
        delta = jnp.sign(dB_dt)

        numerator = delta_m * (M_an - M) + self.c * self.k * delta * dM_an_dH_e
        denominator = self.k * delta - self.alpha * numerator

        dM_dH = numerator / denominator

        return dM_dH

    def ode(self, B, B_next, H):
        dB_dt_est = (B_next - B) / self.tau
        M = B / self.mu_0 - H
        dM_dH = self.dM_dH(H, M, dB_dt_est)
        dM_dB = dM_dH / (self.mu_0 * (1 + dM_dH))
        dM_dt = dM_dB * dB_dt_est
        dH_dt = 1 / self.mu_0 * dB_dt_est - dM_dt

        dH_dt = softclip(dH_dt, limit=1e8)

        return dH_dt, dB_dt_est

    def step(self, H, B, B_next):
        dH_dt, _ = self.ode(B, B_next, H)
        H_next = H + self.tau * dH_dt
        B_next = B_next
        return H_next, B_next

    def __call__(self, H0, B_seq):

        def body_fun(carry, B_pair):
            H_prev = carry
            B_curr, B_next = B_pair
            H_next, _ = self.step(H_prev, B_curr, B_next)
            return H_next, H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        _, H_seq = jax.lax.scan(body_fun, H0, B_pairs)
        H_seq = jnp.concatenate([jnp.array([H0]), H_seq], axis=0)
        return H_seq


class JilesAthertonStatic2(eqx.Module):
    Ms_param: jax.Array
    a_param: jax.Array
    alpha_param: jax.Array
    k_param: jax.Array
    c_param: jax.Array
    k1_param: jax.Array
    k2_param: jax.Array
    d_param: jax.Array
    rho_param: jax.Array
    mu_0: float = 4e-7 * jnp.pi
    tau: float = 1 / 16e6

    @property
    def a(self):
        return 2000 * jax.nn.sigmoid(self.a_param)

    @property
    def alpha(self):
        return 1e-2 * jax.nn.sigmoid(self.alpha_param)

    @property
    def k(self):
        return 1000.0 * jax.nn.sigmoid(self.k_param)

    @property
    def c(self):
        return jax.nn.sigmoid(self.c_param)

    @property
    def Ms(self):
        return 6e6 * jax.nn.sigmoid(self.Ms_param)

    @property
    def d(self):
        return 1e-6 + 5e-4 * jax.nn.softplus(self.d_param)

    @property
    def rho(self):
        return 0.1e-6 + 0.9e-6 * jax.nn.softplus(self.rho_param)

    @property
    def k1(self):
        return 0.1 + 0.9 * jax.nn.softplus(self.k1_param)

    @property
    def k2(self):
        return 0.0 + 1.0 * jax.nn.softplus(self.k2_param)

    def __init__(self, key, **kwargs):
        super().__init__(**kwargs)
        k_key, alpha_key, c_key, Ms_key, a_key, d_key, rho_key, k1_key, k2_key = jax.random.split(key, 9)
        self.k_param = jax.random.uniform(k_key, ()) * 0.05 + 0.5
        self.c_param = jax.random.uniform(c_key, ()) * 0.05 + 0.5
        self.a_param = jax.random.uniform(a_key, ()) * 0.05 + 0.5
        self.Ms_param = jax.random.uniform(Ms_key, ()) * 0.05 + 0.5
        self.alpha_param = jax.random.uniform(alpha_key, ()) * 0.001 + 0.0002
        self.d_param = jax.random.uniform(d_key, ()) * 0.05 + 0.5
        self.rho_param = jax.random.uniform(rho_key, ()) * 0.05 + 0.5
        self.k1_param = jax.random.uniform(k1_key, ()) * 0.05 + 0.5
        self.k2_param = jax.random.uniform(k2_key, ()) * 0.05 + 0.5

    def coth(self, x):
        return 1 / jnp.tanh(x)

    def coth_stable(self, x):
        eps = 1e-7
        x = jnp.where(jnp.abs(x) < eps, eps * jnp.sign(x), x)
        return 1 / jnp.tanh(x)

    # Updated dM_dH function
    def dM_dH(self, H, M, dB_dt, B):
        H_e = H + self.alpha * M
        M_an = self.Ms * (self.coth_stable(H_e / self.a) - self.a / H_e)
        delta_m = 0.5 * (1 + jnp.sign((M_an - M) * dB_dt))

        dM_an_dH_e = self.Ms / self.a * (1 - (self.coth_stable(H_e / self.a)) ** 2 + (self.a / H_e) ** 2)
        delta = jnp.sign(dB_dt)

        # Dynamic extension (Bertotti DHM)
        K_exc = self.k1 * (1.0 + self.k2 * (B**2))
        H_clas = (self.d**2) / (12 * self.rho) * dB_dt
        H_exc = K_exc * self.d * jnp.sqrt(jnp.abs(dB_dt)) * delta
        H_dyn = H_clas + H_exc

        numerator = delta_m * (M_an - M) + self.c * self.k * delta * dM_an_dH_e
        denominator = self.k * delta + H_dyn - self.alpha * numerator

        dM_dH = numerator / denominator

        return dM_dH

    def ode(self, B, B_next, H):
        dB_dt_est = (B_next - B) / self.tau
        M = B / self.mu_0 - H
        dM_dH = self.dM_dH(H, M, dB_dt_est, B)
        dM_dB = dM_dH / (self.mu_0 * (1 + dM_dH))
        dM_dt = dM_dB * dB_dt_est
        dH_dt = 1 / self.mu_0 * dB_dt_est - dM_dt

        dH_dt = softclip(dH_dt, limit=1e8)

        return dH_dt, dB_dt_est

    def step(self, H, B, B_next):
        dH_dt, _ = self.ode(B, B_next, H)
        H_next = H + self.tau * dH_dt
        B_next = B_next
        return H_next, B_next

    def __call__(self, H0, B_seq):

        def body_fun(carry, B_pair):
            H_prev = carry
            B_curr, B_next = B_pair
            H_next, _ = self.step(H_prev, B_curr, B_next)
            return H_next, H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        _, H_seq = jax.lax.scan(body_fun, H0, B_pairs)
        H_seq = jnp.concatenate([jnp.array([H0]), H_seq], axis=0)
        return H_seq


import jax
import jax.numpy as jnp
import equinox as eqx
from mc2.data_management import EXPERIMENT_LOGS_ROOT, MODEL_DUMP_ROOT
from mc2.models.model_interface import load_model


class JilesAthertonWithGRU(eqx.Module):
    ja: JilesAthertonStatic
    gru: GRU

    def __init__(self, key, in_size, hidden_size=8, **kwargs):
        ja_key, gru_key = jax.random.split(key, 2)
        self.ja = JilesAthertonStatic(ja_key, **kwargs)
        self.gru = GRU(in_size, hidden_size=hidden_size, key=gru_key)


class JilesAthertonGRU(eqx.Module):
    ja: JilesAthertonStatic = eqx.field(static=True)
    gru: GRUwLinear
    normalizer: eqx.Module = eqx.field(static=True)

    def __init__(self, normalizer, key, in_size, hidden_size=8, **kwargs):
        ja_key, gru_key = jax.random.split(key, 2)
        self.ja = load_model(MODEL_DUMP_ROOT / "b8d1fe17-2f6f-40.eqx", JilesAthertonStatic)  # using pretrained ja model

        # JilesAthertonStatic(ja_key, **kwargs)

        self.gru = GRUwLinear(in_size=in_size, out_size=1, hidden_size=hidden_size, key=gru_key)
        self.normalizer = normalizer

    def __call__(self, H0, B_seq, features_seq):
        def body_fun(carry, inputs):
            H_prev, h_gru = carry
            B_curr, B_next, feat_next = inputs

            H_next_phys, _ = self.ja.step(H_prev, B_curr, B_next)
            H_next_phys_norm = self.normalizer.normalize_H(H_next_phys)
            gru_in = jnp.concatenate([jnp.array([H_next_phys_norm]), feat_next])
            h_gru_new = self.gru.cell(gru_in, h_gru)
            delta_H_norm = self.gru.linear(h_gru_new) + self.gru.bias
            # delta_H_norm = h_gru_new[0]
            delta_H = self.normalizer.denormalize_H(jnp.squeeze(delta_H_norm))
            H_next = H_next_phys + delta_H

            return (H_next, h_gru_new), H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        inputs = (B_pairs[:, 0], B_pairs[:, 1], features_seq[:-1])  # features_seq[1:]
        h0 = jnp.zeros(self.gru.hidden_size)

        (_, _), H_seq = jax.lax.scan(body_fun, (H0, h0), inputs)
        H_seq = jnp.concatenate([jnp.array([H0]), H_seq], axis=0)
        return H_seq
