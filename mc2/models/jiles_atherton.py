import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from mc2.data_management import MODEL_DUMP_ROOT, Normalizer
from mc2.models.RNN import GRU, GRUwLinear
from mc2.model_interfaces.model_interface import load_model

# Adjoint method for estimating Jiles-Atherton hysteresis
# model parameters


# On physical aspects of the Jiles-Atherton hysteresis models
def softclip(x, limit=1e8):
    return limit * jnp.tanh(x / limit)


def softplus(x, eps=1e-6):
    return jnp.log1p(jnp.exp(x)) + eps


class JAStatic(eqx.Module):
    """
    Classical static Jiles-Atherton hysteresis model.

    Parameters:
        Ms_param, a_param, alpha_param, k_param, c_param : Trainable model parameters
        mu_0 : Vacuum permeability
        tau : Time constant for numerical integration
    """

    Ms_param: jax.Array
    a_param: jax.Array
    alpha_param: jax.Array
    k_param: jax.Array
    c_param: jax.Array
    mu_0: float = 4e-7 * jnp.pi
    tau: float = 1 / 16e6

    @property
    def physical_params(self):
        return dict(
            Ms=self.Ms,
            a=self.a,
            alpha=self.alpha,
            k=self.k,
            c=self.c,
        )

    @property
    def params(self):
        return dict(
            Ms_param=self.Ms_param,
            a_param=self.a_param,
            alpha_param=self.alpha_param,
            k_param=self.k_param,
            c_param=self.c_param,
        )

    @property
    def a(self):
        return 100 * jax.nn.sigmoid(self.a_param)

    @property
    def alpha(self):
        return 1e-4 * jax.nn.sigmoid(self.alpha_param)

    @property
    def k(self):
        return 100 * jax.nn.sigmoid(self.k_param)

    @property
    def c(self):
        return jax.nn.sigmoid(self.c_param)

    @property
    def Ms(self):
        return 2e6 * jax.nn.sigmoid(self.Ms_param)

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
            B_prev, B_next = B_pair
            H_next, _ = self.step(H_prev, B_prev, B_next)
            return H_next, H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        _, H_seq = jax.lax.scan(body_fun, H0, B_pairs)
        # H_seq = jnp.concatenate([jnp.array([H0]), H_seq], axis=0)
        return H_seq


class JAStatic2(eqx.Module):
    """
    Extended dynamic Jiles-Atherton variant using Bertotti's Dynamic Hysteresis Model.

    Note: did not really improve performance
    """

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
            B_prev, B_next = B_pair
            H_next, _ = self.step(H_prev, B_prev, B_next)
            return H_next, H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        _, H_seq = jax.lax.scan(body_fun, H0, B_pairs)
        # H_seq = jnp.concatenate([jnp.array([H0]), H_seq], axis=0)
        return H_seq


class JAStatic3(eqx.Module):
    """
    Jiles-Atherton hysteresis model with dynamic extensions from
    "Hybrid magnetic field formulation based on the losses separation method for modified dynamic inverse Jiles-Atherton model".

    Parameters:
        Ms_param, a_param, alpha_param, k_param, c_param : Trainable model parameters
        mu_0 : Vacuum permeability
        tau : Time constant for numerical integration
    """

    Ms_param: jax.Array
    a_param: jax.Array
    alpha_param: jax.Array
    k_param: jax.Array
    c_param: jax.Array
    C_edd_param: jax.Array
    C_exc_param: jax.Array
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
    def C_edd(self):
        return jax.nn.sigmoid(self.C_edd_param)

    @property
    def C_exc(self):
        return jax.nn.sigmoid(self.C_exc_param)

    def __init__(self, key, **kwargs):
        super().__init__(**kwargs)
        k_key, alpha_key, c_key, Ms_key, a_key, C_edd_key, C_exc_key = jax.random.split(key, 7)
        self.k_param = jax.random.uniform(k_key, ()) * 0.05 + 0.5
        self.c_param = jax.random.uniform(c_key, ()) * 0.05 + 0.5
        self.a_param = jax.random.uniform(a_key, ()) * 0.05 + 0.5
        self.Ms_param = jax.random.uniform(Ms_key, ()) * 0.05 + 0.5
        self.alpha_param = jax.random.uniform(alpha_key, ()) * 0.001 + 0.0002
        self.C_edd_param = jax.random.uniform(C_edd_key, ()) * 0.001 + 0.001
        self.C_exc_param = jax.random.uniform(C_exc_key, ()) * 0.001 + 0.001

    def coth(self, x):
        return 1 / jnp.tanh(x)

    def coth_stable(self, x):
        eps = 1e-7
        x = jnp.where(jnp.abs(x) < eps, eps * jnp.sign(x), x)
        return 1 / jnp.tanh(x)

    # Updated dM_dH function
    def dM_dH(self, H, M, dB_dt):
        H_edd = self.C_edd * dB_dt
        H_exc = self.C_exc * dB_dt * 1 / (jnp.sqrt(jnp.abs(dB_dt)))

        H_e = H + self.alpha * M - H_edd - H_exc
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
            B_prev, B_next = B_pair
            H_next, _ = self.step(H_prev, B_prev, B_next)
            return H_next, H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        _, H_seq = jax.lax.scan(body_fun, H0, B_pairs)
        # H_seq = jnp.concatenate([jnp.array([H0]), H_seq], axis=0)
        return H_seq


class JAEnsemble(eqx.Module):

    vector_ja: JAStatic

    def __init__(self, key, n_models):
        ja_keys = jax.random.split(key, n_models)
        self.vector_ja = eqx.filter_vmap(JAStatic)(ja_keys)

    def __call__(self, H0, B_seq):

        def call_model(model, H0, B_seq):
            return model(H0, B_seq)

        batched_H_seq = eqx.filter_vmap(call_model, in_axes=(0, None, None))(self.vector_ja, H0, B_seq)
        H_seq = jnp.mean(batched_H_seq, axis=0)
        return H_seq


class JAWithExternGRU(eqx.Module):
    """
    Static Jiles-Atherton model with an external GRU correction.

    Workflow:
        1. Compute the H-field trajectory using the static JA model.
        2. Apply a GRU across the full trajectory to correct residual errors.
    """

    ja: JAStatic #= eqx.field(static=True)
    gru: GRU

    def __init__(self, key, in_size, hidden_size=8, **kwargs):
        ja_key, gru_key = jax.random.split(key, 2)
        self.ja = JAStatic(ja_key, **kwargs)
        #self.ja = load_model(MODEL_DUMP_ROOT / "4ec8f810-298b-49.eqx", JAStatic)
        self.gru = GRU(in_size, hidden_size=hidden_size, key=gru_key)


class JAWithGRUlin(eqx.Module):
    """
    Static Jiles-Atherton model with GRU correction.

    Correction:
        - GRU with linear output layer models the discrepancy between the physical JA model and measured data.
        - Normalized H-field values are used as input to the GRU.
    """

    ja: JAStatic = eqx.field(static=True)
    gru: GRUwLinear
    normalizer: eqx.Module = eqx.field(static=True)

    def __init__(self, normalizer, key, in_size, hidden_size=8, **kwargs):
        ja_key, gru_key = jax.random.split(key, 2)
        self.ja = load_model(MODEL_DUMP_ROOT / "4ec8f810-298b-49.eqx", JAStatic)  # using pretrained ja model
        # self.ja = JAStatic(ja_key, **kwargs)

        self.gru = GRUwLinear(in_size=in_size, out_size=1, hidden_size=hidden_size, key=gru_key)
        self.normalizer = normalizer

    def __call__(self, H0, B_seq, features_seq, init_hidden):
        def body_fun(carry, inputs):
            H_prev, h_gru = carry
            B_prev, B_next, feat_next = inputs

            H_next_phys, _ = self.ja.step(H_prev, B_prev, B_next)
            H_next_phys_norm = self.normalizer.normalize_H(H_next_phys)
            gru_in = jnp.concatenate([jnp.array([H_next_phys_norm]), feat_next])
            h_gru_new = self.gru.cell(gru_in, h_gru)
            delta_H_norm = self.gru.linear(h_gru_new) + self.gru.bias
            delta_H = self.normalizer.denormalize_H(jnp.squeeze(delta_H_norm))
            H_next = H_next_phys + delta_H

            return (H_next, h_gru_new), H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        inputs = (B_pairs[:, 0], B_pairs[:, 1], features_seq[:])  # features_seq[:-1] # 1:]

        (_, _), H_seq = jax.lax.scan(body_fun, (H0, init_hidden), inputs)
        # H_seq = jnp.concatenate([jnp.array([H0]), H_seq], axis=0)
        return H_seq

    def warmup_call(self, H0, B_seq, features_seq, init_hidden, H_true):
        def body_fun(carry, inputs):
            H_prev, h_gru = carry
            B_curr, B_next, feat_next, h_true = inputs

            H_next_phys, _ = self.ja.step(H_prev, B_curr, B_next)
            H_next_phys_norm = self.normalizer.normalize_H(H_next_phys)
            gru_in = jnp.concatenate([jnp.array([H_next_phys_norm]), feat_next])
            h_gru_new = self.gru.cell(gru_in, h_gru)
            delta_H_norm = self.gru.linear(h_gru_new) + self.gru.bias
            delta_H = h_true - H_next_phys  # self.normalizer.denormalize_H(jnp.squeeze(delta_H_norm))
            H_next = H_next_phys + delta_H

            return (H_next, h_gru_new), H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        inputs = (B_pairs[:, 0], B_pairs[:, 1], features_seq[:], H_true)  # features_seq[:-1] # 1:]

        (_, final_hidden), H_seq = jax.lax.scan(body_fun, (H0, init_hidden), inputs)
        # H_seq = jnp.concatenate([jnp.array([H0]), H_seq], axis=0)
        return H_seq, final_hidden


class JAWithGRU(eqx.Module):
    """
    Static Jiles-Atherton model with GRU correction.

    Correction:
        - GRU models the discrepancy between the physical JA model and measured data.
        - Normalized H-field values are used as input to the GRU.
    """

    ja: JAStatic #= eqx.field(static=True)
    gru: GRU
    normalizer: eqx.Module = eqx.field(static=True)

    def __init__(self, normalizer, key, in_size, hidden_size=8, **kwargs):
        ja_key, gru_key = jax.random.split(key, 2)
        # self.ja = load_model(
        #     MODEL_DUMP_ROOT / "4ec8f810-298b-49.eqx", JAStatic
        # )  # using pretrained ja model # b8d1fe17-2f6f-40,
        self.ja = JAStatic(ja_key, **kwargs)
        self.gru = GRU(in_size=in_size, hidden_size=hidden_size, key=gru_key)
        self.normalizer = normalizer

    def __call__(self, H0, B_seq, features_seq, init_hidden):
        def body_fun(carry, inputs):
            H_prev, h_gru = carry
            B_prev, B_next, feat_next = inputs

            H_next_phys, _ = self.ja.step(H_prev, B_prev, B_next)
            H_next_phys_norm = self.normalizer.normalize_H(H_next_phys)
            gru_in = jnp.concatenate([jnp.array([H_next_phys_norm]), feat_next])
            h_gru_new = self.gru.cell(gru_in, h_gru)
            delta_H_norm = h_gru_new[0]
            delta_H = self.normalizer.denormalize_H(jnp.squeeze(delta_H_norm))
            H_next = H_next_phys + delta_H

            return (H_next, h_gru_new), H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        inputs = (B_pairs[:, 0], B_pairs[:, 1], features_seq[:])  # features_seq[:-1] # 1:]

        (_, _), H_seq = jax.lax.scan(body_fun, (H0, init_hidden), inputs)
        # H_seq = jnp.concatenate([jnp.array([H0]), H_seq], axis=0)
        return H_seq

    def warmup_call(self, H0, B_seq, features_seq, init_hidden, H_true):
        def body_fun(carry, inputs):
            H_prev, h_gru = carry
            B_curr, B_next, feat_next, h_true = inputs

            H_next_phys, _ = self.ja.step(H_prev, B_curr, B_next)
            H_next_phys_norm = self.normalizer.normalize_H(H_next_phys)
            gru_in = jnp.concatenate([jnp.array([H_next_phys_norm]), feat_next])
            h_gru_new = self.gru.cell(gru_in, h_gru)
            h_gru_new = h_gru_new.at[0].set(self.normalizer.normalize_H(h_true - H_next_phys))
            delta_H_norm = h_gru_new[0]
            delta_H = self.normalizer.denormalize_H(jnp.squeeze(delta_H_norm))
            H_next = H_next_phys + delta_H

            return (H_next, h_gru_new), H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        inputs = (B_pairs[:, 0], B_pairs[:, 1], features_seq[:], H_true)  # features_seq[:-1] # 1:]

        (_, final_hidden), H_seq = jax.lax.scan(body_fun, (H0, init_hidden), inputs)
        # H_seq = jnp.concatenate([jnp.array([H0]), H_seq], axis=0)
        # final_hidden = final_hidden.at[0].set(0) -> maybe ?
        return H_seq, final_hidden

class GRUWithJA(eqx.Module):
    """
    Static Jiles-Atherton model with GRU correction.

    Correction:
        - GRU models the discrepancy between the physical JA model and measured data.
        - Normalized H-field values are used as input to the GRU.
    """
    ja: JAStatic = eqx.field(static=True)
    gru: GRU
    normalizer: eqx.Module = eqx.field(static=True)

    def __init__(self, normalizer, key, in_size, hidden_size=8, **kwargs):
        ja_key, gru_key = jax.random.split(key, 2)
        self.ja = load_model(
            MODEL_DUMP_ROOT / "4ec8f810-298b-49.eqx", JAStatic
        )  # using pretrained ja model # b8d1fe17-2f6f-40,
        #self.ja = JAStatic(ja_key, **kwargs)
        self.gru = GRU(in_size=in_size, hidden_size=hidden_size, key=gru_key)
        self.normalizer = normalizer

    def __call__(self, H0, B_seq, features_seq, init_hidden):
        def body_fun(carry, inputs):
            H_prev, h_gru = carry
            B_prev, B_next, feat_next = inputs

            H_next_phys, _ = self.ja.step(H_prev, B_prev, B_next)
            H_next_phys_norm = self.normalizer.normalize_H(H_next_phys)
            gru_in = jnp.concatenate([jnp.array([H_next_phys_norm]), feat_next])
            h_gru_new = self.gru.cell(gru_in, h_gru)
            H_next_norm = h_gru_new[0]
            H_next = self.normalizer.denormalize_H(jnp.squeeze(H_next_norm))

            return (H_next, h_gru_new), H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        inputs = (B_pairs[:, 0], B_pairs[:, 1], features_seq[:])  # features_seq[:-1] # 1:]

        (_, _), H_seq = jax.lax.scan(body_fun, (H0, init_hidden), inputs)
        # H_seq = jnp.concatenate([jnp.array([H0]), H_seq], axis=0)
        return H_seq

    def warmup_call(self, H0, B_seq, features_seq, init_hidden, H_true):
        def body_fun(carry, inputs):
            H_prev, h_gru = carry
            B_curr, B_next, feat_next, h_true = inputs

            H_next_phys, _ = self.ja.step(H_prev, B_curr, B_next)
            H_next_phys_norm = self.normalizer.normalize_H(H_next_phys)
            gru_in = jnp.concatenate([jnp.array([H_next_phys_norm]), feat_next])
            h_gru_new = self.gru.cell(gru_in, h_gru)
            h_gru_new = h_gru_new.at[0].set(self.normalizer.normalize_H(h_true))
            H_next_norm = h_gru_new[0]
            H_next = self.normalizer.denormalize_H(jnp.squeeze(H_next_norm))

            return (H_next, h_gru_new), H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        inputs = (B_pairs[:, 0], B_pairs[:, 1], features_seq[:], H_true)  # features_seq[:-1] # 1:]

        (_, final_hidden), H_seq = jax.lax.scan(body_fun, (H0, init_hidden), inputs)
        # H_seq = jnp.concatenate([jnp.array([H0]), H_seq], axis=0)
        #final_hidden = final_hidden.at[0].set(0) -> maybe ?
        return H_seq, final_hidden




class LFRWithGRUJA(eqx.Module):
    """
    https://arxiv.org/pdf/2404.01901

    """
    ja: JAStatic = eqx.field(static=True)
    gru: GRU
    intercon_matrix : jax.Array
    x_k_size: int
    normalizer: eqx.Module = eqx.field(static=True)

    def __init__(self, normalizer, key, in_size, hidden_size=8,x_k_size=5, **kwargs):
        ja_key, gru_key = jax.random.split(key, 2)
        self.ja = load_model(
            MODEL_DUMP_ROOT / "4ec8f810-298b-49.eqx", JAStatic
        )  
        self.gru = GRU(in_size=in_size, hidden_size=hidden_size, key=gru_key)
        self.normalizer = normalizer
        n_features=len(normalizer.norm_fe_max) +1 # +1 for temperature
        self.x_k_size=x_k_size
                                                    
        intercon_matrix_init = jax.random.uniform(key, (x_k_size + 1 + 3 + in_size, x_k_size+ (3+n_features) + 1 + 1)) * 0.1 # (x_k_size + y_size + ja_in_size + gru_in_size, x_k_size + u_k_size + output_ja_size + output_gru_size)

        #H_prev -> z_ja_Hprev 
        intercon_matrix_init = intercon_matrix_init.at[x_k_size + 1, x_k_size + 0].set(1.0)
        #B_prev -> z_ja_Bprev 
        intercon_matrix_init = intercon_matrix_init.at[x_k_size + 2, x_k_size + 1].set(1.0)
        #B_next -> z_ja_Bnext 
        intercon_matrix_init = intercon_matrix_init.at[x_k_size + 3, x_k_size + 2].set(1.0)

        #GRU inputs
        for i in range(n_features):
            intercon_matrix_init = intercon_matrix_init.at[x_k_size + 4 + i, x_k_size + 3 + i].set(1.0)
        self.intercon_matrix = intercon_matrix_init

    def __call__(self, H0, B_seq, features_seq, init_hidden):
        x_k0= jnp.zeros(self.x_k_size)
        def body_fun(carry, inputs):
            H_prev, h_gru, x_k = carry
            B_prev, B_next, feat_next = inputs
            H_prev_vec = jnp.atleast_1d(H_prev)
            B_prev_vec = jnp.atleast_1d(B_prev)
            B_next_vec = jnp.atleast_1d(B_next)
            u=jnp.concatenate([H_prev_vec, B_prev_vec, B_next_vec, feat_next])
            in_pre=jnp.concatenate([x_k,u,jnp.zeros(1),jnp.zeros(1)])# loop otherwise
            out_pre = jnp.matmul(self.intercon_matrix, in_pre)
            z_ja_Hprev = self.normalizer.denormalize_H(out_pre[self.x_k_size+1])
            z_ja_Bprev = out_pre[self.x_k_size+2] * (self.normalizer.B_max)
            z_ja_Bnext = out_pre[self.x_k_size+3] * (self.normalizer.B_max)
            z_gru = out_pre[self.x_k_size+4:]

            H_next_phys, _ = self.ja.step(z_ja_Hprev, z_ja_Bprev, z_ja_Bnext)
            w_ja = self.normalizer.normalize_H(H_next_phys)
            h_gru_new = self.gru.cell(z_gru, h_gru)
            w_gru = h_gru_new[0]

            w_gru=jnp.atleast_1d(w_gru)
            w_ja=jnp.atleast_1d(w_ja)
            in_=jnp.concatenate([x_k,u,w_ja,w_gru])
            out = jnp.matmul(self.intercon_matrix, in_)
            x_k1 = out[:self.x_k_size]
            H_next= out[self.x_k_size]

            return (H_next, h_gru_new, x_k1), H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        inputs = (B_pairs[:, 0], B_pairs[:, 1], features_seq[:])  # features_seq[:-1] # 1:]

        (_, _,_), H_seq = jax.lax.scan(body_fun, (H0, init_hidden, x_k0), inputs)
        # H_seq = jnp.concatenate([jnp.array([H0]), H_seq], axis=0)
        return H_seq

    # def warmup_call(self, H0, B_seq, features_seq, init_hidden, H_true):
    #     def body_fun(carry, inputs):
    #         H_prev, h_gru = carry
    #         B_curr, B_next, feat_next, h_true = inputs

    #         H_next_phys, _ = self.ja.step(H_prev, B_curr, B_next)
    #         H_next_phys_norm = self.normalizer.normalize_H(H_next_phys)
    #         gru_in = jnp.concatenate([jnp.array([H_next_phys_norm]), feat_next])
    #         h_gru_new = self.gru.cell(gru_in, h_gru)
    #         h_gru_new = h_gru_new.at[0].set(self.normalizer.normalize_H(h_true))
    #         H_next_norm = h_gru_new[0]
    #         H_next = self.normalizer.denormalize_H(jnp.squeeze(H_next_norm))

    #         return (H_next, h_gru_new), H_next

    #     B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
    #     inputs = (B_pairs[:, 0], B_pairs[:, 1], features_seq[:], H_true)  # features_seq[:-1] # 1:]

    #     (_, final_hidden), H_seq = jax.lax.scan(body_fun, (H0, init_hidden), inputs)
    #     # H_seq = jnp.concatenate([jnp.array([H0]), H_seq], axis=0)
    #     #final_hidden = final_hidden.at[0].set(0) -> maybe ?
    #     return H_seq, final_hidden


class JADynamic(JAStatic):
    param_scale: jax.Array

    def __init__(self, base_model: JAStatic, param_scale):
        self.Ms_param = base_model.Ms_param
        self.a_param = base_model.a_param
        self.alpha_param = base_model.alpha_param
        self.k_param = base_model.k_param
        self.c_param = base_model.c_param
        self.mu_0 = base_model.mu_0
        self.tau = base_model.tau
        self.param_scale = param_scale

    @property
    def Ms(self):
        return 6e6 * jax.nn.sigmoid(self.Ms_param) * self.param_scale[0]

    @property
    def a(self):
        return 2000 * jax.nn.sigmoid(self.a_param) * self.param_scale[1]

    @property
    def alpha(self):
        return 1e-2 * jax.nn.sigmoid(self.alpha_param) * self.param_scale[2]

    @property
    def k(self):
        return 1000.0 * jax.nn.sigmoid(self.k_param) * self.param_scale[3]

    @property
    def c(self):
        return jax.nn.sigmoid(self.c_param) * self.param_scale[4]


class JAParamGRUlin(eqx.Module):
    """
    Dynamic scaling of Jiles-Atherton parameters via GRU + linear layer.

    Each timestep:
        1. GRU + linear layer predicts parameter scaling factors.
        2. The scaled JA model is applied to compute the next H-field.
    """

    ja: JAStatic = eqx.field(static=True)
    gru: GRUwLinear
    normalizer: eqx.Module = eqx.field(static=True)

    def __init__(self, normalizer, key, in_size, hidden_size=8):
        ja, gru_key = jax.random.split(key, 2)
        self.ja = load_model(MODEL_DUMP_ROOT / "4ec8f810-298b-49.eqx", JAStatic)
        self.gru = GRUwLinear(in_size=in_size, out_size=5, hidden_size=hidden_size, key=gru_key)
        self.normalizer = normalizer

    def __call__(self, H0, B_seq, features_seq, init_hidden):
        def body_fun(carry, inputs):
            H_prev, h_gru = carry
            B_prev, B_next, feat_next = inputs
            H_prev_norm = self.normalizer.normalize_H(H_prev)
            gru_in = jnp.concatenate([jnp.array([H_prev_norm]), feat_next])
            h_gru_new = self.gru.cell(gru_in, h_gru)
            raw_scales = self.gru.linear(h_gru_new) + self.gru.bias
            param_scale = 1.0 + jnp.tanh(raw_scales)

            ja_dyn = JADynamic(self.ja, param_scale)

            H_next, _ = ja_dyn.step(H_prev, B_prev, B_next)

            return (H_next, h_gru_new), H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        inputs = (B_pairs[:, 0], B_pairs[:, 1], features_seq[:])
        (_, _), H_seq = jax.lax.scan(body_fun, (H0, init_hidden), inputs)
        return H_seq

    def warmup_call(self, H0, B_seq, features_seq, init_hidden, H_true):

        def body_fun(carry, inputs):
            H_prev, h_gru = carry
            B_prev, B_next, feat_next, h_true = inputs

            gru_in = jnp.concatenate([jnp.array([H_prev]), feat_next])
            h_gru_new = self.gru.cell(gru_in, h_gru)
            return (h_true, h_gru_new), h_true

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        inputs = (B_pairs[:, 0], B_pairs[:, 1], features_seq[:], H_true)
        (_, final_hidden), H_seq = jax.lax.scan(body_fun, (H0, init_hidden), inputs)
        # does not work well therefore just using default init_hidden
        final_hidden = init_hidden
        return H_seq, final_hidden


class JAParamMLP(eqx.Module):
    """
    Dynamic scaling of Jiles-Atherton parameters via MLP.

    Each timestep:
        1. MLP predicts parameter scaling factors from H-field and features.
        2. The scaled JA model is applied to compute the next H-field.
    """

    ja: JAStatic = eqx.field(static=True)
    mlp: eqx.nn.MLP
    normalizer: eqx.Module = eqx.field(static=True)

    def __init__(self, normalizer, key, in_size, hidden_size=32, depth=2):
        ja_key, mlp_key = jax.random.split(key, 2)
        self.ja = load_model(MODEL_DUMP_ROOT / "4ec8f810-298b-49.eqx", JAStatic)
        # self.ja = JAStatic(ja_key)
        self.normalizer = normalizer
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=5,
            width_size=hidden_size,
            depth=depth,
            activation=jnp.tanh,  # jax.nn.leaky_relu
            key=mlp_key,
        )

    def __call__(self, H0, B_seq, features_seq):
        def body_fun(carry, inputs):
            H_prev = carry
            B_prev, B_next, feat_next = inputs
            H_prev_norm = self.normalizer.normalize_H(H_prev)
            mlp_in = jnp.concatenate([jnp.array([H_prev_norm]), feat_next])

            raw_scales = self.mlp(mlp_in)
            param_scale = 1.0 + jnp.tanh(raw_scales)

            ja_dyn = JADynamic(self.ja, param_scale)

            H_next, _ = ja_dyn.step(H_prev, B_prev, B_next)

            return (H_next), H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        inputs = (B_pairs[:, 0], B_pairs[:, 1], features_seq[:])
        (_), H_seq = jax.lax.scan(body_fun, (H0), inputs)
        return H_seq


def set_parameters(model, params):

    Ms_param, a_param, alpha_param, k_param, c_param = params

    model = eqx.tree_at(lambda m: m.Ms_param, model, jnp.array((Ms_param).astype(float)))
    model = eqx.tree_at(lambda m: m.a_param, model, jnp.array((a_param).astype(float)))
    model = eqx.tree_at(lambda m: m.alpha_param, model, jnp.array((alpha_param).astype(float)))
    model = eqx.tree_at(lambda m: m.k_param, model, jnp.array((k_param).astype(float)))
    model = eqx.tree_at(lambda m: m.c_param, model, jnp.array((c_param).astype(float)))
    return model


class JADirectParamGRU(eqx.Module):
    ja: JAStatic
    gru: eqx.nn.GRUCell
    normalizer: Normalizer = eqx.field(static=True)
    n_ja_params: int

    def __init__(self, normalizer, key, in_size, hidden_size):
        ja_key, gru_key = jax.random.split(key, 2)
        self.ja = JAStatic(ja_key)
        self.n_ja_params = len(self.ja.params)
        self.normalizer = normalizer

        self.gru = eqx.nn.GRUCell(
            input_size=in_size,
            hidden_size=hidden_size,
            key=gru_key,
        )

    def __call__(self, H0, B_seq, features_seq, init_hidden):
        hidden = init_hidden

        def body_fun(carry, inputs):
            H_prev, hidden = carry
            B_prev, B_next, feat_next = inputs
            H_prev_norm = self.normalizer.normalize_H(H_prev)
            B_next_norm = B_next / self.normalizer.B_max

            inp = jnp.concatenate([H_prev_norm[None], feat_next], axis=-1)
            new_hidden = self.gru(inp, hidden)
            ja_params = new_hidden[: self.n_ja_params]
            ja_model = set_parameters(self.ja, ja_params)

            H_next, _ = ja_model.step(H_prev, B_prev, B_next)

            return (H_next, new_hidden), H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        inputs = (B_pairs[:, 0], B_pairs[:, 1], features_seq[:])
        (_), H_seq = jax.lax.scan(body_fun, (H0, hidden), inputs)
        return H_seq
    
    def warmup_call(self, H0, B_seq, features_seq, init_hidden, H_true):
        def body_fun(carry, inputs):
            H_prev, hidden = carry
            B_prev, B_next, feat_next, h_true = inputs
            H_prev_norm = self.normalizer.normalize_H(H_prev)

            inp = jnp.concatenate([H_prev_norm[None], feat_next], axis=-1)
            new_hidden = self.gru(inp, hidden)
            H_next= h_true

            return (H_next, new_hidden), H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        inputs = (B_pairs[:, 0], B_pairs[:, 1], features_seq[:], H_true)  # features_seq[:-1] # 1:]

        (_, final_hidden), H_seq = jax.lax.scan(body_fun, (H0, init_hidden), inputs)
        return H_seq, final_hidden


from functools import partial


class JAWithGRUlinFinal(eqx.Module):
    """
    Final training of pretrained JAWithGRUlin. JA and GRU are trained at the time.

    Correction:
        - GRU with linear output layer models the discrepancy between the physical JA model and measured data.
        - Normalized H-field values are used as input to the GRU.
    """

    ja: JAStatic  # = eqx.field(static=True)
    gru: GRUwLinear
    normalizer: eqx.Module  # = eqx.field(static=True)

    def __init__(self, normalizer, key, in_size, hidden_size=8, **kwargs):
        ja_key, gru_key = jax.random.split(key, 2)
        JAWithGRUlin_part = partial(JAWithGRUlin, normalizer=normalizer)
        model = load_model(MODEL_DUMP_ROOT / "50ef802d-7ccc-4c.eqx", JAWithGRUlin_part)
        # self.ja = load_model(MODEL_DUMP_ROOT / "4ec8f810-298b-49.eqx", JAStatic)  # using pretrained ja model
        # self.ja = JAStatic(ja_key, **kwargs)
        self.ja = model.ja
        # self.gru = GRUwLinear(in_size=in_size, out_size=1, hidden_size=hidden_size, key=gru_key)
        self.gru = model.gru
        self.normalizer = normalizer

    def __call__(self, H0, B_seq, features_seq, init_hidden):
        def body_fun(carry, inputs):
            H_prev, h_gru = carry
            B_prev, B_next, feat_next = inputs

            H_next_phys, _ = self.ja.step(H_prev, B_prev, B_next)
            H_next_phys_norm = self.normalizer.normalize_H(H_next_phys)
            gru_in = jnp.concatenate([jnp.array([H_next_phys_norm]), feat_next])
            h_gru_new = self.gru.cell(gru_in, h_gru)
            delta_H_norm = self.gru.linear(h_gru_new) + self.gru.bias
            delta_H = self.normalizer.denormalize_H(jnp.squeeze(delta_H_norm))
            H_next = H_next_phys + delta_H

            return (H_next, h_gru_new), H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        inputs = (B_pairs[:, 0], B_pairs[:, 1], features_seq[:])  # features_seq[:-1] # 1:]

        (_, _), H_seq = jax.lax.scan(body_fun, (H0, init_hidden), inputs)
        # H_seq = jnp.concatenate([jnp.array([H0]), H_seq], axis=0)
        return H_seq

    def warmup_call(self, H0, B_seq, features_seq, init_hidden, H_true):
        def body_fun(carry, inputs):
            H_prev, h_gru = carry
            B_curr, B_next, feat_next, h_true = inputs

            H_next_phys, _ = self.ja.step(H_prev, B_curr, B_next)
            H_next_phys_norm = self.normalizer.normalize_H(H_next_phys)
            gru_in = jnp.concatenate([jnp.array([H_next_phys_norm]), feat_next])
            h_gru_new = self.gru.cell(gru_in, h_gru)
            delta_H_norm = self.gru.linear(h_gru_new) + self.gru.bias
            delta_H = h_true - H_next_phys  # self.normalizer.denormalize_H(jnp.squeeze(delta_H_norm))
            H_next = H_next_phys + delta_H

            return (H_next, h_gru_new), H_next

        B_pairs = jnp.stack([B_seq[:-1], B_seq[1:]], axis=1)
        inputs = (B_pairs[:, 0], B_pairs[:, 1], features_seq[:], H_true)  # features_seq[:-1] # 1:]

        (_, final_hidden), H_seq = jax.lax.scan(body_fun, (H0, init_hidden), inputs)
        # H_seq = jnp.concatenate([jnp.array([H0]), H_seq], axis=0)
        return H_seq, final_hidden
