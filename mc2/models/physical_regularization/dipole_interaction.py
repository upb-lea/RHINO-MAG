import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx


def random_unit_vectors_2d(n, key):
    angles = jax.random.uniform(key, n, minval=0, maxval=2 * jnp.pi)
    vectors = jnp.column_stack((jnp.cos(angles), jnp.sin(angles)))
    return vectors


class DipoleGrid(eqx.Module):
    m: jax.Array
    pos: jax.Array
    omega: jax.Array
    n_elements: int
    n_x: int
    n_y: int
    distance: float
    # T0: jax.Array

    @property
    def dim(self):
        return self.m.shape[-1]

    @property
    def shape(self):
        return self.m.shape

    @property
    def xrange(self):
        return jnp.arange(0, self.n_x, 1) * self.distance

    @property
    def yrange(self):
        return jnp.arange(0, self.n_y, 1) * self.distance

    @property
    def theta(self):
        return jnp.arctan2(self.m[..., 1], self.m[..., 0])

    @classmethod
    def from_data(cls, m, pos, omega, distance: float = 0.1):
        n_x, n_y, _ = m.shape
        n_elements = n_x * n_y
        assert pos.shape == m.shape
        assert omega.shape == m.shape[:-1]

        return cls(
            m=m,
            n_elements=n_elements,
            n_x=n_x,
            n_y=n_y,
            pos=pos,
            omega=omega,
            distance=distance,
            # T0=T0,
        )

    @classmethod
    def from_random_key(cls, key, n_elements, distance: float = 0.1):
        n_x = int(jnp.sqrt(n_elements))
        n_y = n_x
        m = random_unit_vectors_2d(n_elements, key).reshape(n_x, n_y, 2)

        xx, yy = jnp.meshgrid(jnp.arange(0, n_x, 1), jnp.arange(0, n_y, 1), indexing="ij")
        pos = jnp.concatenate([xx[..., None], yy[..., None]], axis=-1)

        return cls(
            m=m,
            n_elements=n_elements,
            n_x=n_x,
            n_y=n_y,
            pos=pos,
            omega=jnp.zeros(pos.shape[:-1]),
            distance=distance,
            # T0=jnp.zeros(pos.shape[:-1]),  # random_unit_vectors_2d(n_elements, key).reshape(n_x, n_y, 2)[..., 0],
        )

    @staticmethod
    def get_exchange_field_single(state_single: jax.Array, pos_single: jax.Array, state: "DipoleGrid", mu0: float):
        r = pos_single - state.pos
        r_mag = jnp.linalg.norm(r, axis=-1, keepdims=True)
        r_unit = r / r_mag

        prefactor = mu0 / (4 * jnp.pi * r_mag**3)
        field = prefactor * (3 * (state.m * r_unit) * r_unit - state.m)
        return jnp.sum(field, axis=(0, 1), where=~jnp.isnan(field))

    def get_exchange_field(self, mu0: float = 1.0):
        exchange_fields = eqx.filter_vmap(
            self.get_exchange_field_single,
            in_axes=(0, 0, None, None),
        )(self.m.reshape(-1, 2), self.pos.reshape(-1, 2), self, mu0)

        return exchange_fields.reshape(self.n_x, self.n_y, 2)

    def get_torque(self, ext_field: jax.Array, mu0: float = 1.0):
        exchange_field = self.get_exchange_field()
        effective_field = exchange_field + ext_field

        torque = jnp.cross(self.m, effective_field)
        return torque

    def visualize(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        xx, yy = jnp.meshgrid(self.xrange, self.yrange, indexing="ij")

        ax.quiver(xx, yy, self.m[..., 0], self.m[..., 1], pivot="mid")
        ax.grid(True, alpha=0.3)
        return fig, ax


@eqx.filter_jit
def step_rot_ode(state, ext_field, tau, J, d):

    theta = state.theta
    omega = state.omega

    torques = state.get_torque(ext_field)

    # domega = (torques - state.T0) / J - d * omega
    domega = torques / J - d * omega
    dtheta = omega

    next_omega = omega + tau * domega
    next_theta = theta + tau * dtheta

    next_m = (
        jnp.concatenate([jnp.cos(next_theta)[..., None], jnp.sin(next_theta)[..., None]], axis=-1)
        * jnp.linalg.norm(state.m, axis=-1)[..., None]
    )

    next_state = DipoleGrid.from_data(
        m=next_m,
        omega=next_omega,
        pos=state.pos,
        # T0=state.T0,
    )
    return next_state


def simulate_rot_ode(init_state, ext_fields, tau, J, d):
    state = init_state

    states = [state]
    ms = [state.m]
    omegas = [state.omega]

    for i, ext_field in tqdm.tqdm(enumerate(ext_fields)):
        next_state = step_rot_ode(state, ext_field, tau, J, d)
        state = next_state

        states.append(state)
        ms.append(state.m)
        omegas.append(state.omega)

    return states, jnp.stack(ms), jnp.stack(omegas)
