import jax
import jax.numpy as jnp
import jax.nn as jnn
import diffrax
import equinox as eqx


class StateSpaceMLP(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, obs_dim, action_dim, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=(obs_dim + action_dim),
            out_size=obs_dim,
            width_size=width_size,
            depth=depth,
            activation=jnn.leaky_relu,
            key=key,
        )

    def __call__(self, obs, action):
        obs_action = jnp.hstack([obs, action])
        return self.mlp(obs_action)


class NeuralODE(eqx.Module):
    func: StateSpaceMLP
    _solver: diffrax.AbstractSolver

    def __init__(self, solver, obs_dim, action_dim, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = StateSpaceMLP(obs_dim, action_dim, width_size, depth, key=key)
        self._solver = solver

    def __call__(self, init_obs, actions, tau):

        args = (actions, None)

        def action_helper(t, args):
            actions = args
            return actions[jnp.array(t / tau, int)]

        def vector_field(t, y, args):
            actions, _ = args

            action = action_helper(t, actions)
            dy_dt = self.func(y, action)
            return tuple(dy_dt)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = tau * actions.shape[0]

        y0 = tuple(init_obs)
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 1 + int(t1 / tau)))
        solution = diffrax.diffeqsolve(term, self._solver, t0, t1, dt0=tau, y0=y0, args=args, saveat=saveat)

        return jnp.transpose(jnp.array(solution.ys))


class NeuralEulerODE(eqx.Module):
    func: StateSpaceMLP

    def __init__(self, obs_dim, action_dim, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = StateSpaceMLP(obs_dim, action_dim, width_size, depth, key=key)

    def step(self, obs, action, tau):
        next_obs = obs + tau * self.func(obs, action)
        return next_obs

    def __call__(self, init_obs, actions, tau):

        def body_fun(carry, action):
            obs = carry
            obs = self.step(obs, action, tau)
            return obs, obs

        _, observations = jax.lax.scan(body_fun, init_obs, actions)
        observations = jnp.concatenate([init_obs[None, :], observations], axis=0)
        return observations


class HiddenStateNeuralEulerODE(eqx.Module):
    state_func: StateSpaceMLP
    obs_func: callable
    obs_dim: int
    state_dim: int
    action_dim: int

    def __init__(self, obs_dim, state_dim, action_dim, width_size, depth, obs_func_type, *, key, **kwargs):
        super().__init__(**kwargs)
        self.state_func = StateSpaceMLP(state_dim, action_dim, width_size, depth, key=key)

        if obs_func_type == "identity":
            self.obs_func = lambda x: x[0]
        else:
            raise NotImplementedError()

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

    def step(self, state, action, tau):
        next_state = state + tau * self.state_func(state, action)
        next_obs = self.obs_func(state)

        return next_state, next_obs

    def __call__(self, init_obs, actions, tau):

        init_state = init_obs.repeat(self.state_dim, axis=0)

        def body_fun(carry, action):
            state = carry
            state, obs = self.step(state, action, tau)
            return state, jnp.hstack([state, obs])

        _, states_observations = jax.lax.scan(body_fun, init_state, actions)

        init_obs = self.obs_func(init_state)
        states_observations = jnp.concatenate(
            [jnp.hstack([init_state, init_obs])[None, :], states_observations], axis=0
        )

        states = states_observations[:, : self.state_dim]
        observations = states_observations[:, self.state_dim :]
        return states, observations
