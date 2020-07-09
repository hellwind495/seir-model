import numpy as np

from dataclasses import dataclass

from seir.cli import MetaCLI, LockdownCLI, OdeParamCLI
from seir.parameters import MetaParams, LockdownParams, OdeParams


class BaseODE:

    def __call__(self, y, t):
        raise NotImplementedError


@dataclass
class CovidSeirODE(BaseODE):

    meta_params: MetaParams
    lockdown_params: LockdownParams
    ode_params: OdeParams

    def __post_init__(self):
        assert self.lockdown_params.nb_samples == self.ode_params.nb_samples
        assert self.lockdown_params.nb_samples == self.meta_params.nb_samples
        self._assert_param_age_group()
        self._assert_valid_params()

    def _assert_param_age_group(self):
        self.lockdown_params._assert_shapes(self.meta_params.nb_groups)
        self.ode_params._assert_shapes(self.meta_params.nb_groups)

    def _assert_valid_params(self):
        assert self.meta_params.nb_samples > 0
        assert self.meta_params.nb_groups > 0
        for x in self.lockdown_params.rel_beta_lockdown:
            assert np.all(x >= 0), \
                f"Values in 'rel_beta_lockdown' in given lockdown_params are smaller than zero"
        for x in self.lockdown_params.rel_beta_period:
            assert np.all(x > 0), \
                f"Values in 'rel_beta_period' in given lockdown_params are smaller than zero."
        assert np.all(self.ode_params.r0 > 0), "Values in 'r0' in given ode_params are less than 0"
        assert np.all(self.ode_params.beta > 0), "Values in 'beta' in given ode_params are less than 0"
        assert np.all(self.ode_params.rel_beta_asymptomatic > 0), \
            "Values in 'rel_beta_asymptomatic' in ode_params are smaller than zero"
        ode_vars = vars(self.ode_params)
        for k in ode_vars:
            if 'prop' in k:
                assert np.all(ode_vars[k] >= 0), \
                    f"Proportion parameter '{k}' in given ode_params has values smaller than 0"
                assert np.all(ode_vars[k] <= 1), \
                    f"Proportion parameter '{k}' in given ode_params has values larger than 1"
            if 'time' in k:
                assert np.all(ode_vars[k] > 0), \
                    f"Given transition time '{k}' in ode_params has values less than or equal to 0."


    def rel_beta_t_func(self, t):
        if t < 0 or t > self.lockdown_params.cum_periods[-1]:
            return 1
        else:
            return self.lockdown_params.rel_beta_lockdown[np.argmin(self.lockdown_params.cum_periods < t)]

    def __call__(self, y, t):
        # shape should be y[states, groups, samples]
        s = y[0]
        e = y[1]
        i_a = y[2]
        i_m = y[3]
        i_s = y[4]
        r_sh = y[5]
        r_sc = y[6]
        h_r = y[7]
        h_c = y[8]
        h_d = y[9]
        c_r = y[10]
        c_d = y[11]
        # r_a = y[12]
        # r_m = y[13]
        # r_h = y[14]
        # r_c = y[15]
        # d_h = y[16]
        # d_c = y[17]

        infectious_strength = np.sum(self.ode_params.rel_beta_asymptomatic * i_a + i_m + i_s, axis=0, keepdims=True)

        if self.ode_params.contact_k > 0:
            ds_coeff = self.ode_params.contact_k * np.log1p(
                self.rel_beta_t_func(t) * self.ode_params.beta * infectious_strength
                / (self.ode_params.contact_k * self.population)
            )
        else:
            ds_coeff = self.rel_beta_t_func(t) * self.ode_params.beta * infectious_strength / self.population

        ds = - ds_coeff * s
        de = ds_coeff * s - e / self.ode_params.time_incubate
        di_a = self.ode_params.prop_a * e / self.ode_params.time_incubate - i_a / self.ode_params.time_infectious
        di_m = self.ode_params.prop_m * e / self.ode_params.time_incubate - i_m / self.ode_params.time_infectious
        di_s = self.ode_params.prop_s * e / self.ode_params.time_incubate - i_s / self.ode_params.time_infectious
        dr_sh = self.ode_params.prop_s_to_h * i_s / self.ode_params.time_infectious \
            - r_sh / self.ode_params.time_rsh_to_h
        dr_sc = self.ode_params.prop_s_to_c * i_s / self.ode_params.time_infectious \
            - r_sc / self.ode_params.time_rsc_to_c
        dh_r = self.ode_params.prop_h_to_r * r_sh / self.ode_params.time_rsh_to_h - h_r / self.ode_params.time_h_to_r
        dh_c = self.ode_params.prop_h_to_c * r_sh / self.ode_params.time_rsh_to_h - h_c / self.ode_params.time_h_to_c
        dh_d = self.ode_params.prop_h_to_d * r_sh / self.ode_params.time_rsh_to_h - h_d / self.ode_params.time_h_to_d
        dc_r = self.ode_params.prop_c_to_r * (h_c / self.ode_params.time_h_to_c + r_sc / self.ode_params.time_rsc_to_c)\
            - c_r / self.ode_params.time_c_to_r
        dc_d = self.ode_params.prop_c_to_d * (h_c / self.ode_params.time_h_to_c + r_sc / self.ode_params.time_rsc_to_c)\
            - c_d / self.ode_params.time_c_to_d
        dr_a = i_a / self.ode_params.time_infectious
        dr_m = i_m / self.ode_params.time_infectious
        dr_h = h_r / self.ode_params.time_h_to_r
        dr_c = c_d / self.ode_params.time_c_to_d
        dd_h = h_d / self.ode_params.time_h_to_d
        dd_c = c_d / self.ode_params.time_c_to_d

        return np.stack([ds, de, di_a, di_m, di_s, dr_sh, dr_sc, dh_r, dh_c, dh_d,
                         dc_r, dc_d, dr_a, dr_m, dr_h, dr_c, dd_h, dd_c], axis=0)

    @classmethod
    def from_default(cls):
        meta_params = MetaParams.from_default()
        lockdown_params = LockdownParams.from_default(meta_params.nb_samples)
        ode_params = OdeParams.from_default(meta_params.nb_samples)
        return cls(meta_params=meta_params, lockdown_params=lockdown_params, ode_params=ode_params)

    @classmethod
    def from_cli(cls, meta_cli: MetaCLI, lockdown_cli: LockdownCLI, ode_cli: OdeParamCLI):
        meta_params = MetaParams.from_cli(meta_cli)
        lockdown_params = LockdownParams.from_cli(lockdown_cli, meta_params.nb_samples)
        ode_params = OdeParams.from_cli(ode_cli, meta_params.nb_samples)
        return cls(meta_params=meta_params, lockdown_params=lockdown_params, ode_params=ode_params)
