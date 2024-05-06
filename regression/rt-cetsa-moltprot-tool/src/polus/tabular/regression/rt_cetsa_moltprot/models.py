"""Copyright 2020,2021 Vadim Kotov, Thomas C. Marlovits.

This file is part of MoltenProt.

MoltenProt is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MoltenProt is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MoltenProt.  If not, see <https://www.gnu.org/licenses/>.
"""
"""
NOTE
This file formalizes models used in MoltenProt fitting (MPModel class)
User-defined models can be added here
"""

# finding number of arguments in a function
from inspect import signature

import numpy as np

# numeric integration
from scipy.integrate import solve_ivp

### Constants
R = 8.314  # universtal gas constant
T_std = 298.15  # standard temperature, in Kelvins


class MoltenProtModel:
    # dummy function is defined here, because it does not make sense for the
    # instances to be able to have a different function
    # the first argument HAS to be the X-axis (independent var)
    fun = lambda x, k, b: k * x + b

    # same applies for the model description - no point to make an instance-specific value
    _description = "A dummy MoltenProt model"

    # since a single class is created for each model, then the long name is already in the class name
    # here we can also add a short name used in the main module
    short_name = "model_template"

    # what measure to use in final sorting - calculated in MoltenProtFit
    # NOTE all measures are selected in such a way that higher values correspond to higher stability
    # if None, then final sorting is skipped
    sortby = None

    def __init__(self, scan_rate=None) -> None:
        """In a general case scan rate is not relevant, so it is set to None
        For kinetic data it needs to be set for sure.
        """
        self.scan_rate = scan_rate

    def __repr__(self) -> str:
        """For interactive prompts or print() - show the model description."""
        return self._description

    def __str__(self) -> str:
        """When converting to a string return the short name."""
        return self.short_name

    def param_names(self):
        """Return parameter names encoded in the function declaration as a list."""
        params = signature(self.fun).parameters
        # skip parameter self
        return list(params)[1:]

    def param_init(self, input_data=None):
        """Return starting parameters based on the input values
        input_data must be a pd.Series with an index (e.g. Temperature)
        returns None if the parameters should be guessed by curve_fit.
        """
        if input_data is None:
            return

    def param_bounds(self, input_data=None):
        """Return the bounds based on the input data.
        input_data must be a pd.Series with an index (e.g. Temperature)
        returns (-np.inf, np.inf) if no bounds are to be set.
        """
        if input_data is None:
            return (-np.inf, np.inf)
        return None


class EquilibriumTwoState(MoltenProtModel):
    short_name = "santoro1988"
    _description = "N <-> U"
    sortby = "dG_std"  # type: ignore[assignment]

    # original function
    # d -> dHm
    # NOTE parameter names do matter. If kN,bN,kU,bU and Tm are present, then _estimate_baseline routine
    # will be run by MoltenProtFit instance to get the best possible starting values
    def fun(self, T, kN, bN, kU, bU, dHm, Tm):
        return (kN * T + bN + (kU * T + bU) * np.exp(dHm / R * (1 / Tm - 1 / T))) / (
            1 + np.exp(dHm / R * (1 / Tm - 1 / T))
        )

    def param_bounds(self, input_data=None):
        # if no data supplied, run the default action from the master class
        # otherwise compute bounds from plate index or hard-coded
        if input_data is None:
            return super().param_bounds(None)
        else:
            return (
                (-np.inf, -np.inf, -np.inf, -np.inf, 60000, min(input_data.index)),
                (np.inf, np.inf, np.inf, np.inf, 4000000, max(input_data.index)),
            )

    def param_init(self, input_data=None):
        # Initial parameters - pre baseline has no intercept and 45 degree slope
        # original implementation did not put values for Tm, which was computed dynamically in main code
        # here a good starting Tm is just the middle of th range (or 0 if no data provided)
        # NOTE for custom models any parameter initialization code should be implemented here
        if input_data is None:
            return (1, 0, 2, 0, 100000, 0)
        else:
            return (
                1,
                0,
                2,
                0,
                100000,
                min(input_data.index)
                + (min(input_data.index) + max(input_data.index)) / 2.0,
            )


class EquilibriumThreeState(MoltenProtModel):
    short_name = "santoro1988i"
    _description = "N <-> I <-> U"
    # in theory total stability of the protein is the sum of stabilities of N and I
    sortby = "dG_comb_std"  # type: ignore[assignment]

    def fun(self, T, kN, bN, kU, bU, kI, dHm1, T1, dHm2, dT2_1):
        # dT2_1 = T2 - T1, i.e. the distance between the two transitions
        return (
            kN * T
            + bN
            + kI * np.exp(dHm1 / R * (1 / T1 - 1 / T))
            + (kU * T + bU)
            * np.exp(dHm1 / R * (1 / T1 - 1 / T))
            * np.exp(dHm2 / R * (1 / (T1 + dT2_1) - 1 / T))
        ) / (
            1
            + np.exp(dHm1 / R * (1 / T1 - 1 / T))
            + np.exp(dHm1 / R * (1 / T1 - 1 / T))
            * np.exp(dHm2 / R * (1 / (T1 + dT2_1) - 1 / T))
        )

    def param_bounds(self, input_data=None):
        # TESTING preliminary results show that no limits for dHm are better in intermediate mode
        if input_data is None:
            return super().param_bounds(None)
        else:
            # by definition T2 follows T1, so dT2_1 is > 0
            # the upper bound for dT2_1 is 1/2 of the full temperature range (i.e. the limit on max distance between the two Tms)
            return (
                (
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    min(input_data.index),
                    -np.inf,
                    0,
                    # allow dT2_1 to be +/-
                ),
                (
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    max(input_data.index),
                    np.inf,
                    (max(input_data.index) - min(input_data.index)) / 2,
                ),
            )

    def param_init(self, input_data=None):
        # Initial parameters - pre baseline has no intercept and are 45 degree slope
        # For dT2_1 we start from the assumption that dT2_1=0, i.e. there is no 2nd transition
        if input_data is None:
            return (1, 0, 2, 0, 1, 100000, 0, 100000, 0)
        else:
            # T1 is heuristically placed in the middle of the temp range
            temp_range = max(input_data.index) - min(input_data.index)
            return (
                1,
                0,
                2,
                0,
                1,
                100000,
                min(input_data.index) + 0.5 * temp_range,
                100000,
                0,
            )


class EmpiricalTwoState(MoltenProtModel):
    short_name = "santoro1988d"
    _description = "Same as santoro1988, but fits Tm and T_onset"
    sortby = "T_eucl"  # type: ignore[assignment]
    # NOTE onset threshold is hard-coded to 0.01, i.e. onset point is 1% unfolded
    onset_threshold = 0.01

    def fun(self, T, kN, bN, kU, bU, T_onset, Tm):
        return (
            kN * T
            + bN
            + (kU * T + bU)
            * np.exp(
                (T - Tm)
                * np.log(self.onset_threshold / (1 - self.onset_threshold))
                / (T_onset - Tm),
            )
        ) / (
            1
            + np.exp(
                (T - Tm)
                * np.log(self.onset_threshold / (1 - self.onset_threshold))
                / (T_onset - Tm),
            )
        )

    def param_bounds(self, input_data=None):
        if input_data is None:
            return super().param_bounds(None)
        else:
            return (
                (
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    min(input_data.index),
                    min(input_data.index),
                ),
                (
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    max(input_data.index) - 2,
                    max(input_data.index),
                ),
            )

    def param_init(self, input_data=None):
        # Initial parameters - pre baseline has no intercept and are 45 degree slope
        if input_data is None:
            return (1, 0, 2, 0, 1, 2)
        else:
            return (
                1,
                0,
                2,
                0,
                min(input_data.index) + 10,
                (min(input_data.index) + max(input_data.index)) / 2.0,
            )


class EmpiricalThreeState(MoltenProtModel):
    short_name = "santoro1988di"
    _description = "Same as santoro1988i, but fits Tm and T_onset"
    # similar to thermodynamic 3-state model: sum up Euclidean temperature distance
    # for both reaction steps
    sortby = "T_eucl_comb"  # type: ignore[assignment]
    # NOTE onset threshold is hard-coded to 0.01, i.e. onset point is 1% unfolded
    onset_threshold = 0.01

    def fun(self, T, kN, bN, kU, bU, kI, T_onset1, T1, T_onset2, T2):
        return (
            kN * T
            + bN
            + kI
            * np.exp(
                (T - T1)
                * np.log(self.onset_threshold / (1 - self.onset_threshold))
                / (T_onset1 - T1),
            )
            + (kU * T + bU)
            * np.exp(
                (T - T2)
                * np.log(self.onset_threshold / (1 - self.onset_threshold))
                / (T_onset2 - T2),
            )
            * np.exp(
                (T - T1)
                * np.log(self.onset_threshold / (1 - self.onset_threshold))
                / (T_onset1 - T1),
            )
        ) / (
            1
            + np.exp(
                (T - T1)
                * np.log(self.onset_threshold / (1 - self.onset_threshold))
                / (T_onset1 - T1),
            )
            + np.exp(
                (T - T2)
                * np.log(self.onset_threshold / (1 - self.onset_threshold))
                / (T_onset2 - T2),
            )
            * np.exp(
                (T - T1)
                * np.log(self.onset_threshold / (1 - self.onset_threshold))
                / (T_onset1 - T1),
            )
        )

    def param_bounds(self, input_data=None):
        if input_data is None:
            return super().param_bounds(None)
        else:
            return (
                [
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    min(input_data.index),
                    min(input_data.index),
                    min(input_data.index),
                    min(input_data.index),
                ],
                [
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    max(input_data.index),
                    max(input_data.index),
                    max(input_data.index),
                    max(input_data.index),
                ],
            )

    def param_init(self, input_data=None):
        # Initial parameters - pre baseline has no intercept and are 45 degree slope
        if input_data is None:
            return (1, 0, 2, 0, 1, 1, 1, 1, 1)
        else:
            temp_range = max(input_data.index) - min(input_data.index)
            # with these starting values first transition is supposed to start in the beginning of the curve
            # and the second transition starts in the end of the curve. Then they should meet
            return (
                1,
                0,
                2,
                0,
                1,
                min(input_data.index) + 0.2 * temp_range,
                min(input_data.index) + 0.4 * temp_range,
                max(input_data.index) - 0.4 * temp_range,
                max(input_data.index) - 0.2 * temp_range,
            )


class IrreversibleTwoState(MoltenProtModel):
    short_name = "irrev"
    _description = "N -> U"
    sortby = "pk_std"  # type: ignore[assignment]
    xn = 1  # y0 (starting condition) for differential equation

    def __init__(self, scan_rate) -> None:
        # scan rate is an essential parameter and must thus be set explicitly
        if scan_rate is not None:
            self.scan_rate = scan_rate
        else:
            msg = f"{self.short_name} model requires scan_rate to be set"
            raise ValueError(
                msg,
            )

    def arrhenius(self, t, Tf, Ea):
        """Arrhenius equiation: defines dependence of reaction rate constant k on temperature
        In this version of the equation we use Tf (a temperature of k=1)
        to get rid of instead of pre-exponential constant A.
        """
        return np.exp(-Ea / R * (1 / t - 1 / Tf))

    def ode(self, t, xn, Tf, Ea):
        """Ordinary differential equation for fraction native versus temperature
        dxn/dT = -1/v*k(T)*xn.

        start_value xn - should be always 1 because at the start of assay we assume everything is folded
        v - scan rate to convert minutes of scan rate to degrees (default 1)
        xn - fraction native (xn + xagg = 1)
        k(T) - temperature-dependent rate constant of aggregation
        """
        return -1 / self.scan_rate * self.arrhenius(t, Tf, Ea) * xn

    def fun(self, t, kN, bN, kU, bU, Tf, Ea):
        """Returns aggregation signal at given temperature
        Signal(T) = (kN*T + bN)*xn +(kU*T + bU)*xu
        k, b - baseline parameters (N or U state)
        xn, xu - fraciton native/unfolded, xn + xu = 1
        in other words:
        Signal(T) = kU*T + bU + (kN*T + bN - kU*T - bU) * xn.
        """
        # step 1: numerically integrate agg_ode for given parameters - gives xn(T)
        ivp_result = solve_ivp(
            self.ode,
            t_span=[min(t), max(t)],
            t_eval=t,
            y0=[self.xn],
            args=(Tf, Ea),
            method="BDF",
        )

        # step 2: return the result of the signal
        return kU * t + bU + (kN * t + bN - kU * t - bU) * ivp_result.y[0, :]

    def param_init(self, input_data=None):
        if input_data is None:
            # without input data it's hard to guess starting values
            # but it seems that starting with a high Tf may help
            # kN, bN, kU, bU, Tf, Ea
            return (0, 1, 0, 1, 400, 100000)
        else:
            # the baselines will have a better guess in MoltenProt
            # since Tf may or may not coincide with the derivative peak
            # it is taken as the middle of the temp range
            return (
                0,
                1,
                0,
                1,
                min(input_data.index)
                + (max(input_data.index) - min(input_data.index)) / 2.0,
                50000,
            )

    def param_bounds(self, input_data=None):
        # NOTE it may happen that Tf is ouside the temperature range, but the curve is still OK
        # also, MoltenProt calculations are always done in Kelvins
        # thus, the default bounds are quite relaxed
        if input_data is None:
            return (
                (-np.inf, -np.inf, -np.inf, -np.inf, 1, 0),
                (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
            )
        else:
            return (
                (-np.inf, -np.inf, -np.inf, -np.inf, min(input_data.index) - 100, 0),
                (np.inf, np.inf, np.inf, np.inf, max(input_data.index) + 100, np.inf),
            )


"""
NOTE given the current computational workload even of a simple kinetic equiation implementation of more complex
equations makes little sense

class IrreversibleThreeState(IrreversibleTwoState):
    short_name = 'irrev'
    _description = "N -> I -> U"
"""

# TODO another (potentially useful) variant could be to implement N <-kF,kR-> U without any A
# the class can be called ReversibleTwoState, because there are only U and N, and U can sometimes become N


class LumryEyring(IrreversibleTwoState):
    short_name = "lumry_eyring"
    _description = "N <- kF,kR -> U -> A"

    # the kF/kR at std temperature; take as -log10 to have higher values for higher stability
    sortby = "pk_ratio_std"  # type: ignore[assignment]

    # fmt: off
    #        ~_~
    z0 =    (0,0)     # starting values for system of 2x ode
    #      /,www,\
    #      ||wWw||    an owl?!
    #       \|||/
    #     ~~~m'm~~~
    # fmt: on
    def __init__(self, scan_rate, tfea=[None, None]) -> None:
        # scan rate is an essential parameter and must thus be set explicitly
        if scan_rate is not None:
            self.scan_rate = scan_rate
        else:
            msg = f"{self.short_name} model requires scan_rate to be set"
            raise ValueError(
                msg,
            )

        # tfea are characteristics of the irreversible aggregation reaction (Tf and Ea)
        # and are obtained separately; they are fixed during fitting
        self.tfea = tfea

    def set_fixed(self, tfea):
        """Set parameters that are needed in the fit equation, but not being fit
        tfea must be a list with two values [Tf, Ea] (floats)
        NOTE only the length of the input is checked, but not the type of list elements.
        """
        if len(tfea) != 2:
            msg = "LE model requires tfea to be a list of two floats"
            raise ValueError(msg)
        self.tfea = tfea

    def ode(
        self,
        t,
        z0,
        TfF,
        EaF,
        TfR,
        EaR,
        Tf2,
        Ea2,
    ):
        """A function to process a system of differential equations x and y packaged into array z = [x,y]
        Implements a general case of Lumry-Eyring equation (case D in Mazurenko2017)
        N <- kF, kR -> U - k2 -> A
        there is an quasi-equilibrium between N and U, but irreversible conversion to A
        As described in the paper:

        fN + fU + fA = 1
        dfAdT = 1/v * k2(T) * fU(T) ...................................... (eq. y)
        dfUdT = 1/v* ( kF(T)*fN - (kR(T)+k2(T))*fU )

        since fN = 1 - fU - fA, can rewrite dfUdT like this:

        dfUdT = 1/v * ( kF(T) * (1 - fU - fA) - ( kR(T)+k2(T) )*fU ) ..... (eq. x)

        the formula for rate in reaction i would then be:
        ki(T) = exp(-Eai/R * (1/T - 1/Tfi))
        i can be F or R for N<->U and 2 for U->A

        t - temperature scale (converted to kinetic time using v, the scan rate (global constant)
        z0 - starting values for equations in the system [fU=0, fA=0]
        """
        # unpack initial values
        x0, y0 = z0

        dxdt = (
            1
            / self.scan_rate
            * (
                self.arrhenius(t, TfF, EaF) * (1 - x0 - y0)
                - (self.arrhenius(t, TfR, EaR) + self.arrhenius(t, Tf2, Ea2)) * x0
            )
        )
        dydt = 1 / self.scan_rate * self.arrhenius(t, Tf2, Ea2) * x0

        return (dxdt, dydt)

    # def fun(self, T, TfF, EaF, TfR, EaR, kNF, bNF, kUF, kAF, bAF):
    def fun(self, T, kN, bN, kU, bU, kI, TfF, EaF, TfR, EaR):
        """Uses pre-computed Tf and Ea for scattering data to model fluorescence signal
        NOTE the original order and naming of the parameters is as follows:
        T, TfF, EaF, TfR, EaR, kNF, bNF, kUF, kAF, bAF
        T - temperature/time
        TfF - temperature at which the rate constant kF=1 (reaction N->U)
        EaF - activation energy for N->U
        TfR,EaR - Tf and Ea for reaction U->N
        kNF, bNF - slope and intercept for the baseline of state N
        kUF - slope for the fluorescence of state U (assumed to be short-lived and not abundant, see Bedouelle2016)
        kAF, bAF - slope/intercept for the baseline of state A, which ultimately makes up the post-transition baseline.

        For MoltenProt to recognize pre- and post- baseline parameters the have to be placed first
        and renamed as follows:
        kNF, bNF - kN, bN
        kAF, bAF - kU, bU
        kUF - kI (similar to three-state cases above)

        NOTE this is not the only way to define the law of signal; for instance, we can assume
        that the fluorescence has a similar time dependence for states U and A; then kI (aka kUF) is not needed
        and the law of signal will be:
        (kN * T + bN) * fN + (kU*T + bU) * (fA + fU)
        """
        ivp_result = solve_ivp(
            self.ode,
            t_span=[min(T), max(T)],
            y0=self.z0,
            args=(TfF, EaF, TfR, EaR, self.tfea[0], self.tfea[1]),
            t_eval=T,
            method="BDF",
        )

        # based on diff eqn compute fractions of each state
        fU = ivp_result.y[0, :]
        fA = ivp_result.y[1, :]
        fN = 1 - fU - fA
        # return modelled fluorescence signal
        return (kN * T + bN) * fN + kI * fU + (kU * T + bU) * fA

    def param_init(self, input_data=None):
        if input_data is None:
            # without input data it's hard to guess starting values
            # kN, bN, kU, bU, kI, TfF, EaF, TfR, EaR
            return (0, 1, 0, 1, 0, 400, 100000, 400, 100000)
        else:
            # the baselines will have a better guess in MoltenProt
            # Tf is probably not similar to Tm, however, it makes sense to try the middle of the range
            temp_range = max(input_data.index) - min(input_data.index)
            return (
                0,
                1,
                0,
                1,
                0,
                min(input_data.index) + temp_range / 2.0,
                50000,
                min(input_data.index) + temp_range / 2.0,
                50000,
            )

    def param_bounds(self, input_data=None):
        # NOTE it may happen that Tf is ouside the temperature range, but the curve is still OK
        # also, MoltenProt calculations are always done in Kelvins
        # thus, the default bounds are quite relaxed
        # NOTE Tf should not be zero, see the Arrhenius formula
        if input_data is None:
            return ((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 1, 0, 1, 0), np.inf)
        else:
            return (
                (
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    min(input_data.index) - 100,
                    0,
                    min(input_data.index) - 100,
                    0,
                ),
                (
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    max(input_data.index) + 100,
                    np.inf,
                    max(input_data.index) + 100,
                    np.inf,
                ),
            )
