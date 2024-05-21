"""Copyright 2018-2021 Vadim Kotov, Thomas C. Marlovits.

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

### Citation
# a simple dict with strings that provide different citation formatting
citation = {
    "long": "\nIf you found MoltenProt helpful in your work, please cite:\nKotov et al., Protein Science (2021)\ndoi: 10.1002/pro.3986\n",
    "html": """<p>If you found MoltenProt helpful in your work, please cite: </p>
                      <p>Kotov et al., Protein Science (2021)</p>
                      <p><a href="https://dx.doi.org/10.1002/pro.3986">doi: 10.1002/pro.3986</a></p>""",
    "short": "Citation: Kotov et al., Protein Science (2021) doi: 10.1002/pro.3986",
}


### Modules
# some useful mathematical functions
# For printing line number and file name
from inspect import currentframe
from inspect import getframeinfo

# for generating htmls
from string import Template

# plotting
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# for color conversion
from matplotlib.colors import rgb2hex

# interpolation
from scipy.interpolate import interp1d

# fitting routine from scipy
from scipy.optimize import curve_fit

# median filtering
from scipy.signal import medfilt

cf = currentframe()
cfFilename = getframeinfo(cf).filename  # type: ignore[arg-type]

# handling exceptions
# saving class instances to JSON format
import json

# creating folders
import os
import sys

# function to recognize module versions
from distutils.version import LooseVersion

# for compression of output JSON
# for timestamps
from time import strftime

# data processing
import pandas as pd

# import the fitting models
from . import models

# A variable for reliable access to other resources of MoltenProt (e.g. report template)
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# MoltenProt is stored in a plain-text file VERSION (also used by setup.py)
# extract and save it to a variable
with open(os.path.join(__location__, "MOLTENPROT_VERSION")) as version_file:
    __version__ = version_file.read().strip()

# get scipy version (some methods may not be available in earlier versions)
scipy_version = sys.modules["scipy"].__version__

# check if running from a PyInstaller bundle
if hasattr(sys, "frozen") and hasattr(sys, "_MEIPASS"):
    from_pyinstaller = True
else:
    from_pyinstaller = False

# for parallel processing of some for loops using joblib (may hang up)
try:
    from joblib import Parallel
    from joblib import delayed

    # this is only needed if we want to auto-estimate the available cores - currently done in MoltenProtMain
    # access module version without importing the whole module:
    joblib_version = sys.modules["joblib"].__version__
    # check joblib version: only 0.12 and above can pickle instance methods (which is used in mp)
    if LooseVersion(joblib_version) >= LooseVersion("0.12"):
        parallelization = True
    else:
        print(
            "Warning: available joblib version ({}) is incompatible with current code parallelization".format(
                joblib_version,
            ),
        )
        print("Information: Consider updating to joblib 0.12 or above")
        parallelization = False
except ImportError:
    print("Warning: joblib module not found, parallelization of code is not possible.")
    joblib_version = "None"  # only for printing version info
    parallelization = False

# NOTE MoltenProtFit and MoltenProtFitMultiple have different parallelization approaches:
# MoltenProtFit - can only parallelize figure plotting and n_jobs=3 works well
# MoltenProtFitMultiple - reads and runs several MoltenProtFit instances in parallel (F330, F350 etc),
# but each of them gets only one job (i.e. MoltenProtFit.n_jobs is always 1)

### Constants imported from models.py
R = models.R
T_std = models.T_std

# Standard plate index
# NOTE currently the software requires the layout to contain A1-H12 index, even if only a small part is used
alphanumeric_index = []
for k in ["A", "B", "C", "D", "E", "F", "G", "H"]:
    for l in range(1, 13):
        alphanumeric_index.append(k + str(l))
# convert index to pandas Series and set its name
alphanumeric_index = pd.Series(data=alphanumeric_index)
alphanumeric_index.name = "ID"  # type: ignore[attr-defined]

# dictionary holding the default values and and their description for CLI interface/tooltips
# dictionary key is the name of the option, each entry contains a tuple of default parameter value and its descriptions
# NOTE for mfilt the default value is only used when the option is supplied in the CLI

# default data preparation parameters
prep_defaults = {
    "blanks": [],
    "blanks_h": "Input the sample ID's with buffer-only control",  # subtract blanks in prep step
    "exclude": [],
    "exclude_h": 'Specify the well(s) to omit during analysis; this option is intended for simple removal of some bad wells; if many samples must be excluded, use a layout and add "Ignore" to the annotation of the sample',
    "invert": False,  # DELETE?
    "invert_h": "Invert the curve",
    "mfilt": None,
    "mfilt_h": "Apply median filter with specificed window size (in temperature units) to remove spikes; 4 degrees is a good starting value",
    "shrink": None,
    "shrink_h": "Average the data points to a given degree step;\n may help to make trends more apparent and speeds up the computation;\n typical values 0.5-3.0",
    "trim_max": 0,
    "trim_max_h": "Decrease the finishing temperature by this value",
    "trim_min": 0,
    "trim_min_h": "Increase the starting temperature by this value",
}

# default analysis parameters
analysis_defaults = {
    "model": "santoro1988",
    "model_h": "Select a model for describing the experimental data",
    "baseline_fit": 10,
    "baseline_fit_h": "The length of the input data (in temperature degrees) for initial estimation of pre- and post-transition baselines",
    "baseline_bounds": 3,
    "baseline_bounds_h": "Baseline bounds are set as multiples of stdev of baseline parameters obtained in the pre-fitting routine; this should stabilize the fit and speed up convergence; set to 0 to remove any bounds for baselines",
    "dCp": 0,
    "dCp_h": "Heat capacity change of unfolding for all samples (in J/mol/K), used only in equilibrium models; this value overrides the respective column in the layout",
    "onset_threshold": 0.01,
    "onset_threshold_h": "Percent unfolded to define the onset of unfolding",
    "savgol": 10,
    "savgol_h": "Set window size (in temperature units) for Savitzky-Golay filter used to calculate the derivative",
}

# all other settings for MoltenProtFit
defaults = {
    "debug": False,  # currently not exposed in the CLI
    "debug_h": "Print developer information to the console",
    "dec": ".",
    "dec_h": "CSV decimal separator, enclosed in quotes",
    "denaturant": "C",
    "denaturant_h": "For plain CSV input only; specify temperature scale that drives denaturation, in K or C",
    "j": 1,  # TODO not related to the core functions, supply to respective methods (output etc)
    "j_h": "Number of jobs to be spawned by parallelized parts of the code; should not be higher than the amount of CPU's in the computer; for most recent laptops a value of 3 is recommended",
    "layout": None,  # TODO should not be set in SetAnalysisOptions, but rather in __init__
    "layout_h": "CSV file with layout",
    "sep": ",",
    "sep_h": "CSV separator, enclosed in quotes",
    "readout": "Signal",
    "readout_h": "For plain CSV input only; specify type of input signal",
    "spectrum": False,
    "spectrum_h": "If true, columns in the input CSV will be treated as separate wavelengths of a spectrum",
    "heatmap_cmap": "coolwarm_r",  # a color-safe heatmap color with red being "bad" (low value)
    "heatmap_cmap_h": "Matplotlib code for colormap that would be used to color-code heatmaps in reports or images",
}

# dictionary with avialble models
# NOTE to be automatically identified the model has to be subclass or subsubclass of MoltenProtModel
avail_models = {}
for model in models.MoltenProtModel.__subclasses__():
    avail_models[model.short_name] = model  # subclass of MoltenProtModel
    for submodel in model.__subclasses__():  # subsubclass of MoltenProtModel
        avail_models[submodel.short_name] = submodel

# add a dummy model to indicate that the dataset should not be analysed
avail_models["skip"] = "skip"  # type: ignore[assignment]

### Utility functions


def normalize(input, new_min=0, new_max=1, from_input=False):
    """Helper function to normalize a pandas Series.

    Make the data occupy a specified range (defined by new_min and new_max)

    Parameters
    ----------
    input : pd.Series
        input Series to normalize
    new_min, new_max : float
        range for normalization (default [0-1])
    from_input : bool
        use absolute vales of min/max of the input series to create normalization range
        this is used to perform inversion of curves (mirror around X-axis)

    References:
    ----------
    The general formula for normalization is from here:
    https://en.wikipedia.org/wiki/Normalization_(image_processing)
    """
    if from_input:
        new_min = min(abs(input))
        new_max = max(abs(input))

    return (input - input.min()) * (new_max - new_min) / (
        input.max() - input.min()
    ) + new_min


def to_odd(input, step):
    """Helper function to convert the window length in temperature units to an odd number of datapoints.

    NOTE for user's simplicity the window size is specified in temperature units
    for filters, however, an odd integer is needed

    Parameters
    ----------
    input : float
        number in degrees to convert to a number of datapoints
    step : float
        temperature step in the dataset

    Returns:
    -------
    float
        an odd number of datapoints which should more or less fit in the
        requested temperature range
    """
    # calculate the approximate amount of datapoints corresponding to the range
    input = input / step
    # convert input to an integer
    input = int(input)
    # if it can be divided by 2 without the remainder (Euclidean division)
    # than it is even and needs to become odd
    # % computes the remainder from Euclidean division
    # the complementary operator is //
    if not input % 2:
        input = input + 1
    return input


def analysis_kwargs(input_dict):
    """Takes a dict and returns a valid argument dict for SetAnalysisOptions
    Unknown options are removed, default values are supplied.
    """
    output = {}
    for i, j in input_dict.items():
        if i in list(analysis_defaults.keys()) or i in list(prep_defaults.keys()):
            output[i] = j
    return output


def showVersionInformation():
    """Print dependency version info."""
    print(cfFilename, currentframe().f_lineno)
    print(f"MoltenProt v. {__version__}")
    if from_pyinstaller:
        print("(PyInstaller bundle)")
    pd.show_versions(as_json=False)  # matplotlib is also there
    print(f"joblib           : {joblib_version}")


### Wrappers


def mp_read_excel(filename, sheetname, index_col):
    """Read XLSX files to pandas DataFrames and return None if errors occur.

    Parameters
    ----------
    filename
        filename for reading
    sheetname
        which sheet to read from file
    index_col
        which column to use as index

    Returns:
    -------
    pd.DataFrame from Excel file or None if reading failed
    """
    try:
        # NOTE in pandas below 0.24 even though the data should be considered index-free (index_col=None)
        # the first column was sometimes used as index (corresponds to Time in Prometheus XLSX)
        # for now explicitly pass some column to use as index
        return pd.read_excel(filename, sheetname, index_col=index_col)
    except:
        # NOTE XLSX parsing module has its own error naming system, so individual errors cannot be catched unless xlrd module is imported
        print(
            'Warning: sheet "{}" does not exist in the input file {}'.format(
                sheetname,
                filename,
            ),
        )
        return None


### JSON I/O functions and variables


def serialize(obj):
    """Helper function for serialization of DataFrames and other non-standard objects."""
    if isinstance(obj, pd.core.frame.DataFrame):
        # NOTE original data contains 12 decimal digits, while default encoding parameter is 10
        # pandas calculates with 16 decimal digits (or more?) and in this part there will be certain discrepancy
        # between the runs (i.e. run analysis>save to json>run analysis again)
        # in any case the impact on the ultimate result is not measurable
        return {
            "DataFrame": obj.to_json(
                force_ascii=False,
                double_precision=15,
                orient="split",
            ),
        }
    elif isinstance(obj, MoltenProtFit):
        output = obj.__dict__
        # delete some method descriptions (only needed for parallel processing)
        if "plotfig" in output:
            del output["plotfig"]

        # delete layout attribute, it will be set from parent MPFM when loading back
        if "layout" in output:
            del output["layout"]

        return {"MoltenProtFit": output}
    elif isinstance(obj, MoltenProtFitMultiple):
        output = obj.__dict__
        # a better way to query a dict, see:
        # https://docs.quantifiedcode.com/python-anti-patterns/correctness/not_using_get_to_return_a_default_value_from_a_dictionary.html
        # TODO this deletion is probably not needed?
        if output.get("PrepareAndAnalyseSingle"):
            del output["PrepareAndAnalyseSingle"]
        if output.get("WriteOutputSingle"):
            del output["WriteOutputSingle"]
        # add version info and timestamp
        return {
            "MoltenProtFitMultiple": output,
            "version": __version__,
            "timestamp": strftime("%c"),
        }
    return None


def deserialize(input_dict):
    """Helper function to deserialize MoltenProtFit instances and DataFrames from JSON files (read into a dict).

    If the object contains any plate_ entries, cycle through them and convert to DataFrames
    also applies to the layout entry (again, a DataFrame)
    """
    if input_dict.get("MoltenProtFit"):
        # read a MPF instance from the dict
        # plate_ variables conversion occurs during intitialization.
        # the downstream part of the json (e.g. a single MPFIT instance)
        output = MoltenProtFit(None, input_type="from_dict")
        output.__dict__.update(input_dict["MoltenProtFit"])
        return output
    elif input_dict.get("MoltenProtFitMultiple"):
        # this level also has version info and timestamp
        # NOTE if needed, this section can be used to discard too old JSON sessions
        if input_dict.get("version") and input_dict.get("timestamp"):
            print(
                "Information: JSON session was created with MoltenProt v. {} on {}".format(
                    input_dict["version"],
                    input_dict["timestamp"],
                ),
            )
        else:
            print(
                "Warning: JSON session contains no version and timestamp, is it an old one?",
            )
        # create an empty instance and update its dict
        output = MoltenProtFitMultiple()
        output.__dict__.update(input_dict["MoltenProtFitMultiple"])

        # set layouts in all datasets
        output.UpdateLayout()
        return output
    elif input_dict.get("DataFrame"):
        return pd.read_json(input_dict["DataFrame"], precise_float=True, orient="split")
    return input_dict


### Classes


class MoltenProtFit:
    """Stores and analyses a single dataset.

    Attributes:
    ----------
    defaults : dict
        dictionary holding the default values and and their description
    hm_dic : dict
        dictionary for important heatmap parameters
    plate - the workhorse variable, contains the values currently being procesed
    plate_raw - initially imported data, but without invalid wells or the ones excluded by user
    plate_results - fit values and sort score for individual wells
    plate_derivative - the derivative curve for raw (?) data
    plate_fit - the curves computed based on the fit parameters
    resultfolder - a subfolder created from the file name to write the output (assigned by MoltenProtMain.py)
    dT - step of temperature
    xlim - the range for x-axis on RFU(T) plots
    filename - the name of the input file
    Lists holding removed wells:
    bad_fit - ID of wells that could not be fit
    blanks - blank wells (subtracted prior to analysis) -> now part of SetAnalysisOptions
    exclude - supplied by user through --exclude option -> now part of SetAnalysisOptions
    bad_Tm - the melting temperature is out of range of T
    readout_type - the signal in the assay

    Notes:
    -----
    This is an internal class, do not create it manually.
    All official messages must be sent through print_message() method (enforces selecting message type)
    Output information style:
    "Fatal: " - error that causes the program to stop
    "Warning: " - something may be messed up, but the program can proceed
    "Information: " any other message, that has nothing to do with the program flow
    """

    # dictionary for important heatmap parameters:
    # > which values are good high or low
    # > what is the title for plot
    # > what are the tick labels
    # TODO for Tm, dHm and the like, substitute text info to the min and max numbers
    # TODO use this as is a general hm_dic for all types of analysis
    hm_dic = {
        "S": {
            "lowerbetter": True,
            "title": "Std. Error of Estimate Heatmap",
            "tick_labels": ["Bad fit", "Reference", "Good fit"],
        },
        "dHm_fit": {
            "lowerbetter": False,
            "title": "Unfolding Enthalpy Heatmap",
            "tick_labels": ["Low dHm", "Reference", "High dHm"],
        },
        "Tm_fit": {
            "lowerbetter": False,
            "title": "Melting Temperature Heatmap",
            "tick_labels": ["Low Tm", "Reference", "High Tm"],
        },
        "d_fit": {
            "lowerbetter": True,
            "title": "Slope Heatmap",
            "tick_labels": ["Flat", "Reference", "Steep"],
        },
        "Tm_init": {
            "lowerbetter": False,
            "title": "Melting Temperature Heatmap",
            "tick_labels": ["Low Tm", "Reference", "High Tm"],
        },
        "T_onset": {
            "lowerbetter": False,
            "title": "Onset Temperature Heatmap",
            "tick_labels": ["Low T_ons", "Reference", "High T_ons"],
        },
        "a_fit": {
            "lowerbetter": True,
            "title": "Slope Heatmap",
            "tick_labels": ["Flat", "Reference", "Steep"],
        },
        "Tagg_fit": {
            "lowerbetter": False,
            "title": "Aggregation Temperature Heatmap",
            "tick_labels": ["Low Tagg", "Reference", "High Tagg"],
        },
        "Tagg_init": {
            "lowerbetter": False,
            "title": "Aggregation Temperature Heatmap",
            "tick_labels": ["Low Tagg", "Reference", "High Tagg"],
        },
        "dG_std": {
            "lowerbetter": False,
            "title": "Standard Gibbs Free Energy of Unfolding",
            "tick_labels": ["na", "na", "na"],
        },
        "BS_factor": {
            "lowerbetter": False,
            "title": "Dimesionless Signal Window",
            "tick_labels": ["Narrow", "Reference", "Wide"],
        },
    }

    def __init__(
        self,
        filename,
        scan_rate=None,
        denaturant=defaults["denaturant"],
        sep=defaults["sep"],
        dec=defaults["dec"],
        debug=defaults["debug"],
        input_type="csv",
        parent_filename="",
        readout_type="Signal",
    ) -> None:
        """Parameters
        ----------
        filename
            for csv files the file to parse, when MPF is made by MPFM this will be substituted to the filename used to create MPFM
        scan_rate
            scan rate in degrees per min (required for kinetic models)
        denaturant
            temperature (C or K) or chemical (under construction)
        sep,dec
            csv import parameters
        debug
            print additional info
        input_type
            defines where the data is coming from:
                > csv - default value, corresponds to the standard csv file
                > from_xlsx - means that the instance of MoltenProtFit is being created by MoltenProtFitMultiple, input will be a ready-to-use DataFrame
        parent_filename
            the filename of the original xlsx file
        readout_type
            the name of the readout (for plot Y-axis)

        Notes:
        -----
        """
        # set debugging mode: if the __version__ is "git", then this is a development file
        # and debugging is done by default. In all other cases, it is False
        # TODO add debug to CLI interface
        if __version__[:3] == "git":
            self.debug = True
        else:
            self.debug = debug

        if input_type == "from_dict":
            # this would return an empty object
            # all manipulations are done by deserialize function
            pass
        else:
            # NOTE the logic of overwriting etc is handled by CLI/GUI code
            self.resultfolder = None
            # attribute to hold diagnostic messages.
            self.protocolString = ""
            self.scan_rate = scan_rate
            self.fixed_params = None
            # load the dataset into the plate variable
            if input_type == "csv":
                try:
                    self.plate = pd.read_csv(
                        filename,
                        sep=sep,
                        decimal=dec,
                        index_col="Temperature",
                        encoding="utf-8",
                    )
                except ValueError:
                    self.print_message("Input *.csv file is invalid!", "f")
                    print(
                        "Please check if column called 'Temperature' exists and separators are specified correctly.",
                    )
                # the filename is needed for report title
                self.filename = filename
            elif input_type == "from_xlsx":
                self.plate = filename
                # BUG this is only needed for report generation, a dummy value
                # Ideally, it should be the name of parent xlsx file
                self.filename = parent_filename

            # If requested, convert temperature of input files to Kelvins
            if denaturant == "K":
                # input temperature is already in internal units
                self.denaturant = "K"
            elif denaturant == "C":
                # convert Celsius to Kelvins
                self.plate.index = self.plate.index + 273.15
                self.denaturant = "K"
            elif denaturant == "F":
                msg = "NO Fahrenheit, please!"
                raise ValueError(msg)
            else:
                # if temperature was not recognized, set denaturant to chemical type
                # NOTE currently not implemented in MoltenProtFit
                print(f"Assuming that {denaturant} is a chemical denaturant")
                self.denaturant = denaturant
                msg = "Chemical denaturation not implemented yet"
                raise NotImplementedError(msg)

            # save a copy of raw data
            self.plate_raw = self.plate.copy()

            # compute the (average) step of Temperature (required for conversion of temperature ranges to sets of datapoints)
            self.dT = (max(self.plate.index) - min(self.plate.index)) / (
                len(self.plate.index) - 1
            )

            # for plots only: compute xlim from Temperature range
            self.xlim = [min(self.plate.index) - 5, max(self.plate.index) + 5]

            # currently only for plots: give a proper label to Y-axis
            self.readout_type = readout_type

            self.bad_fit = []  # type: ignore[var-annotated]
            self.bad_Tm = []  # type: ignore[var-annotated]

    def __getstate__(self):
        """A special method to enable pickling of class methods (for parallel exectution)."""
        output = self.__dict__
        output["plotfig"] = self.plotfig
        return output

    def converter96(self, use_column, reference=None):
        """Reads self.plate_results with well ID in the index (A1, A2 etc) and some values in columns (Tm_fit, etc) and returns a DataFrame emulating a 96-well plate where each well has a normalized respective column value.

        Parameters
        ----------
        use_column
            which column to use in self.plate_results
        reference
            well to use as reference

        Notes:
        -----
        Normalization is done based on lowerbetter information from self.hm_dic:
        1 - means the highest possible value and the best value as well
        0 - means the worst and the lowest possible value

        If a reference is supplied, it is subtracted from the data
        Reference code _was not maintained_ for a while and is probably faulty!
        """
        # create a DataFrame emulating a 96-well plate
        output = pd.DataFrame(
            index=["A", "B", "C", "D", "E", "F", "G", "H"],
            columns=list(range(1, 13)),
        )

        # check if the use_column value is a valid one
        self.print_message(
            "Creating heatmap for column {}".format(
                self.plate_results[use_column].name,
            ),
            "i",
        )
        for i in output.index:
            for j in output.columns:
                a = i + str(j)
                if (use_column in self.plate_results.columns) and (
                    a in self.plate_results.index
                ):
                    output.loc[[i], [j]] = self.plate_results.loc[a, use_column]

        # if reference value is supplied, subtract it from the values
        if reference is not None:
            try:
                # probe if the well is OK (not NaN), that's easier to do with the input DataFrame
                output = output - self.plate_results[use_column][reference]
            except KeyError:
                self.print_message(
                    "Supplied reference is invalid, creating a reference-free heatmap",
                    "w",
                )

        # invert the values if lowerbetter=true
        # just a precaution against outdated self.hm_dic
        try:
            if self.hm_dic[use_column]["lowerbetter"]:
                output = output - max(output.max())
                output = output * -1
        except KeyError:
            self.print_message(
                "{} was not found in hm_dic, the colors may be wrong!".format(
                    use_column,
                ),
                "w",
            )

        # bring everything to range 0-1
        # normalize by max value (this way we make sure that Nans changed to 1000 are out of range)
        # NOTE in some rare cases, when there is only one sample left and it has value 0 in plate96
        min_val = output.min().min()
        max_val = output.max().max()
        if min_val == max_val:
            self.print_message(
                "Only one sample left after pre-processing and fitting!",
                "w",
            )
            output = output * 0
        else:
            output = (output - min_val) / (max_val - min_val)
        # make all Nan's equal to 1000
        output.fillna(1000, inplace=True)
        return output

    def heatmap(
        self,
        output_path,
        plate96,
        use_column,
        heatmap_cmap=defaults["heatmap_cmap"],
        lowerbetter=False,
        title="Result Heatmap",
        tick_labels=["Unstable", "Reference", "Stable"],
        pdf_report=False,
        save=True,
    ):
        """Create a heatmap.

        Parameters
        ----------
        output_path
            where to save the image
        plate96
            dataframe created with method converter96
        use_column
            which curve parameter to use for heatmap
        heatmap_cmap
            matplotlib colormap for heatmap
        lowerbetter
            indicated if lower values of a parameters correspond to higher stability (e.g. S)
        title
            the title of the heatmap
        tick_labels
            how to label the colorbar
        pdf_report
            if True, will only return a figure object (and size will be adjusted to meet A4)
        save
            if True, save image to disk
        """
        # tweak colormap to have outside values colored as gray
        cmap = plt.get_cmap(heatmap_cmap).copy()
        cmap.set_over("0.5")

        # check if there are negative values
        # (plate96_sort<0).any() returns a Series with True/False indicating presence of negative values in respective columns
        # we then use any() for this series to get the True statement. Very weird way to do it indeed.
        if ((plate96 < 0).any()).any():
            # negative values are present, so we're dealing with ref-based data set

            # now check if we are dealing with higherbetter or lowerbetter values
            if lowerbetter:
                # we have to invert sign so that all values that are less than the reference are positive
                plate96 = plate96 * -1

            # Normalize positive values by the highest value and negative values by the lowest
            # this is not fully correct because the slope of color(value) dependence line is different for values
            # below zero and above zero. Other ways to do it are too hacky anyway
            plate96[plate96 > 0] = plate96[plate96 > 0] / max(plate96.max())
            plate96[plate96 < 0] = plate96[plate96 < 0] / abs(min(plate96.min()))
            vmin = -1
            tick_values = [-1, 0, 1]
        else:
            vmin = min(plate96.min())
            tick_values = [vmin, 1]
            # this format means how many decimal digits is allowed
            # set tick labels from use_column
            col_min = f"{self.plate_results[use_column].min():.3f}"
            col_max = f"{self.plate_results[use_column].max():.3f}"
            tick_labels = [col_max, col_min] if lowerbetter else [col_min, col_max]

        # Making not available values gray (e.g. bad fit or blanks)
        # either convert pd to numpy array and make a mask, or
        # create figure canvas
        # A4 is 8.3 x 11.7 inches, for report the whole page is needed
        # for other outputs we need half the hight (i.e. A5 in landscape orientation)
        if pdf_report:
            fig, ax = plt.subplots(3, 1, figsize=(8.3, 11.7))
            cbar_shrink = 0.3
            cbar_orient = "horizontal"
            heatmap_axis = ax[0]
            axis_aspect = ["auto"]
        else:
            fig = plt.figure(figsize=(8.3, 11.7 / 2), tight_layout=True)
            cbar_shrink = 8 / 12
            cbar_orient = "vertical"
            heatmap_axis = fig.gca()
            axis_aspect = ["equal", "box"]
        fig.suptitle(
            title,
            fontweight="bold",
        )
        # create the heatmap
        # NOTE in some rare cases when the heatmap consists of a single sample matplotlib will raise a warning:
        """
        RuntimeWarning: invalid value encountered in less_equal
        b = b[(b <= intv[1] + eps) & (b >= intv[0] - eps)]
        RuntimeWarning: invalid value encountered in greater_equal
        b = b[(b <= intv[1] + eps) & (b >= intv[0] - eps)]
        """
        c = heatmap_axis.pcolor(plate96, edgecolors="k", cmap=cmap, vmin=vmin, vmax=1)

        # cycle through all wells and write there the ID
        for i in plate96.index:
            for j in plate96.columns:
                x = plate96.columns.get_loc(j)
                y = plate96.index.get_loc(i)
                heatmap_axis.text(
                    x + 0.5,
                    y + 0.5,
                    i + str(j),
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        # y axis has to be inverted so that well A1 is in top left corner
        heatmap_axis.invert_yaxis()
        # addtional hacks: enforce square size of wells, hide axes and ticks
        heatmap_axis.set_aspect(*axis_aspect)
        heatmap_axis.axis("off")
        # create a colorbar with text labels
        cbar = fig.colorbar(
            c,
            ax=heatmap_axis,
            ticks=tick_values,
            shrink=cbar_shrink,
            orientation=cbar_orient,
        )
        # set colorbar ticks depending on the requested orientation
        if cbar_orient == "horizontal":
            cbar.ax.set_xticklabels(tick_labels)
        elif cbar_orient == "vertical":
            cbar.ax.set_yticklabels(tick_labels)
        # label is not needed, because it will be in the figure title

        if pdf_report:
            # return the figure object for subsequent manipulations
            return (fig, ax)

        if save:
            plt.savefig(
                os.path.join(output_path, "heatmap_" + str(use_column) + ".png"),
                dpi=(200),
                tight_layout=True,
            )
        # clean up after plotting so that no parameters are carried over to genpics
        plt.close("all")
        return None

    def print_message(self, text, message_type):
        """Prints messages and saves them to protocolString.

        Parameters
        ----------
        text
            message text
        message_type : i/w/f
            type of message (Information, Warning, Fatal)
        """
        # line No and file name are only printed in debug mode and only for Fatals and Warnings
        if self.debug and (message_type != "i"):
            cf = currentframe()
            cfFilename = getframeinfo(cf).filename
            print(f"Line {cf.f_back.f_lineno}, in file {cfFilename}")
        if message_type == "f":
            print("Fatal: " + text + f" ({self.readout_type})")
            sys.exit(1)
        elif message_type == "w":
            msg = "Warning: " + text + f" ({self.readout_type})"
            print(msg)
            msg = msg + "\n"
            self.protocolString = self.protocolString + msg
        elif message_type == "i":
            msg = "Information: " + text + f" ({self.readout_type})"
            print(msg)
            msg = msg + "\n"
            self.protocolString = self.protocolString + msg
        else:
            raise ValueError(f"Unknown message type '{message_type}'")

    def _calculate_raw_corr(self):
        """Compute baseline-corrected raw data; requires presence of kN, bN, kU, bU in plate_results.

        Notes:
        -----
        calculate fraction unfolded (funf)
        How to derive the formula:
        K_eq(T) = (Fn(T) - F(T)) / (F(T) - Fu(T))
        K_eq() = funf/fnat = funf / (1-funf)
        funf = (Fn(T) - F(T)) / (Fn(T) - Fu(T))
        where T is temperature, F(T) assay signal, Fu and Fn - baseline signal
        funf - fraction unfolded; fnat - fraction native
        whichever formula is used for calculation, the result is the same

        TODO: the same calculation using plate_fit can be done to yield "fit" variant of funf
        """
        # initiate an empty DataFrame
        self.plate_raw_corr = pd.DataFrame(index=self.plate.index)

        # a helper function for pandas.apply()
        def calculate_raw_corr_series(input_series):
            # NOTE the badly fitted samples will be present in the column names of self.plate
            # but they are not present in the index of self.plate_results, thus we need to check for it
            # Obviously the calculation cannot be done for the curves that could not be fit
            if input_series.name in self.plate_results.index:
                # here we need transposed plate_results to have sample ID's in the columns
                output = (
                    input_series.index * self.plate_results["kN_fit"][input_series.name]
                    + self.plate_results["bN_fit"][input_series.name]
                    - input_series
                ) / (
                    input_series.index * self.plate_results["kN_fit"][input_series.name]
                    + self.plate_results["bN_fit"][input_series.name]
                    - input_series.index
                    * self.plate_results["kU_fit"][input_series.name]
                    - self.plate_results["bU_fit"][input_series.name]
                )
                # calculation above produces an index object, which we have to convert to a pd.Series
                output = pd.Series(
                    output,
                    index=self.plate.index,
                    name=input_series.name,
                )
                self.plate_raw_corr = pd.concat(
                    [self.plate_raw_corr, pd.Series(output)],
                    axis=1,
                )

        self.plate.apply(calculate_raw_corr_series)

    def _estimate_baseline(self, input_series, fit_length, estimate_Tm=False):
        """Estimates pre- and post-transition baselines for a series
        function for a single Series (index is Temperature, name is sample ID, values are RFU's).

        Parameters
        ----------
        input_series
            pd.Series with Temperature in index, name is sample ID, values are the signal
        fit_length
            number of degrees to be used from the start or end of data for fitting
        estimate_Tm
            additionally estimate melting temperature Tm with a heuristic
        """
        # convert fit length in temperature degrees to datapoint number (using self.dT)
        fit_datapoints = int(fit_length / self.dT)
        # NOTE do not use Nan's to prevent issues during fitting
        pre_fit, pre_covm = np.polyfit(
            input_series.dropna().iloc[:fit_datapoints].index,
            input_series.dropna().iloc[:fit_datapoints],
            1,
            cov=True,
        )
        post_fit, post_covm = np.polyfit(
            input_series.dropna().iloc[-fit_datapoints:].index,
            input_series.dropna().iloc[-fit_datapoints:],
            1,
            cov=True,
        )
        self.plate_results.loc[["kN_init"], [input_series.name]] = pre_fit[0]
        self.plate_results.loc[["bN_init"], [input_series.name]] = pre_fit[1]
        self.plate_results.loc[["kU_init"], [input_series.name]] = post_fit[0]
        self.plate_results.loc[["bU_init"], [input_series.name]] = post_fit[1]
        pre_stdev = np.sqrt(np.diagonal(pre_covm))
        post_stdev = np.sqrt(np.diagonal(pre_covm))
        # estimate_Tm stdev of each parameter and write it to plate_results_stdev (used to set bounds)
        self.plate_results_stdev.loc[["kN_init"], [input_series.name]] = pre_stdev[0]
        self.plate_results_stdev.loc[["bN_init"], [input_series.name]] = pre_stdev[1]
        self.plate_results_stdev.loc[["kU_init"], [input_series.name]] = post_stdev[0]
        self.plate_results_stdev.loc[["bU_init"], [input_series.name]] = post_stdev[1]

        if estimate_Tm:
            # now the Tm part - find the maximum of smoothened derivative for S-shaped curves (low-to-high)
            # or the minimum for Z-shaped ones
            # intersection of the difference line with zero -> intersection of baselines
            dintersect = -(post_fit[1] - pre_fit[1]) / (post_fit[0] - pre_fit[0])
            # min/max/middle temperature range
            tmin = min(input_series.index)
            tmax = max(input_series.index)
            tmid = (tmin + tmax) / 2
            # value of baseline difference at tmid
            b_diff = (post_fit[0] - pre_fit[0]) * tmid + (post_fit[1] - pre_fit[1])

            if (dintersect > tmin + fit_length) and (dintersect < tmax - fit_length):
                # NOTE rule out intersecting baselines - in this case post-baseline is not always
                # above or below the pre-baseline
                self.plate_results.loc["Tm_init", input_series.name] = tmid
            else:
                if b_diff > 0:
                    # low-to-high curve - use max of the deriv as Tm_init
                    self.plate_results.loc[
                        "Tm_init",
                        input_series.name,
                    ] = self.plate_derivative[input_series.name].idxmax()
                elif b_diff < 0:
                    # high-to-low curve - use min of the deriv as Tm_init
                    self.plate_results.loc[
                        "Tm_init",
                        input_series.name,
                    ] = self.plate_derivative[input_series.name].idxmin()
                else:
                    # rare case - curves are identical raise a warning and use mid-range
                    self.print_message(
                        "Baselines are identical in sample {}".format(
                            input_series.name,
                        ),
                        "w",
                    )
                    self.print_message(
                        "Using the middle of temperature range as Tm_init",
                        "i",
                    )
                    self.plate_results.loc["Tm_init", input_series.name] = tmid

    def _calc_Tons(self, Tm_col, dHm_col, onset_threshold):
        """Computes onset temperature Tons based on supplied column names with dHm and Tm
        and adds the value to plate_results.

        Parameters
        ----------
        Tm_col
            column with Tm values (Tm_fit, Tm1_fit, etc)
        dHm_col
            column with slope values
        onset_threshold
            fraction unfolded that corresponds to onset of unfolded (e.g. 0.01 - 1% must be unfolded)
        """
        self.plate_results["T_onset"] = 1 / (
            1 / self.plate_results[Tm_col]
            - R
            / self.plate_results[dHm_col]
            * np.log(onset_threshold / (1 - onset_threshold))
        )

        # also calculate stdev for T_onset using error propagation from Tm_fit and dHm_fit
        self.plate_results_stdev["T_onset"] = (
            np.sqrt(
                (self.plate_results_stdev[Tm_col] / self.plate_results[Tm_col]) ** 2
                + (self.plate_results_stdev[dHm_col] / self.plate_results[dHm_col]) ** 2
                + (
                    np.sqrt(
                        self.plate_results_stdev[dHm_col] ** 2
                        + (
                            R
                            * np.log(onset_threshold / (1 - onset_threshold))
                            * self.plate_results_stdev[Tm_col]
                        )
                        ** 2,
                    )
                    / (
                        self.plate_results[Tm_col]
                        - R
                        * np.log(onset_threshold / (1 - onset_threshold))
                        * self.plate_results[Tm_col]
                    )
                )
                ** 2,
            )
            * self.plate_results.T_onset
        )

    def plotfig(
        self,
        output_path,
        wellID,
        datatype="overview",
        save=True,
        show=False,
        data_ax=None,
        vline_legend=False,
    ):
        """Plot the curves from individual wells.
        Creates two subplots - top the fit + data, lower - derivative.

        Parameters
        ----------
        output_path
            where to write the file (can be also a dummy value)
        wellID
            from which well to plot
        datatype
            what is being plotted:
                > overview - plot experimental data , fit data and some fit parameters
                > very_raw - a workaround parameter plotting from plate attribute; made for the GUI when the data is loaded, but not processed yet
                > raw - plot unprocessed data
                > normalized - data after preprocessing
                > derivative - first derivative
                > fitted - based on the equation
        save
            actually save the file
        show
            show image instead
        data_ax
            instead of creating new ones, plot in these axes (for PDF reports: disables derivative plot and legend plot)
        vline_legend
            if True then all vlines will be added to the legend (looks bad when individual images are saved)
        """
        if data_ax is None:
            # create the figure object
            fig = plt.figure(1, figsize=(8, 7))
            # create a specification for the relative plot sizes
            gs = gridspec.GridSpec(3, 1, height_ratios=[4, 2, 0.05], figure=fig)
            # get objects of individual subplots
            data_ax = fig.add_subplot(gs[0])  # experimental data, fit, etc
            deriv_ax = fig.add_subplot(gs[1])  # the derivative
        else:
            data_ax = data_ax
            deriv_ax = None

        # NOTE currently all internal manipulations are done in K
        # and conversion back to original scale is not done
        if self.denaturant == "K" or self.denaturant == "C":
            degree_sign = "K"
        else:
            # NOTE this is a placeholder for chemical denaturant scale
            pass

        # a carry-over from the cycle
        i = wellID

        if datatype == "overview":
            # plot the fit
            # plot the experimental data
            # TODO use markevery=n to plot every n-th datapoint
            # format used to be kx, however, for bigger datasets this doesn't look good
            data_ax.plot(
                self.plate[i].index.values,
                self.plate[i],
                "k.",
                mew=1,
                label="Experiment",
            )  # , markevery=40)

            data_ax.plot(
                self.plate[i].index.values,
                self.plate_fit[i],
                label="Fit",
            )

            # a label for Y-axis
            data_ax.set_ylabel(self.readout_type)

            # calculate the offsets for the plot based on the overall length
            max_val = max(self.plate[i].dropna())
            min_val = min(self.plate[i].dropna())
            y_range = max_val - min_val
            data_ax.set_ylim([min_val - 0.1 * y_range, max_val + 0.1 * y_range])

            # force specific range for x-axis
            data_ax.set_xlim(self.xlim)
        else:
            # HACK hide the derivative plot
            if deriv_ax is not None:
                deriv_ax.set_visible(False)
            # set the source dataframe based on the supplied option
            if datatype == "very_raw":
                sourcedf = self.plate
                ylabel = self.readout_type
            elif datatype == "raw":
                sourcedf = self.plate_raw
                ylabel = self.readout_type
            elif datatype == "normalized":
                # after all processing normalized curves are in the plate variable
                sourcedf = self.plate
                ylabel = self.readout_type
            elif datatype == "derivative":
                sourcedf = self.plate_derivative
                ylabel = "dRFU/dT"
            elif datatype == "fitted":
                sourcedf = self.plate_fit
                ylabel = "Fitted RFU"
            else:
                self.print_message("Invalid plotting source requested", "w")

            if datatype == "derivative":
                # for derivative plots plot all values
                plt.plot(sourcedf[i].index.values, sourcedf[i], label=i)
            else:
                data_ax.plot(
                    sourcedf[i].index.values,
                    sourcedf[i],
                    "k.",
                    mew=1,
                    label="Experiment",
                )
            data_ax.set_ylabel(ylabel)

        data_ax.set_xlabel(f"Temperature, {degree_sign}")
        data_ax.set_title("Sample " + str(i), fontsize=12, y=1.05)
        data_ax.grid(True, which="both")

        # commands specific only to overview mode:
        if datatype == "overview":
            # plot the determined baselines
            # create np.poly1d objects with respective fit parameters
            # TODO for line just 2 points are needed...
            poly_pre = np.poly1d(
                *self.plate_results.loc[[i], ["kN_fit", "bN_fit"]].values,
            )
            poly_post = np.poly1d(
                *self.plate_results.loc[[i], ["kU_fit", "bU_fit"]].values,
            )
            data_ax.plot(
                self.plate[i].index.values,
                poly_post(self.plate[i].index.values),
                label="Post- baseline",
                linestyle="--",
            )
            data_ax.plot(
                self.plate[i].index.values,
                poly_pre(self.plate[i].index.values),
                label="Pre- baseline",
                linestyle="--",
            )

            # visualization of the lines requested (specific to different model types)
            # NOTE this would not print the value/stdev on the plot, has to be done separately
            for parameter_name in self.plotlines:
                if vline_legend:
                    data_ax.axvline(
                        self.plate_results[parameter_name][i],
                        ls="dotted",
                        c="b",
                        lw=3,
                        label=parameter_name,
                    )
                else:
                    # NOTE in this case lines are not labeled so that they are not listed in the legend
                    data_ax.axvline(
                        self.plate_results[parameter_name][i],
                        ls="dotted",
                        c="b",
                        lw=3,
                    )
                    # add text with the parameter used to generate the line - doesn't look nice in some caes
                    data_ax.text(
                        self.plate_results[parameter_name][i],
                        data_ax.get_ylim()[0] + 0.05 * y_range,
                        " " + parameter_name,
                        fontsize=12,
                    )
            if deriv_ax is not None:
                fig.legend(loc="lower center", ncol=4, fontsize=12)
                # commands for derivative plot (used only in overview mode)
                deriv_ax.plot(
                    self.plate[i].index.values,
                    self.plate_derivative[i],
                    color="k",
                )
                deriv_ax.set_xlabel(f"Temperature, {degree_sign}")
                deriv_ax.set_ylabel(f"d({self.readout_type})/dT")

                # delete the X-label on the data axes
                data_ax.set_xlabel("")

                # xlim for derivative plot and data plot must be the same!
                deriv_ax.set_xlim(self.xlim)
                deriv_ax.grid(True, which="both")

        if deriv_ax is None:
            # if data_ax was provided externally, then showing/saving should not be done
            return
        if show:
            plt.show()
        elif save:
            plt.savefig(output_path + "/" + str(i) + ".png", dpi=(100))
        plt.close("all")

    ## internal methods for creating HTML report elements
    def html_heatmap(self, heatmap_cmap, display):
        """Returns a div-block with a heatmap of the sortby parameter (the last column in self.plate_results).

        Parameters
        ----------
        heatmap_cmap
            matplotlib colormap for heatmap
        display
            whether the heatmap is shown or not in the final HTML:
                > table (standard view)
                > none (not visible)
                > block (compact view)

        Notes:
        -----
        The heatmap that the user sees when opening the HTML will have display=table, all other will start as display=none
        """
        # this string template corresponds to a single sample entry (has to be wrapped within rows)
        # NOTE when gray cells are clicked a window still pops up but says that there is no such image
        # due to limitations of possible attributes within html, the possible layout info is stored in the
        # title attribute which also gets displayed as a tooltip (and will be also shown in the bottom of the heatmap)
        sample = '        <div class="Cell" onmouseover="mouseOver(this.id, this.title)" title="$CONDITION" id="$ID" onclick="window.open(window.currentHeatmap + \'_resources/$ID.png\',\'Sample $ID\', \'width=450,height=450\')" style="background-color:$COLOR">$ID</div>\n'
        sample = Template(sample)
        # row template
        # TODO add extra spaces for readability of output HTML file
        row = '<div class="Row" id=$ROWNAME>\n$SAMPLES</div>'
        row = Template(row)

        # extract the heatmap from matplotlib
        cmap = plt.get_cmap(heatmap_cmap)

        # by convention the model-specific final sorting parameter is stored in the last column
        # TODO add support creating HTML heatmaps for an arbitrary column
        colors = self.plate_results.iloc[:, -1]
        colors = normalize(colors)

        # convert numbers to HEX colors
        colors = colors.apply(lambda x: rgb2hex(cmap(x)))

        # outer cycle creates rows, inner cycle creates lines for 1-12
        row_output = ""
        for i in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            line = ""
            for j in range(1, 13):
                # if a respective color exists then colorise the <div>
                # if not, then set it to "lightgray"
                sample_id = str(i) + str(j)
                # NOTE modify this line to add additional information (e.g. Tm) to the text under heatmap
                CONDITION = self.layout["Condition"][sample_id]

                if sample_id in colors.index:
                    line = line + sample.substitute(
                        ID=sample_id,
                        COLOR=colors[sample_id],
                        CONDITION=CONDITION,
                    )
                else:
                    # lightgray blends with mid-range of coolwarm, so change to gray
                    line = line + sample.substitute(
                        ID=sample_id,
                        COLOR="gray",
                        CONDITION=CONDITION,
                    )
            row_output = row_output + row.substitute(ROWNAME="Row_" + i, SAMPLES=line)

        # once the heatmap itself is created, it is wrapped around in the additional table
        # that would control if the hm is shown, define the title, etc
        output_template = '<div class="Table" id=$IDENTIFIER style="display:$DISPLAY">\n    <div class="Title">\n        <p>$TITLE_TEXT</p>\n        <i style="font-size:0.8em; font-weight:normal"> Click on the wells to open plots in a separate window </i>\n    </div>\n    $HEATMAP\n</div>'
        output_template = Template(output_template)
        title_text = "{}: heatmap of {} (model <i>{}</i>)".format(
            self.readout_type,
            self.plate_results.iloc[:, -1].name,
            self.model,
        )
        return output_template.substitute(
            DISPLAY=display,
            IDENTIFIER=self.readout_type,
            TITLE_TEXT=title_text,
            HEATMAP=row_output,
        )

    def html_button(self):
        """Returns an HTML button string with the readout name on top."""
        button_template = '<input title="Switch readout to $IDENTIFIER" type="button" onclick="openHeatmap(this.value)" value="$IDENTIFIER" style="float:left" />\n'
        button_template = Template(button_template)
        return button_template.substitute(IDENTIFIER=self.readout_type)

    ## Methods for GUI communication
    def printAnalysisSettings(self):
        """Prints current analysis settings."""
        for setting, def_value in analysis_defaults.items():
            if setting[-2:] == "_h":
                # filter out the help message entries
                pass
            else:
                print(
                    "{} = {} (default: {})".format(
                        setting,
                        getattr(self, setting),
                        def_value,
                    ),
                )
        print("\n")

    def analysisHasBeenDone(self):
        """Check if analysis compleded and self.plate_results is created."""
        return "plate_results" in self.__dict__

    def testWellID(self, wellID, ignore_results=False):
        """Check if a well exists in self.plate_results (return True), otherwise return False.

        Parameters
        ----------
        wellID
            sample ID to check
        ignore_results: bool
            if True, will look for the ID's in plate_raw, even if self.plate_results exists
        """
        # check if analysis was done, and depending on that choose the index to check
        if "plate_results" in self.__dict__:
            index_tocheck = self.plate_results.index
        else:
            index_tocheck = self.plate_raw.columns

        if ignore_results:
            index_tocheck = self.plate_raw.columns

        return wellID in index_tocheck

    def getResultsColumns(self):
        """Returns a tuple of columns from self.plate_results that are relevant for GUI."""
        if "plate_results" in self.__dict__:
            output = [self.plate_results.iloc[:, -1].name, *self.plotlines]

            # BS-factor is more useful than S, but not always available
            if "BS_factor" in self.plate_results.columns:
                output = [*output, "BS_factor"]
            else:
                output = [*output, "S"]
            return output
        else:
            self.print_message("No plate_results attribute found", "w")
            self.print_message("Please perform analysis first", "i")
            return None

    ## Big methods for data input, processing and output
    def SetAnalysisOptions(
        self,
        model=analysis_defaults["model"],
        baseline_fit=analysis_defaults["baseline_fit"],
        baseline_bounds=analysis_defaults["baseline_bounds"],
        dCp=analysis_defaults["dCp"],
        onset_threshold=analysis_defaults["onset_threshold"],
        savgol=analysis_defaults["savgol"],
        # these are pre-processing options (defaults stored in a different dict)
        blanks=prep_defaults["blanks"],  # TODO set in layout instead?
        exclude=prep_defaults["exclude"],  # TODO set in layout instead?
        invert=prep_defaults["invert"],
        mfilt=prep_defaults["mfilt"],
        shrink=prep_defaults["shrink"],
        trim_max=prep_defaults["trim_max"],
        trim_min=prep_defaults["trim_min"],
        # TODO layouts must be handled somewhere else...
        layout=None,
        layout_input_type="csv",
    ):
        """Sets in the MoltenProt instance all analysis-related parameters that will then  be used by methods PrepareData() and ProcessData(). For parameter description see analysis_defaults and prep_defaults dicts.

        References:
        ----------
        https://homepages.inf.ed.ac.uk/rbf/HIPR2/median.htm
        the median filter: how it works and why it may be better than the mean
        """
        self.model = model
        self.baseline_fit = baseline_fit
        self.baseline_bounds = baseline_bounds
        # current value for onset_threshold (used in santoro1988(d) to obtain values of T_onset)
        self.onset_threshold = 0.01
        # the value for savgol window to compute the derivative
        self.savgol = savgol

        self.blanks = blanks
        self.exclude = exclude
        self.invert = invert
        # to avoid confustion, medfilt is the imported method, mfilt is the respective analysis flag
        self.mfilt = mfilt
        self.shrink = shrink
        # NOTE changes made by trim_min/trim_max are saved directly to self.plate
        self.trim_min = trim_min
        self.trim_max = trim_max

        # NOTE to prevent carry-over from previous run (e.g. in JSON) reset the bad fit list
        self.bad_fit = []

        # TESTING setting layout and dCp moved to a separate method
        self.SetLayout(layout=layout, layout_input_type=layout_input_type, dCp=dCp)

    def SetLayout(
        self,
        layout=None,
        layout_input_type="csv",
        dCp=analysis_defaults["dCp"],
    ):
        """Sets layout and dCp."""
        if layout is not None:
            if layout_input_type == "csv":
                try:
                    # the format for layout is more strict: it is always a csv with commas as separators
                    # and index column called ID; more restrictions will follow
                    # TODO add check for the size of layout DataFrame - should be always 96
                    # NOTE it may be better to use ";" as csv separator, because it would be easier to write stuff
                    self.layout = pd.read_csv(layout, index_col="ID", encoding="utf_8")
                except:
                    self.print_message(
                        "Unsupported layout format! No layout info will be available",
                        "w",
                    )
                    self.layout = None
            elif layout_input_type == "from_xlsx":
                self.layout = layout
            # Read blanks and skipped samples from the layout information
            self.blanks = list(
                self.layout[self.layout["Condition"].astype(str) == "Blank"].index,
            )
            self.exclude = list(
                self.layout[self.layout["Condition"].astype(str) == "Ignore"].index,
            )
        else:
            # in all other cases set the instance attribute to None
            self.layout = None

        # heat capacity change values for the whole plate
        # self.dCp can be one of the following: "from_layout" or value specified by the user
        # TODO overwrite layout value instead
        if dCp >= 0:
            # user-set dCp from CLI overrides dCp supplied in the layout
            self.print_message(f"dCp for all samples is set to {dCp}", "i")
            self.dCp = dCp
        elif self.layout is not None:
            # if there is a layout, there may or may not be dCp values
            self.print_message(
                "Using per-sample dCp values as provided in the layout (invalid values will be turned to 0)",
                "i",
            )
            # ensure that dCp values are numbers, but not something else
            # if an invalid value occurs, set it to the default value of 0
            self.layout["dCp"] = pd.to_numeric(self.layout["dCp"], errors="coerce")
            self.layout["dCp"].fillna(0, inplace=True)
            # also negative values must be turned to 0
            self.layout.loc[layout["dCp"] < 0, ["dCp"]] = 0
            self.dCp = "from_layout"
        else:
            msg = f"Incorrect value for dCp ({dCp})!"
            raise ValueError(msg)

    def SetFixedParameters(self, fixed_params):
        """Takes a dataframe with alphanumeric columns (A1-H12) and index being names of
        the parameters (e.g. Tf_fit, Ea_fit) and sets it as the attribute self.fixed_params.

        Notes:
        -----
        No sanity checks of the input are currently done
        """
        self.fixed_params = fixed_params

    def PrepareData(self):
        """Prepares input data for processing."""
        # copy raw data to the main plate
        # NOTE this is primarily needed for json i/o, to ensure that the analysis runs
        # are the same after save/load cycle
        # also the dT step size must be reset
        self.plate = self.plate_raw.copy()
        self.dT = (max(self.plate.index) - min(self.plate.index)) / (
            len(self.plate.index) - 1
        )

        # Remove all-Nan columns
        self.plate = self.plate.dropna(how="all", axis=1)

        # remove the user-specified unneeded wells
        if self.exclude:
            self.plate = self.plate.drop(
                self.plate.columns.intersection(self.exclude),
                axis=1,
            )

        # trim data from the beggining or end
        if self.trim_min:
            self.plate = self.plate[
                self.plate.index >= min(self.plate.index) + self.trim_min
            ]
        if self.trim_max:
            self.plate = self.plate[
                self.plate.index <= max(self.plate.index) - self.trim_max
            ]

        # invert the curves
        if self.invert:
            # reflect the curve relative to the x-axis
            self.plate = self.plate * -1
            # normalize
            self.plate = self.plate.apply(normalize, from_input=True)

        # if blank wells were specified, average them and subtract from the remaining data
        if self.blanks:
            self.print_message("Subtracting background...", "i")
            try:
                bg = self.plate[self.blanks].mean(axis=1)
                # remove (drop) buffer-only columns from the dataset
                self.plate = self.plate.drop(
                    self.plate.columns.intersection(self.blanks),
                    axis=1,
                )
                # subtract the background
                self.plate = self.plate.sub(bg, axis=0)
            except KeyError:
                self.print_message("Same well was supplied as Blank and Ignore", "f")
                print(
                    "Please check parameters for --blank --exclude and the layout file",
                )

        # apply median filter
        if self.mfilt:
            # convert the temperature range to an odd integer window size
            self.mfilt = to_odd(self.mfilt, self.dT)

            # check if the window is bigger than the whole dataset
            if len(self.plate.index) < self.mfilt:
                # NOTE the output may be confusing to the user, because user gives degrees, but gets datapoints
                msg = f"Specified medfilt window size ({self.mfilt} datapoints) is bigger than the dataset length ({len(self.plate.index)} datapoints)!"
                raise ValueError(
                    msg,
                )

            self.plate = self.plate.apply(medfilt, kernel_size=self.mfilt)

        # NOTE this must be done after median filtering (spikes are bad for averaging)
        if self.shrink is not None:
            if self.shrink > self.dT:
                # create the range for Temperature binning (will be also used as the index)
                bin_range = np.arange(
                    np.floor(min(self.plate.index)),
                    np.ceil(max(self.plate.index)) + self.shrink,
                    self.shrink,
                )
                # temporary DataFrame for binning
                self.plate_binned = pd.DataFrame(
                    index=bin_range[:-1],
                    columns=self.plate.columns,
                    dtype="float64",
                )

                # average values using the temperature range (plate_binned index, plate_binned index + bin step)
                for i in self.plate_binned.index:
                    self.plate_binned.loc[i, :] = self.plate[
                        (self.plate.index > i) & (self.plate.index < i + self.shrink)
                    ].mean()

                self.plate = self.plate_binned
                # remove possible empty rows
                self.print_message(
                    f"Input data binned down to {self.shrink} degree step",
                    "i",
                )
                # required for proper derivative computation
                self.dT = self.shrink
            else:
                msg = f"Requested shrinking step ({self.shrink} degrees) is less than the average temperature step ({self.dT} degrees)"
                raise ValueError(
                    msg,
                )

        # compute the derivative
        # NOTE the best way to do it for any type of data - savgol filtering
        # which is not present in older scipy versions. In this case use a less correct approach
        # NOTE it is not smart to save converted values of savgol to self.savgol, rather use internal window_length var
        try:
            from scipy.signal import savgol_filter

            # convert window size in temperature units to an odd integer
            window_length = to_odd(self.savgol, self.dT)
            # check if the window is bigger than the whole dataset
            if len(self.plate.index) < window_length:
                # NOTE the output may be confusing to the user, because user gives degrees, but gets datapoints
                msg = f"Specified savgol window size ({window_length} datapoints) is bigger than the dataset length ({len(self.plate.index)} datapoints)!"
                raise ValueError(
                    msg,
                )

            # NOTE an additional check for savgol requirements (4 is polyorder used by default)
            if window_length < 4:
                msg = f"Specified savgol window size ({window_length} datapoints) is smaller than the polynomial order of the filter (4); increase savgol window size, or do not shrink the data"
                raise ValueError(
                    msg,
                )
            """
            NOTE default mode for savgol is interp, which just uses polynomial fit of the last window to
            create the data values beyond the end of the dataset. This may create tails in the derivative
            that interfere with peak detection. "nearest" mode just takes the first value and uses it
            for the missing parts of the window at data edges; see here for other modes:
            https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.savgol_filter.html
            """
            self.plate_derivative = self.plate.apply(
                savgol_filter,
                window_length=window_length,
                polyorder=4,
                deriv=1,
                mode="nearest",
            )
        except:
            """
            NOTE usually, this exception would happen with old scipy version
            However, this also happened in pyinstaller bundle for Windows:
            LinAlgError: SVD did not converge in Linear Least Squares
            The derivate is not a crucial element (fits usually converge even with a random Tm guess),
            so it should be OK to proceed.
            """
            self.print_message(
                "Cannot compute a smoothened derivative, fit results can be suboptimal",
                "w",
            )
            self.print_message("Falling back to the difference method", "i")
            self.plate_derivative = (self.plate - self.plate.shift(periods=1)) / self.dT

    def ProcessData(self):
        """Performs curve fitting and creates results dataframes.

        Notes:
        -----
        The names for the parameters and the contents of plate_results variable are different for various methods, so they are set up based on the selected analysis
        """
        if self.model == "skip":
            self.print_message("Dataset was omitted from analysis", "i")
            # check if there are plate_results attributes from previous analysis and delete them
            # NOTE there are more leftover attributes which will be saved to JSON, but without plate_results the dataset should not be recognized as a processed one
            if self.analysisHasBeenDone():
                del self.plate_results
                del self.plate_results_stdev
            # return statement allows to silently terminate the ProcessData method
            return

        # check if model name is correct
        if self.model not in avail_models.keys():
            msg = f"Unknown model '{self.model}'"
            raise ValueError(msg)

        model = avail_models[self.model](scan_rate=self.scan_rate)

        self.print_message("Processing data...", "i")

        # Set here scan rate for the model
        # None - not specified (e.g. for CSV), doesn't matter for equilibrium and empirical models; can be set in CLI
        # XLSX - parsed from Excel and always available
        # kinetic models should raise a ValueError
        model.scan_rate = self.scan_rate

        # the function to do curve fitting
        f = model.fun
        # generate parameter names
        result_index = []
        for i in model.param_names():
            result_index.append(i + "_init")
        for i in model.param_names():
            result_index.append(i + "_fit")
        result_index.append("S")

        # select dataframe to be fit (e.g. can be changed to plate_derivative)
        df_for_fitting = self.plate
        # create dataframe to store results and populate with initial parameter values
        # NOTE by default empty dataframes get "object" data type, which can cause problems when using numpy
        # if the datatype is specified explicitly, the problem should not exist
        self.plate_results = pd.DataFrame(
            index=result_index,
            columns=self.plate.columns.values,
            dtype="float64",
        )

        # create a dataframe to store stdev for fit parameters
        # for simplicity the dataframe will also have initial parameters included
        # they will be dropped in the end of processing
        self.plate_results_stdev = pd.DataFrame(
            index=result_index,
            columns=self.plate.columns.values,
            dtype="float64",
        )

        # an empty p0 variable
        p0 = []
        # run a cycle through all columns and calculate fits
        self.print_message("Fitting curves...", "i")

        for i in df_for_fitting.columns.values:
            # drop Nan values to prevent crashes of fitting
            data = df_for_fitting[i].dropna()
            T = np.float64(data.index.values)
            # guess initial parameters
            p0 = model.param_init(data)
            self.plate_results[i][0 : len(p0)] = p0
            # get parameter bounds
            param_bounds = model.param_bounds(data)
            # if the model has these parameters, then run a more precise initial parameter estimation
            if {"kN", "bN", "kU", "bU"}.issubset(model.param_names()):
                # furthermore, if there is a Tm than run smart Tm pre-estimation
                estimate_Tm = "Tm" in model.param_names()
                self._estimate_baseline(
                    data,
                    fit_length=self.baseline_fit,
                    estimate_Tm=estimate_Tm,
                )

                # Set bounds for kN, bN, kU, bU to be a multiple of stdev of baseline-prefitting
                # NOTE this assumes that the parameters are the first four in the list of bounds/p0
                # originally param_bounds is a tuple, so it has to be converted to a list...
                if self.baseline_bounds > 0:
                    param_bounds = list(param_bounds)
                    param_bounds[0] = list(param_bounds[0])
                    param_bounds[1] = list(param_bounds[1])
                    param_bounds[0][:4] = list(
                        self.plate_results[i][:4]
                        - self.plate_results_stdev[i][:4] * self.baseline_bounds,
                    )
                    param_bounds[1][:4] = list(
                        self.plate_results[i][:4]
                        + self.plate_results_stdev[i][:4] * self.baseline_bounds,
                    )
                elif self.baseline_bounds < 0:
                    msg = f"Expected a non-negative int for baseline_bounds, but got {self.baseline_bounds}"
                    raise ValueError(
                        msg,
                    )

            # if this MoltenProtFit instance has any fixed parameter dataframe,
            # they will be supplied to the model
            if self.fixed_params is not None:
                model.set_fixed(list(self.fixed_params[i]))
            try:
                # for some reason, pandas Series for initial parameters have to be converted to a list
                # the fit gives two arrays: fit parameters (p) and covariance matrix (covm for stdev estimation)

                # depending on scipy version enforce the parameter limits or not
                if (LooseVersion(scipy_version) >= LooseVersion("0.17")) and (
                    param_bounds is not None
                ):
                    # NOTE adding ftol=0.01 and xtol=0.01 may in some cases speed up the fitting (loosens the convergence criteria)
                    # default values 1e-8 are a bit too conservative; in preliminary tests the speedup was marginal
                    p, covm = curve_fit(
                        f,
                        T,
                        data,
                        list(self.plate_results[i][0 : len(p0)]),
                        bounds=param_bounds,
                    )
                else:
                    p, covm = curve_fit(
                        f,
                        T,
                        data.values,
                        list(self.plate_results[i][0 : len(p0) + 1]),
                    )
                self.plate_results[i][len(p0) : (len(p0)) * 2] = p
                # this is the official way to compute stdev error (from scipy docs)
                self.plate_results_stdev[i][len(p0) : (len(p0)) * 2] = np.sqrt(
                    np.diagonal(covm),
                )
            except RuntimeError:
                # these probably correspond to bad fitting
                self.print_message(
                    f"Curve fit for {i} failed, probably invalid transition.",
                    "w",
                )
                # generate a list of bad fits
                self.bad_fit.append(i)
            except ValueError as e:
                self.print_message(e.__str__(), "w")
                # catch some problems of fitting with santoro1988d
                self.print_message(
                    f"Curve fit for {i} failed unexpectedly (ValueError)",
                    "w",
                )
                # generate a list of bad fits
                self.bad_fit.append(i)

            # supply sqrt(n) for S calculation
            # to calculate RMSE we don't care about the amount of parameters,
            # however, they must be included for standrard error of estimate
            # (more info here: http://people.duke.edu/~rnau/compare.htm)
            try:
                self.plate_results[i].loc["S"] = np.sqrt(len(T) - len(p0))
            except TypeError:
                self.print_message(
                    f"Curve for sample {i} has just one value!",
                    "w",
                )
                # add to bad samples list
                self.bad_fit.append(i)

        # drop the wells that could not be fit
        self.plate_results.drop(self.bad_fit, axis=1, inplace=True)
        self.plate_results_stdev.drop(self.bad_fit, axis=1, inplace=True)

        # empty DataFrame with temperature index to store computed fitted curves
        self.plate_fit = pd.DataFrame(index=self.plate.index)

        def calculate_fit(input_series):
            """Helper function to compute fit curves
            input DataFrame is self.plate_results.
            """
            # use the fit parameters from plate_results and the index of plate_fit to compute fit curves
            self.plate_fit = pd.concat(
                [
                    self.plate_fit,
                    pd.Series(
                        f(
                            self.plate_fit.index,
                            *list(input_series[len(p0) : (len(p0)) * 2]),
                        ),
                        index=self.plate_fit.index,
                        name=input_series.name,
                    ),
                ],
                axis=1,
            )

        # apply the helper function
        self.plate_results.apply(calculate_fit)

        # calculate S and put it into plate_results
        self.print_message("Estimating S...\n", "i")

        # compute S using df_for_fitting
        plate_S = (df_for_fitting - self.plate_fit) ** 2
        plate_S = np.sqrt(plate_S.sum())
        self.plate_results.loc["S", :] = plate_S / self.plate_results.loc["S", :]

        # transpose the DataFrame to have sample wells in rows
        self.plate_results = self.plate_results.T
        self.plate_results_stdev = self.plate_results_stdev.T

        # compute BS-factor to assess how wide is the signal window relative to the noise
        # this requires knowledge of baseline parameters and Tm
        if {"kN", "bN", "kU", "bU", "Tm"}.issubset(model.param_names()):
            # Use S as an overall proxy for assay noise
            self.plate_results["BS_factor"] = 1 - 6 * self.plate_results.S / abs(
                self.plate_results.kU_fit * self.plate_results.Tm_fit
                + self.plate_results.bU_fit
                - (
                    self.plate_results.kN_fit * self.plate_results.Tm_fit
                    + self.plate_results.bN_fit
                ),
            )

        # compute baseline-corrected raw data if standard baseline parameters are available
        if {"kN", "bN", "kU", "bU"}.issubset(model.param_names()):
            self._calculate_raw_corr()

        # add layout information to the results
        if self.layout is not None:
            self.plate_results = pd.concat(
                [self.layout, self.plate_results],
                join="inner",
                axis=1,
            )

        # remove *_init columns and S from self.plate_results_stdev
        self.plate_results_stdev.drop(
            labels=result_index[: len(p0)],
            axis=1,
            inplace=True,
        )
        self.plate_results_stdev.drop(labels="S", axis=1, inplace=True)

        # based on the model, calculate additional curve characteristics and set the vertical lines for plots
        if model.sortby == "dG_std":
            # Calculate T_onset
            self._calc_Tons("Tm_fit", "dHm_fit", self.onset_threshold)
            # dG and dCp component
            self.CalculateThermodynamic()
            # list of vertical lines to be plotted
            self.plotlines = ["T_onset", "Tm_fit"]
        elif model.sortby == "dG_comb_std":
            # Calculate T_onset
            self._calc_Tons("T1_fit", "dHm1_fit", self.onset_threshold)
            # Calculate T2 from dT2_1_fit
            # TODO error propagation
            self.plate_results["T2_fit"] = (
                self.plate_results["T1_fit"] + self.plate_results["dT2_1_fit"]
            )
            # NOTE dCp is hard to determine for the intermediate, so it is completely neglected
            # dG is calculated for each reaction (N<->I and I<->U) and then combined following the principle of thermodynamic coupling
            self.plate_results["dG_comb_std"] = self.plate_results["dHm1_fit"] * (
                1 - T_std / self.plate_results["T1_fit"]
            ) + self.plate_results["dHm2_fit"] * (
                1 - T_std / self.plate_results["T2_fit"]
            )
            # list of vertical lines to be plotted
            self.plotlines = ["T_onset", "T1_fit", "T2_fit"]
        elif model.sortby == "T_eucl":
            # computes Euclidean distance for Tm/T_onset
            # Tm and T_ons are on the same scale and are orthogonal characteristics of the sigmoidal curve
            # thus, a sample with the most optimal combination is the one that is most far away from T=0
            self.plate_results["T_eucl"] = np.sqrt(
                self.plate_results["Tm_fit"] ** 2
                + self.plate_results["T_onset_fit"] ** 2,
            )
            # list of vertical lines to be plotted
            self.plotlines = ["T_onset_fit", "Tm_fit"]
        elif model.sortby == "T_eucl_comb":
            self.plate_results["T_eucl_comb"] = np.sqrt(
                self.plate_results["T1_fit"] ** 2
                + self.plate_results["T_onset1_fit"] ** 2,
            ) + np.sqrt(
                self.plate_results["T2_fit"] ** 2
                + self.plate_results["T_onset2_fit"] ** 2,
            )
            self.plotlines = [
                "T_onset1_fit",
                "T1_fit",
                "T_onset2_fit",
                "T2_fit",
            ]
        elif model.sortby == "pk_ratio_std":
            # the kF/kR at std temperature; take as -log10 to have higher values for higher stability
            self.plate_results["pk_ratio_std"] = -np.log10(
                model.arrhenius(
                    T_std,
                    self.plate_results.TfF_fit,
                    self.plate_results.EaF_fit,
                )
                / model.arrhenius(
                    T_std,
                    self.plate_results.TfR_fit,
                    self.plate_results.EaR_fit,
                ),
            )
            self.plotlines = ["TfF_fit", "TfR_fit"]
        elif model.sortby == "pk_std":
            # For irreversible reactions calculate the unfolding rate constant at std temperature
            # based on Tf and Ea values
            # NOTE by convention higher sorting value indicates higher stability, but it is not the case
            # for rate constant of reaction N -> U; a common trick in chemistry is to calculate the
            # negative log10
            self.plate_results["pk_std"] = -np.log10(
                model.arrhenius(
                    T_std,
                    self.plate_results.Tf_fit,
                    self.plate_results.Ea_fit,
                ),
            )
            self.plotlines = ["Tf_fit"]
        else:
            self.print_message(
                "Model {} contains no measure for final sorting".format(
                    model.short_name,
                ),
                "w",
            )
            self.plotlines = []

        # sort_values based on the model's sortby property
        if model.sortby is not None:
            self.plate_results.sort_values(
                by=model.sortby,
                inplace=True,
                ascending=False,
            )

        # for proper json i/o
        self.plate_results.index.name = "ID"
        self.plate_results_stdev.index.name = "ID"

    def CalculateThermodynamic(self):
        """Calculates some thermodynamic characteristics of data and append them to self.plate_results:
        > dG_std - Gibbs free energy of unfolding at standard temperature (298 K), extrapolated using the values of dCp
        > dCp - heat capacity change of unfolding (supplied by user either in the layout or in the command line)
        > dHm - enthalpy of unfolding at Tm (calculated during fitting).
        """
        # create temporary dataframe to store td results
        column_names = ["dCp", "dG_std"]
        # NOTE by default empty dataframes get "object" data type, which can cause problems when using numpy
        # if the datatype is specified explicitly, the problem should not exist
        # NOTE since dCp is set externally, it will be just copy-pasted and no stdev can be computed
        td_results = pd.DataFrame(
            index=self.plate_results.index,
            columns=column_names,
            dtype="float64",
        )
        # create a separate dataframe for storing stdev
        td_results_stdev = pd.DataFrame(
            index=self.plate_results.index,
            columns=[column_names[1]],
            dtype="float64",
        )

        if self.dCp == "from_layout":
            td_results["dCp"] = self.layout["dCp"]
        else:
            td_results["dCp"] = self.dCp

        # issue a warning about inaccuracy of extrapolated dG with dCp=0
        if any(td_results["dCp"] == 0):
            self.print_message(
                "One or more dCp values are set to 0; this results in an overestimate of dG_std",
                "w",
            )

        # dG_std is extrapolated to standard temperature using the model described in Becktel and Schellman, 1987
        # Tm is chosen as the reference temperature for equation (4), which also means that dS(Tm) = dH(Tm)/Tm
        td_results["dG_std"] = self.plate_results["dHm_fit"] * (
            1 - T_std / self.plate_results["Tm_fit"]
        ) - td_results["dCp"] * (
            self.plate_results["Tm_fit"]
            - T_std
            + T_std * np.log(T_std / self.plate_results["Tm_fit"])
        )
        td_results_stdev["dG_std"] = np.sqrt(
            self.plate_results_stdev["dHm_fit"] ** 2
            + (self.plate_results["dHm_fit"] * T_std / self.plate_results["Tm_fit"])
            ** 2
            * (
                (self.plate_results_stdev["dHm_fit"] / self.plate_results["dHm_fit"])
                ** 2
                + (self.plate_results_stdev["Tm_fit"] / self.plate_results["Tm_fit"])
                ** 2
            )
            + (td_results["dCp"] * T_std) ** 2
            * (
                (self.plate_results_stdev["Tm_fit"] / T_std) ** 2
                + (self.plate_results_stdev["Tm_fit"] / self.plate_results["Tm_fit"])
                ** 2
            ),
        )

        # calculate contribution of dCp to the real dG
        td_results["dCp_component"] = (
            T_std
            - self.plate_results["Tm_fit"]
            - T_std * np.log(T_std / self.plate_results["Tm_fit"])
        )
        td_results_stdev["dCp_component"] = np.sqrt(
            (self.plate_results_stdev["Tm_fit"]) ** 2
            + (
                T_std
                * self.plate_results_stdev["Tm_fit"]
                / self.plate_results["Tm_fit"]
            )
            ** 2,
        )

        # append td_results to self.plate_results and self.plate_results_stdev
        self.plate_results = pd.concat(
            [self.plate_results, td_results.loc[:, ["dCp_component", "dG_std"]]],
            axis=1,
            sort=True,
        )

        self.plate_results_stdev = pd.concat(
            [self.plate_results_stdev, td_results_stdev],
            axis=1,
            sort=True,
        )

    def CombineResults(self, tm_stdev_filt, bs_filt, merge_dup, tm_key):
        """A helper method to filter results and optionally average duplicates.

        Parameters
        ----------
        tm_stdev_filt
            samples with stdev for Tm above this value will be discarded
        tm_key
            the name for column with Tm
        bs_filt
            samples with BS-factor above this value will be kept
        merge_dup
            whether to merge duplicates (based on annotations in the layout)

        Returns:
        -------
        results - joined plate_results/stdev DataFrame with optional filtering and duplicate averaging

        Notes:
        -----
        * not all plate_results can contain BS-factor, so this part of filtering will be skipped
        """
        # NOTE to prevent irreversible changes to plate_results* attributes, force a copy
        results = self.plate_results.copy()

        # we may have 2 types of stdev, one from fitting and another one from duplicate averaging
        stdev_fit = self.plate_results_stdev.copy()

        # add suffix _stdev to prevent duplicate column names in joined df's
        stdev_fit.columns = stdev_fit.columns + "_stdev"

        # NOTE previously, merge_dup was run _before_ stdev filtering, but this messes up
        # subsequent description of the procedure. Also, this operation joins 2 processes
        # (removal of bad fits and removal of invalid dups) in one, which is bad
        # a better way: first filter to remove crappy fits
        # then merge dups and use stdev that comes from them.
        # This approach also obsoletes the methods for stdev "joining"

        # use stdev_fit dataframe to get ID's of samples that have improper values
        if tm_stdev_filt > 0:
            drop_tm_stdev = stdev_fit[tm_key + "_fit_stdev"][
                stdev_fit[tm_key + "_fit_stdev"] > tm_stdev_filt
            ].index
            results = results.drop(drop_tm_stdev, axis=0)
        if bs_filt > 0:
            try:
                results = results[results["BS_factor"] >= bs_filt]
            except KeyError:
                self.print_message(
                    "Column 'BS_factor' not found in the results DataFrame, cannot filter",
                    "w",
                )

        if merge_dup:
            # TODO add an option for discarding samples that do not have a duplicate
            # averaging capillary numbers doesn't make sense, convert them to strings
            if "Capillary" in results.columns:
                results.Capillary = results.Capillary.apply(str)

            # processing steps: remove duplicates with different aggregation functions:
            # - text data (ID, Capillary) - create a comma-separated string of values
            # - result data - compute mean, stdev and n (amount of duplicated values)

            # prepare DataFrames:
            # split result data to text and numeric
            text_data = results.select_dtypes(include=["object"])
            numeric_data = results.select_dtypes(include=["float64", "int64"])

            # for text data convert ID index to a new column (a bit hacky)
            text_data.reset_index(inplace=True)
            text_data.set_index("ID", inplace=True, drop=False)

            # since the column for de-duplication is text, it has to be re-created for numeric_data and stdev_data
            numeric_data = pd.concat([numeric_data, text_data["Condition"]], axis=1)

            # perform deduplication (grouping)

            # for text data we declare a lambda function
            text_data = text_data.groupby("Condition").agg(lambda x: ",".join(list(x)))

            # results data - count of each sample
            numeric_data_count = numeric_data.groupby("Condition").agg("count")
            # now also stdev (add a proper suffix to index)
            numeric_data_stdev = numeric_data.groupby("Condition").agg(np.std)
            numeric_data_stdev.columns = numeric_data_stdev.columns + "_stdev"
            # and finally average
            numeric_data = numeric_data.groupby("Condition").agg(np.mean)

            # add n information to the text_data
            # BUG if after filtering the dataset is empty, this operation produces an error
            # this is caught with try/except
            try:
                text_data["n"] = numeric_data_count.iloc[:, [0]]
            except ValueError as e:
                self.print_message(e.__str__(), "w")
                self.print_message(
                    "No data left after filtering in CombineResults()",
                    "w",
                )

            # update the results dataframe
            results = pd.concat([text_data, numeric_data, numeric_data_stdev], axis=1)
        else:
            results = pd.concat([results, stdev_fit], axis=1)
        return results

    def RenameResults(self, datatype="sct"):
        """#HACK rename Tm to Tagg in for scattering data
        # also chemical denaturation renaming can be done here.
        """
        if datatype == "sct":
            rename_dict = {"Tm_init": "Tagg_init", "Tm_fit": "Tagg_fit"}
        elif datatype == "chem":
            rename_dict = {"Tm_init": "dGH2O_init", "Tm_fit": "dGH2O_fit"}
            msg = "Chemical denaturation renaming is not there yet"
            raise NotImplementedError(msg)

        self.plate_results.rename(columns=rename_dict, inplace=True)
        self.plate_results_stdev.rename(columns=rename_dict, inplace=True)

    @staticmethod
    def _trim_string(string, length=30, symmetric=True):
        """Helper method to trim a string to a specific length by removing the middle part.

        Arguments:
        ---------
        string
            string to be trimmed
        length
            length of the data to keep
        symmetric
            if true, also show the end of the string
        """
        # ensure that input is a str
        string = str(string)
        # if too short, return as is
        if len(string) <= length:
            return string
        if symmetric:
            return string[: length // 2] + " ... " + string[-length // 2 :]
        else:
            return string[:length] + "..."

    def _plotfig_pdf(self, samples, failed=False):
        """A helper method for smart packing of individual sample plots into figures.

        Parameters
        ----------
        samples
            a valid list of samples to be plotted
        failed
            indicate if samples are from failed fits

        Returns:
        -------
        A list of Figure objects to be added to pages list
        """
        if failed:
            suptitle = "Excluded/failed samples"
            datatype = "raw"
        else:
            suptitle = "Successful fits"
            datatype = "overview"
        pages = []
        n_results = len(samples)
        # intialize figures, axes, and plot counter
        plot_fig, plot_axs = plt.subplots(
            4,
            3,
            sharex=False,
            sharey=False,
            figsize=(8.3, 11.7),
        )
        plot_fig.suptitle(suptitle, fontweight="bold", fontsize="x-large")
        plot_axs = list(plot_axs.flat)
        plot_counter = 0
        for sample in samples:
            if plot_counter < 11:
                self.plotfig(
                    "dummy_output",
                    sample,
                    datatype=datatype,
                    save=False,
                    show=False,
                    data_ax=plot_axs[plot_counter],
                    vline_legend=True,
                )
                plot_counter += 1
            # if the 11th plot was plotted, append old figure and initialize a new one
            # this should be also triggered if less than 11 is plotted
            if (plot_counter == 11) or (plot_counter == n_results):
                # make legend in the next axes
                plot_axs[plot_counter].legend(
                    handles=plot_axs[plot_counter - 1].get_lines(),
                    mode="expand",
                    ncol=1,
                )
                # hide remaining unused axes
                while plot_counter <= 11:
                    plot_axs[plot_counter].set_axis_off()
                    plot_counter += 1
                plot_fig.tight_layout(
                    rect=(0.02, 0.05, 0.98, 0.95),
                )  # rect leaves some margins empty
                pages.append(plot_fig)
                plot_fig, plot_axs = plt.subplots(
                    4,
                    3,
                    sharex=False,
                    sharey=False,
                    figsize=(8.3, 11.7),
                )
                plot_fig.suptitle(
                    suptitle + " (continued)",
                    fontweight="bold",
                    fontsize="x-large",
                )
                plot_axs = list(plot_axs.flat)
                plot_counter = 0
        return pages

    def PdfReport(self, outfile):
        """Generate and write a multi-page PDF report.

        Parameters
        ----------
        outfile
            location of the output, overwrite without confirmation

        Notes:
        -----
        * heatmap and converter96 are ancient methods, so they are just carefully wrapped around
        * each page is a figure object
        * multi-page pdf as per mpl [docs](https://matplotlib.org/stable/gallery/misc/multipage_pdf.html)
        """
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.table import table as mpl_table

        result_columns = (
            self.getResultsColumns()
        )  # the first value should be the recommended sorting parameter followed by vlines and BS/S
        # add condition column
        result_columns = ["Condition", *result_columns]
        sort_parameter = result_columns[1]
        # preprocess the result df
        result_table = self.plate_results.loc[
            :,
            result_columns,
        ].copy()  # to prevent edits to the original DF
        result_table = np.round(result_table, 2)  # round the numeric data
        result_table["Condition"] = result_table["Condition"].apply(
            self._trim_string,
            length=10,
            symmetric=False,
        )
        result_table_colors = result_table.copy()  # color table 1.0 is white 0 is black
        result_table_colors.loc[:, :] = "1.0"
        result_table_colors.iloc[::2] = "0.75"
        result_table_colors = result_table_colors.values
        result_index = result_table.index
        result_table = result_table.values  # convert to a list of lists
        n_results = len(result_table)  # total number of results
        pages = []

        ## Page 1: Heatmap of the respective sortby parameter, top 15 results, run info
        plate96 = self.converter96(sort_parameter, reference=None)
        page1, page1_ax = self.heatmap(
            "dummy_output",
            plate96,
            sort_parameter,
            save=False,
            pdf_report=True,
        )
        page1_ax[0].set_title(
            f"Heatmap of {sort_parameter}",
            loc="left",
            fontweight="bold",
        )
        # mpl tables cannot do word wrapping, so trim the file name
        filename = self._trim_string(self.filename)
        mp_version = __version__
        if from_pyinstaller:
            mp_version += " (PyInstaller bundle)"
        timestamp = strftime("%c")
        # failed fits and user-excluded samples
        excluded = self._get_failed_samples()
        if len(excluded) > 0:
            excluded_str = self._trim_string(", ".join(list(excluded)))
        else:
            excluded_str = "None"

        info_table = [
            ["Timestamp", timestamp],
            ["Input file", filename],
            ["Scan rate, degrees/min", self.scan_rate],
            ["MoltenProt version", mp_version],
            ["Analysis model", self.model],
            ["Excluded/failed samples", excluded_str],
        ]
        info_table_ax = page1_ax[2]
        info_table = mpl_table(
            info_table_ax,
            info_table,
            loc="upper left",
            edges="open",
            cellLoc="left",
        )

        # using the solution from here to set proper font size in the table:
        # https://stackoverflow.com/questions/15514005/how-to-change-the-tables-fontsize-with-matplotlib-pyplot
        info_table.auto_set_font_size(False)
        info_table_ax.set_title("Run info", loc="left", fontweight="bold")
        top10_table_ax = page1_ax[1]
        mpl_table(
            top10_table_ax,
            result_table[:15, :],
            loc="upper left",
            colLabels=result_columns,
            cellLoc="left",
            rowLabels=result_index[:15],
            cellColours=result_table_colors[:15, :],
        )
        top10_table_ax.set_title("Top 15 results", loc="left", fontweight="bold")

        info_table_ax.set_axis_off()
        top10_table_ax.set_axis_off()

        # finalize page 1
        page1.suptitle(
            f"MoltenProt Report: {self.readout_type}",
            fontweight="bold",
            fontsize="x-large",
        )
        # add citation to the bottom of the page
        page1.text(0.5, 0.05, citation["short"], ha="center")  # fontstyle='italic',
        pages.append(page1)

        ## Full result table - create if more than 15 results (but less than 48)
        if n_results > 15:
            page2, page2_ax = plt.subplots(1, 1, figsize=(8.3, 11.7))
            mpl_table(
                page2_ax,
                result_table[:48],
                loc="upper left",
                colLabels=result_columns,
                cellLoc="left",
                rowLabels=result_index[:48],
                cellColours=result_table_colors[:48],
            )
            page2_ax.set_axis_off()
            page2_ax.set_title(
                f"Result table (sorted by {sort_parameter})",
                loc="left",
                fontweight="bold",
            )
            pages.append(page2)

        # if more than 48 samples are present
        if n_results > 48:
            page3, page3_ax = plt.subplots(1, 1, figsize=(8.3, 11.7))
            mpl_table(
                page3_ax,
                result_table[48:],
                loc="upper left",
                colLabels=result_columns,
                cellLoc="left",
                rowLabels=result_index[48:],
                cellColours=result_table_colors[48:],
            )
            page3_ax.set_axis_off()
            page3_ax.set_title(
                "Result table (continued)",
                loc="left",
                fontweight="bold",
            )
            pages.append(page3)

        ## pages with plots of individual curves
        pages += self._plotfig_pdf(self.plate_results.index)
        ## same as above, but for failed fits
        pages += self._plotfig_pdf(excluded, failed=True)

        # write output
        page_no = 1
        page_count = len(pages)
        with PdfPages(outfile) as pdf_file:
            for i in pages:
                # add page number and save
                i.text(
                    0.5,
                    0.025,
                    f"-- Page {page_no} of {page_count} --",
                    fontstyle="italic",
                    ha="center",
                )
                page_no += 1
                pdf_file.savefig(i)

        # clean up mpl objects
        plt.close("all")

    def _get_failed_samples(self):
        """Return a list of samples that were either excluded or not fit.

        Notes:
        -----
        * will raise a value error if the analysis is not done
        """
        return self.plate_raw.columns.difference(self.plate_results.index)

    def WriteOutput(
        self,
        print10=False,
        xlsx=False,
        genpics=False,
        heatmaps=[],
        hm_ref=None,
        heatmap_cmap=defaults["heatmap_cmap"],
        resources_prefix="",
        n_jobs=1,
        no_data=False,
        pdf=False,
    ):
        """Write the results to the disk.

        Parameters
        ----------
        print10
            print 10 best samples to stdout
        xlsx
            write output in XLSX rather than CSV
        genpics
            generate figures for all samples
        heatmaps
            generate heatmaps for selected plate_results columns
        hm_ref
            reference well for heatmap
        heatmap_cmap
            matplotlib colormap for heatmap
        resources_prefix
            extra string to add when saving files (e.g. to prevent overwrites in multi-dataset case)
        n_jobs
            number of subprocesses to run figure plotting in parallel
        no_data
            do not output any data
        pdf
            write a report in PDF format
        """
        # estimate how many samples could not go through the processing by finding the
        # difference between the indexes of original dataset and the final results
        # BUG if the sample is registered in xlsx annotations, but not present in the raw data plate
        # (i.e. it was marked as bad capillary by Prometheus), then it will not be shown here
        # such samples can be easily diagnosed through reports: even the raw-only signal is not plotted
        # BUG failed samples have only ID's, but not layout info
        failed_samples = self._get_failed_samples()

        # if there is a least one empty sample create respective results dataframes for it
        # and temporarily append them to the main results (will not persist as a class attribute)
        # NOTE index objects are immutable, so a new empty df has to be created and concatenated
        if len(failed_samples) > 0:
            failed_results = pd.DataFrame(
                index=failed_samples,
                columns=self.plate_results.columns,
            )
            failed_results_stdev = pd.DataFrame(
                index=failed_samples,
                columns=self.plate_results_stdev.columns,
            )
            output_results = pd.concat([self.plate_results, failed_results])
            output_results_stdev = pd.concat(
                [self.plate_results_stdev, failed_results_stdev],
            )
        else:
            output_results = self.plate_results
            output_results_stdev = self.plate_results_stdev

        output_path = self.resultfolder

        if print10:
            # print the 10 best conditions to the console
            self.print_message("\n\nBest 10 conditions (best->worst):", "i")
            print(", ".join(list(self.plate_results[:10].index)) + "\n")

        # the checks for the folder existence are done in the main script
        self.print_message("Writing results...", "i")
        if not no_data:
            if xlsx:
                # create a holding object for *.xlsx export
                writer = pd.ExcelWriter(
                    os.path.join(output_path, resources_prefix + "_Results.xlsx"),
                )

                self.plate_raw.to_excel(writer, "Raw data")
                self.plate.to_excel(writer, "Preprocessed data")
                self.plate_fit.to_excel(writer, "Fit curves")
                self.plate_raw_corr.to_excel(writer, "Baseline-corrected")
                output_results.to_excel(writer, "Fit parameters")
                output_results_stdev.to_excel(writer, "Standard deviations")

                # this is the longest part in XLSX saving procedure
                writer.save()
            else:
                # convert plate_results* dataframes to *.csv's
                output_results.to_csv(
                    os.path.join(output_path, resources_prefix + "_results.csv"),
                    sep=",",
                    index_label="Parameters",
                    encoding="utf-8",
                )
                output_results_stdev.to_csv(
                    os.path.join(output_path, resources_prefix + "_results_stdev.csv"),
                    sep=",",
                    index_label="Parameters",
                    encoding="utf-8",
                )
                self.plate_fit.to_csv(
                    os.path.join(output_path, resources_prefix + "_fit.csv"),
                    sep=",",
                    encoding="utf-8",
                )
                self.plate.to_csv(
                    os.path.join(output_path, resources_prefix + "_preproc_curves.csv"),
                    sep=",",
                    encoding="utf-8",
                )
                self.plate_raw_corr.to_csv(
                    os.path.join(output_path, resources_prefix + "_raw_corr.csv"),
                    sep=",",
                    encoding="utf-8",
                )
        # PDF report
        if pdf:
            self.PdfReport(os.path.join(output_path, resources_prefix + "_report.pdf"))

        # generate heatmaps
        if len(heatmaps) > 0:
            if "all" in heatmaps:
                # columns shared (presumably) between all models
                heatmaps = ["S", *self.plotlines]
            for i in heatmaps:
                try:
                    plate96 = self.converter96(i, hm_ref)
                    # check if the column name is in the hm_dic, if not use default options
                    try:
                        self.heatmap(
                            output_path,
                            plate96,
                            i,
                            lowerbetter=self.hm_dic[i]["lowerbetter"],
                            title=self.hm_dic[i]["title"],
                            tick_labels=self.hm_dic[i]["tick_labels"],
                            heatmap_cmap=heatmap_cmap,
                        )
                    except KeyError:
                        try:
                            # print that lowerbetter for this heatmap can be wrong
                            self.print_message(
                                "{} was not found in hm_dic, the colors may be wrong!".format(
                                    i,
                                ),
                                "w",
                            )
                            self.heatmap(
                                output_path,
                                plate96,
                                i,
                                title="Heatmap of " + str(i),
                            )
                        except:
                            self.print_message(
                                "Re-check the parameters for --heatmap option",
                                "w",
                            )
                except ValueError:
                    # NOTE a temporary fix for cases when a wrong heatmap parameter was supplied
                    self.print_message(
                        f'"{i}" is an invalid option for heatmap',
                        "w",
                    )
                    print("Valid options are:")
                    print(", ".join(list(self.plate_results.columns)))

            # clean up after plotting so that no parameters are carried over to downstream plottings
            plt.close("all")

        # generate plots of individual samples
        # BUG for multi-dataset instance using --genfigs will overwrite images
        if genpics:
            self.print_message("Generating figures... This may take a while...", "i")
            # depending on the value of parallelization either run a single or multi-processor command
            if parallelization and n_jobs > 1:
                # with picklable plotfig method:
                Parallel(n_jobs=n_jobs)(
                    delayed(self.plotfig)(output_path=output_path, wellID=i)
                    for i in self.plate_fit.columns.values
                )
                # plot raw data of failed samples
                if len(failed_samples) > 0:
                    Parallel(n_jobs=n_jobs)(
                        delayed(self.plotfig)(
                            output_path=output_path,
                            datatype="raw",
                            wellID=i,
                        )
                        for i in failed_samples
                    )
            else:
                for i in self.plate_fit.columns.values:
                    # save to default path
                    self.plotfig(output_path, i)

                # plot failed samples, if any
                # TODO add the derivative curve so that the plots look nicer
                if len(failed_samples) > 0:
                    for i in failed_samples:
                        self.plotfig(output_path=output_path, datatype="raw", wellID=i)


class MoltenProtFitMultiple:
    """The main class in MoltenProt; contains one or more datasets (i.e. MoltenProtFit instances)
    and coordinates their processing.
    """

    def __init__(
        self,
        scan_rate=None,
        denaturant=None,
        layout=None,
        source=None,
    ) -> None:
        """Only core settings are defined at the level of initialization:

        layout:DF - annotation for all samples
        denaturant:str - can be either a unit of temperature (C, K) or denaturant name (GuHCl, Urea)
        all internal processing is done with temperature in Kelvins
        source: string - the name of the file that was used to get the data (currently not the real path)
        debug level?

        All datasets must be added using a dedicated method.
        Defaults to None, which is useful to re-create instances from JSON files
        """
        self.layout = layout
        if layout is not None:
            self.layout_raw = layout.copy()  # a backup copy of the original layout
        else:
            self.layout_raw = None
        self.source = source
        self.scan_rate = scan_rate
        self.denaturant = denaturant
        self.datasets = {}  # type: ignore[var-annotated]

    def __getstate__(self):
        """A special method to enable pickling of class methods."""
        output = self.__dict__
        output["PrepareAndAnalyseSingle"] = self.PrepareAndAnalyseSingle
        output["WriteOutputSingle"] = self.WriteOutputSingle
        return output

    def print_message(self, text, message_type):
        """Print a message and add it to the protocolString of all datasets."""
        if message_type == "w":
            prefix = "Warning: "
        elif message_type == "i":
            prefix = "Information: "
        else:
            msg = f"Unknown message type '{message_type}'"
            raise ValueError(msg)
        message = f"{prefix}{text} (All)"
        print(message)
        message += "\n"
        for i, j in self.datasets.items():
            j.protocolString += message

    def GetAnalysisSettings(self):
        """Get information on the analysis settings (shared settings and per-dataset model settings).

        Returns:
        -------
        a tuple of two variables:
            > dict with current analysis settings
            > dataframe with dataset/model settings (compatible with QtTableWidget)
        """
        # NOTE currently all settings except for model are the same for all datasets
        # start from default settings
        analysis_settings = analysis_kwargs(analysis_defaults)
        model_settings = pd.Series(
            index=self.GetDatasets(),
            name="Model",
            data="santoro1988",
        )
        model_settings.index.name = "Dataset"

        for dataset_name, dataset in self.datasets.items():
            if dataset.analysisHasBeenDone():
                analysis_settings = analysis_kwargs(dataset.__dict__)
                model_settings[dataset_name] = dataset.model
            # unprocessed MPF may not have the model attribute
            if hasattr(dataset, "model") and dataset.model == "skip":
                model_settings[dataset_name] = dataset.model

        # reset index to comply with QtTableWidget
        model_settings = model_settings.reset_index()

        return (analysis_settings, model_settings)

    def AddDataset(self, data, readout):
        """Add an individual Dataset (as MP_fit instance in datasets attribute).

        Parameters
        ----------
        data: pd.DataFrame
            a ready-togo DataFrame with Index called "Temperature" or "Denaturant" (not implemented yet)
        readout: str
            a name for the dataset to be added; will be used to call internal data and thus has to be unique; spaces will be substituted with underscore

        Notes:
        -----
        TODO add an option to indicate that the data is about aggregation (to rename to Tagg)
        TODO set layouts here as well?
        """
        if readout in list(self.datasets.keys()):
            msg = "Readout '{}' is already present in this MoltenProtFitMultiple instance!"
            raise RuntimeError(
                msg,
            )

        # in readout name convert spaces to underscores
        readout = readout.replace(" ", "_")
        # NOTE parent_filename is only used in reports
        self.datasets[readout] = MoltenProtFit(
            data,
            scan_rate=self.scan_rate,
            input_type="from_xlsx",
            parent_filename=self.source,
            denaturant=self.denaturant,
            readout_type=readout,
        )

    def DelDataset(self, todelete):
        """Remove a dataset from the instance.

        Parameters
        ----------
        todelete
            which dataset to delete

        Notes:
        -----
        This permanently removes a dataset, so the data cannot be recovered. If the data should be retained, then set model to 'skip' in Analysis
        """
        if todelete in self.GetDatasets():
            del self.datasets[todelete]
        else:
            self.print_message(f"Dataset '{todelete}' not found", "w")

    def GetDatasets(self, no_skip=False):
        """Returns available dataset names.

        Parameters
        ----------
        no_skip
            if True, then datasets with model 'skip' will not be included
        """
        if no_skip:
            output = []
            for dataset_name, dataset in self.datasets.items():
                # NOTE model attribute is only added after processing
                if hasattr(dataset, "model") and dataset.model == "skip":
                    continue
                output.append(dataset_name)
            return tuple(output)
        else:
            return tuple(self.datasets.keys())

    def UpdateLayout(self):
        """For GUI communication only, update the layout of the datasets after the "master" layout in MPFM was changed.

        Notes:
        -----
        BUG this method does not use the MPF.SetLayout, which makes code redundant
        layouts are set only at SetAnalysisOptions state
        """
        for dataset_id, mp_fit in self.datasets.items():
            mp_fit.layout = self.layout
            # also update the layout info in plate_results (if present)
            if hasattr(mp_fit, "plate_results"):
                mp_fit.plate_results["Condition"] = self.layout["Condition"]

    def ResetLayout(self):
        """Change the master layout in MPFM to layout_raw (recorded during parsing of XLSX in newer versions of moltenprot) and update all MPF instances."""
        if self.layout_raw is not None:
            self.layout = (
                self.layout_raw.copy()
            )  # need to copy, because all edits to the layout will propagate to layout_raw, and it will not be "original"
            self.UpdateLayout()
        else:
            self.print_message("Attribute layout_raw is None, nothing to reset", "w")

    def SetScanRate(self, scan_rate):
        """Sets a new scan rate (degC/min) to all datasets."""
        # too high precision is not relevant
        self.scan_rate = round(scan_rate, 2)
        for dataset_id, mp_fit in self.datasets.items():
            mp_fit.scan_rate = self.scan_rate

    def RenameResultsColumns(self, which, mapping):
        """E.g. in scattering data, the output is not Tm, but Tagg
        Also, for chemical denaturation the columns can be completely different
        NOTE currently done via RenameResults of MoltenProtFit.

        Parameters
        ----------
        which : str
            to which dataset to apply
        mapping : dict
            a dictionary with keys being original names and values being new names
        """
        self.print_message(
            f"Renaming results columns in dataset {which}\n{mapping}",
            "i",
        )
        self.datasets[which].plate_results.rename(columns=mapping, inplace=True)
        self.datasets[which].plate_results_stdev.rename(columns=mapping, inplace=True)

    def SetAnalysisOptions(self, which="all", printout=False, **kwargs):
        """Set analysis options for a single dataset.

        Parameters
        ----------
        which : str
            To which dataset the options are applied; all means apply same settings for all datasets
        printout : bool
            print the settings
        **kwargs
            args for MoltenProtFit.SetAnalysisOptions()
        """
        # Add layout-related options to kwargs
        if kwargs.get("blanks"):
            # check if the current layout has any blanks listed and remove those
            self.layout.Condition = self.layout.Condition.replace("Blank", "")
            try:
                self.layout.loc[kwargs["blanks"], "Condition"] = "Blank"
            except KeyError:
                self.print_message(
                    "One or more blank samples have invalid IDs, no blank info will be available",
                    "w",
                )
                # add to all datasets' protocol string

        # NOTE if the same sample is listed as blank and exclude, it will have only exclude in the end
        # i.e. exclusion by user has higher priority
        if kwargs.get("exclude"):
            # check if the current layout has any excluded samples and remove those
            self.layout.Condition = self.layout.Condition.replace("Ignore", "")
            try:
                self.layout.loc[kwargs["exclude"], "Condition"] = "Ignore"
            except KeyError:
                self.print_message(
                    "One or more excluded samples have invalid IDs, no exclusion done",
                    "w",
                )

        kwargs["layout"] = self.layout
        kwargs["layout_input_type"] = "from_xlsx"
        # NOTE the differences from 1x processing:
        # the layout is shared, so we use self.layout, and also tell MoltenProtFit instance that the layout is already a DataFrame (using layout_input_type)
        if which == "all":
            for i, j in self.datasets.items():
                j.SetAnalysisOptions(**kwargs)
        else:
            self.datasets[which].SetAnalysisOptions(**kwargs)

        if printout:
            print(f"Data type is {i}")
            self.datasets[which].printAnalysisSettings()

    def PrepareAndAnalyseSingle(self, which):
        """Run data processing pipeline on a single dataset.

        Parameters
        ----------
        which
            dataset name
        """
        self.datasets[which].PrepareData()
        self.datasets[which].ProcessData()

        # NOTE return statement is only needed for parallelized code (MoltenProtFitMultiple instance
        # gets overwritten and computed results are not stored)
        return self.datasets[which]

    def PrepareAndAnalyseAll(self, n_jobs=1):
        """Run analysis on all datasets.

        Parameters
        ----------
        n_jobs : int
            how many parallel processes to start
        """
        analysis_tuple = self.GetDatasets()

        # parallelization of analysis routine
        if parallelization and n_jobs > 1:
            results_tuple = Parallel(n_jobs=n_jobs)(
                delayed(self.PrepareAndAnalyseSingle)(i) for i in analysis_tuple
            )
            for i, j in zip(analysis_tuple, results_tuple):
                self.datasets[i] = j
        else:
            for i in analysis_tuple:
                self.PrepareAndAnalyseSingle(i)

    def CombineResults(self, outfile, tm_stdev_filt=-1, bs_filt=-1, merge_dup=False):
        """Join all plate_results/stdev DataFrames and write to a single XLSX file.

        Parameters
        ----------
        outfile : str
            where to write the output (a full path)
        tm_stdev_filt : float
            samples with Tm stdev above this value will be discarded
        bs_filt : float
            samples with BS-factor below this value will be discarded
        merge_dup : bool
            whether to aggregate samples with identical annotations in layout

        Notes:
        -----
        TODO: add more generic filtering options
        """
        analysis_tuple = self.GetDatasets()

        # initiate xlsx writer objects
        writer = pd.ExcelWriter(outfile)

        for i in analysis_tuple:
            tm_key = "Tagg" if i == "Scattering" else "Tm"

            # skip non-processed datasets (model=skip)
            if self.datasets[i].model != "skip":
                output = self.datasets[i].CombineResults(
                    tm_stdev_filt=tm_stdev_filt,
                    bs_filt=bs_filt,
                    merge_dup=merge_dup,
                    # merge_stdev=merge_stdev, DEPRECATED?
                    tm_key=tm_key,
                )
                # NOTE Excel sheets are now named identically to input dataset names
                output.to_excel(writer, i)

        # write XLSX files to the result folder:
        writer.save()

    def GenerateReport(self, heatmap_cmap, template_path=None):
        """Creates an interactive HTML report (as a string).

        Parameters
        ----------
        heatmap_cmap
            matplotlib colormap for heatmap
        template_path
            the HTML template

        Returns:
        -------
        A string made from report template where placeholders were substituted to actual HTML code
        """
        # use default template if none provided
        if template_path is None:
            template_path = os.path.join(__location__, "resources/report.template")

        # open the html template
        with open(template_path) as template_file:
            template = template_file.read()

        # convert it to a string.Template instance
        template = Template(template)

        heatmap_table = ""
        buttons = ""

        # for the first heatmap display is table, for the rest it is none
        display_heatmap = "table"

        # for a single-dataset MPFm do not show buttons
        display_buttons = "none" if len(self.datasets) == 1 else ""

        # cycle through all datasets and get glue up strings to the starting button or heatmap string
        for dataset_name, dataset in self.datasets.items():
            # skip non-processed datasets
            if dataset.model != "skip":
                heatmap_table += dataset.html_heatmap(heatmap_cmap, display_heatmap)
                if display_heatmap == "table":
                    display_heatmap = "none"
                    first_heatmap = dataset_name
                buttons += dataset.html_button()

        return template.substitute(
            FILE=self.source,
            FIRST_HEATMAP=first_heatmap,
            HEATMAP_TABLE=heatmap_table,
            DISPLAY_BUTTONS=display_buttons,
            BUTTONS=buttons,
            VERSION=__version__,
            TIMESTAMP=strftime("%c"),
        )

    def WriteOutputSingle(
        self,
        which,
        outfolder,
        subfolder=False,
        **kwargs,  # keyword args for WriteOutput
    ):
        """Write output to disc for a single dataset.

        Parameters
        ----------
        which : dataset to process
        outfolder : folder for output
        heatmap_cmap : str
            matplotlib colormap for heatmap
        xlsx : bool
            write output in XLSX format (default is CSV)
        genpics : bool
            create figures for samples
        heatmaps : list
            create heatmaps for the column in list
        subfolder : bool
            write output in outfolder (default) or create a subfolder called "which_resources"
        n_jobs : int
            how many parallel processes can be spawned
        no_data : bool
            no data output

        Notes:
        -----
        * No output generated for datasets with model "skip"
        * Do not use this method directly, use WriteOutputAll instead
        """
        if self.datasets[which].model == "skip":
            pass
        else:
            if subfolder:
                outfolder = os.path.join(outfolder, which + "_resources")
                os.makedirs(outfolder, exist_ok=True)

            # HACK to minimize edits to MoltenProtFit assingment of outfolder is done via the attribute
            self.datasets[which].resultfolder = outfolder
            self.datasets[which].WriteOutput(
                resources_prefix=which,
                **kwargs,
            )
            # delete the attribute completely
            del self.datasets[which].resultfolder

    def WriteOutputAll(
        self,
        outfolder,
        # report,
        xlsx=False,
        genpics=False,
        heatmaps=[],
        report_format=None,
        heatmap_cmap=defaults["heatmap_cmap"],
        n_jobs=1,
        no_data=False,
        session=False,
    ):
        """Write output to disc for all associated datasets.

        Parameters
        ----------
        outfolder : str
            the folder where report.html will be placed and per-dataset subfolders
        xlsx : bool
            write output in XLSX format (default is CSV)
        report : bool DEPRECATED
            generate HTML report
        summary : bool DEPRECATED
            create a compact summary XLSX file
        report_format : None or str 'pdf', 'html', 'xlsx'
        n_jobs : int
            how many output processes to run
        genpics : bool
            create figures for samples
        heatmaps : list
            create heatmaps for the column in list
        heatmap_cmap : str
            matplotlib colormap for heatmap
        session : bool
            save MP session in JSON format
        """
        # generate and populate a dict of output settings
        output_kwargs = {}

        if len(self.datasets) == 1:
            # for single-dataset instances (and no reports planned)
            # write everything to outdir
            output_kwargs["subfolder"] = False

        if xlsx:
            output_kwargs["xlsx"] = True
        if heatmaps:
            # heatmaps should be a list with one or more elements
            output_kwargs["heatmaps"] = heatmaps
        if genpics:
            output_kwargs["genpics"] = True

        output_kwargs["heatmap_cmap"] = heatmap_cmap
        output_kwargs["no_data"] = no_data

        # NOTE since reports are pre-defined data bundles, they may override some of the previous settings
        if report_format == "html":
            # generate a reporthtml string
            reporthtml = self.GenerateReport(heatmap_cmap=heatmap_cmap)
            # write all datatets to dedicated subfolders
            output_kwargs["subfolder"] = True
            # write the HTML of the report to outdir
            with open(os.path.join(outfolder, "report.html"), "w") as file:
                file.write(reporthtml)
            output_kwargs["xlsx"] = True
            output_kwargs["genpics"] = True
            output_kwargs["no_data"] = False
        elif report_format == "xlsx":
            self.CombineResults(os.path.join(outfolder, "report.xlsx"), -1, -1, False)
        elif report_format == "pdf":
            output_kwargs["pdf"] = True

        # write output in parallel or serially
        if parallelization and n_jobs > 1:
            if len(self.GetDatasets()) == 1:
                # if there is only a single associated dataset, it makes sense to enable parallel figure plotting
                self.WriteOutputSingle(
                    self.GetDatasets()[0],
                    outfolder,
                    n_jobs=n_jobs,
                    **output_kwargs,
                )
            else:
                Parallel(n_jobs=n_jobs)(
                    delayed(self.WriteOutputSingle)(i, outfolder, **output_kwargs)
                    for i in self.GetDatasets()
                )
        else:
            # resultfolder was cleaned previously or created fresh so we just have to supply a proper prefix
            for i in self.GetDatasets():
                self.WriteOutputSingle(i, outfolder, **output_kwargs)

        # NOTE JSON dumping must be done _AFTER_ all parallelized jobs!
        if session:
            mp_to_json(self, output=os.path.join(outfolder, "MP_session.json"))


class MoltenProtFitMultipleLE(MoltenProtFitMultiple):
    """A special class to handle the lumry_eyring model
    The main difference is that Scattering signal is required and
    the datasets are processed sequentially.
    """

    def PrepareAndAnalyseAll(self, n_jobs=1):
        """First the scattering data is fit to get Ea and Tf for reaction U->A
        Then they are supplied as fixed parameters to fit reaction N <-kF, kR -> U
        This fit is done in all other datasets.
        """
        # rename dataset if refolded data was used
        if "Scattering (Unfolding)" in self.GetDatasets():
            self.datasets["Scattering"] = self.datasets.pop("Scattering (Unfolding)")

        # check if Scattering signal is present
        if "Scattering" not in self.GetDatasets():
            msg = "lumry_eyring model requires a Scattering dataset"
            raise ValueError(msg)

        # run analysis of Scattering data with irrev model (has to be changed from whatever was supplied)
        self.datasets["Scattering"].model = "irrev"
        self.PrepareAndAnalyseSingle("Scattering")
        # cycle through all other datasets and add fixed parameters
        for dataset in self.GetDatasets():
            if dataset != "Scattering":
                # NOTE in GUI it is possible that some datasets are not skipped, are not Scattering and
                # do not have lumry_eyring model, for those SetFixedParameters will cause AttributeError,
                # because non-LE models do not use this feature
                if self.datasets[dataset].model == "lumry_eyring":
                    self.datasets[dataset].SetFixedParameters(
                        self.datasets["Scattering"]
                        .plate_results.loc[:, ["Tf_fit", "Ea_fit"]]
                        .T,
                    )
                self.PrepareAndAnalyseSingle(dataset)


class MoltenProtFitMultipleRefold(MoltenProtFitMultiple):
    """Contains special tweaks to work with refolding data from Prometheus NT.48.

    * Analyse all datasets, but not the refolding datasets (addDataset for normal unfolding, addRefoldDataset
    for refolding datasets, maintain them separately, but use the same keys in datasets dict)
    * take the baseline parameters from unfolding datasets and draw fraction refolded
    * for fraction refolded generate a separate report HTML file
    * how to show this in GUI?

    Eventually:
    * add fraction_irreversibly_unfolded to the equation and fit refolding data
    using the parameters fro unfolding as starting values
    """

    pass


### Data parsers
"""
Functions to create a MoltenProtFitMultiple instance from a specific
experimental data file.
"""


def _csv_helper(filename, sep, dec):
    """Pre-processing steps for reading CSV files.

    Returns:
    -------
    pd.DataFrame with Temperature as index
    """
    try:
        data = pd.read_csv(
            filename,
            sep=sep,
            decimal=dec,
            index_col="Temperature",
            encoding="utf-8",
        )
    except ValueError as e:
        print(e)
        msg = "Input *.csv file is invalid!\nCheck if column called 'Temperature' exists and separators are specified correctly"
        raise ValueError(
            msg,
        )

    # check if index contains duplicates and drop those
    if data.index.duplicated().any():
        print(
            "Warning: Temperature scale contains duplicates, all but first occurence are dropped",
        )
        data = data.loc[data.index.drop_duplicates(), :]
    return data


def parse_plain_csv(
    filename,
    scan_rate=None,
    sep=defaults["sep"],
    dec=defaults["dec"],
    denaturant=defaults["denaturant"],
    readout=defaults["readout"],
    layout=defaults["layout"],
):
    """Parse a standard CSV file with columns Temperature, A1, A2, ...

    Parameters
    ----------
    filename : str
        path to csv file
    sep,dec - csv import parameters
    denaturant : str
        temperature in input file assumed  to be Celsius (default value C), but could be also in K
    readout : str
        name for the experimental technique (e.g. CD or F330), will be used as key in dataset dict
    layout : str or None
        specify a special *.csv file which defines the plate layout (i.e. what conditions are in each sample)

    Returns:
    -------
    MoltenProtFitMultiple instance

    #TODO add removal of zeros in A01, A02, etc
    """
    # read the CSV into a DataFrame
    data = _csv_helper(filename, sep, dec)

    # read layout (if provided)
    if layout is not None:
        try:
            layout = pd.read_csv(layout, index_col="ID", encoding="utf_8")
        except:
            print(
                "Warning: unsupported layout format! No layout info will be available",
            )
            layout = None

    if layout is None:
        # if no layout provided or could not be read, create an empty layout DataFrame
        layout = pd.DataFrame(index=alphanumeric_index, columns=["Condition"])

    # initialize and return a MoltenProtFitMultiple instance
    output = MoltenProtFitMultiple(
        scan_rate=scan_rate,
        denaturant=denaturant,
        layout=layout,
        source=filename,
    )
    output.AddDataset(data, readout)
    return output


def parse_spectrum_csv(
    filename,
    scan_rate=None,
    denaturant=defaults["denaturant"],
    sep=defaults["sep"],
    dec=defaults["dec"],
    readout=defaults["readout"],
):
    """Parse CSV file with columns Temperature,wavelengths...

    Parameters
    ----------
    filename : str
        path to csv file
    sep,dec - csv import parameters
    denaturant : str
        temperature in input file assumed to be Celsius (default value C), but could be also in K
    readout : str
        name for the experimental technique (e.g. CD or F330), will be used as key in dataset dict

    Returns:
    -------
    MoltenProtFitMultiple instance

    Notes:
    -----
    * Temperature axis is not sorted
    * Layouts are generated automatically from column names (assumed to be respective wavelengths)
    """
    data = _csv_helper(filename, sep, dec)

    # if data is too big, take a random subset
    if len(data.columns) > 96:
        print(
            f"Warning: too many wavelengths in the spectrum ({len(data.columns)}), selecting random 96",
        )
        data = data.sample(n=96, axis=1)
    # to be on the safe side, sort columns ascending
    data = data.loc[:, sorted(data.columns)]
    # apply the alphanumeric index
    data = data.T
    data["ID"] = list(alphanumeric_index[: len(data)])
    data.index.name = "Condition"
    data.reset_index(inplace=True)

    # set ID as index
    data.set_index("ID", inplace=True)
    # extract layout info and drop from the main df
    # initialize the layout dataframe
    layout = pd.DataFrame(index=alphanumeric_index, columns=["Condition"])
    layout.index.name = "ID"
    layout.loc[data.index, "Condition"] = data.loc[:, "Condition"].copy()
    data.drop(["Condition"], axis=1, inplace=True)
    data = data.T

    # initialize and return a MoltenProtFitMultiple instance
    output = MoltenProtFitMultiple(
        scan_rate=scan_rate,
        denaturant=denaturant,
        layout=layout,
        source=filename,
    )
    output.AddDataset(data, readout)
    return output


def parse_prom_xlsx(filename, raw=False, refold=False, LE=False, deltaF=True):
    """Parse a processed file from Prometheus NT.48. In these files temperature
    is always in Celsius and the readouts are more or less known. Layout is read
    from the overview sheet.

    Parameters
    ----------
    filename : str
        path to xlsx file
    raw : bool
        if the data is "raw" or "processed" in terms of the manufacturer's GUI
    refold : bool
        if refolding ramp was used (default False)
    LE : bool
        instead of standard instance, create the one with Lumry-Eyring model
    deltaF : bool
        compute an alternative signal-enhanced readout: F350-F330 difference
        it is an extensive readout (proportional to protein conc, like F330 or F350),
        which also makes the transitions more pronounced (like Ratio)

    Returns:
    -------
    MoltenProtFitMultiple instance

    Notes:
    -----
    * Parsing relies on sheet names in English
    * Current implementation can successfully parse raw XLSX as long as there are less than 96 data columns (which is 3 times the number of capillaries). The data will be contaminated with straight lines of temperature and time

    Todo:
    * the sheets have a standardized order in the file: Overview, Readout1_(unfolding), Readout1_(unfolding)_derivative ...,  Readout1_(refolding), Readout1_(refolding)_derivative ... ; this can be used to parse data independently of sheet labels; also, the sheets containing "deriv" can be auto-excluded
    """
    # force the input file to have absolute path (to be stored in JSON session)
    filename = os.path.abspath(filename)

    # read the whole Excel file - get an Ordered Dict
    input_xlsx = pd.read_excel(filename, None)

    # parse the layout
    # layout contains 3 columns: Condition, Capillary and dCp
    # the capillary info can be appended during report generation, but not earlier (needed for Blanks/References etc)
    # NOTE if the user manipulated the Overview sheet, additional non-data rows can be read in and produce a messy layout DF. This doesn't seem to affect the processing
    if "Overview" not in input_xlsx:
        msg = f"Input file {filename} contains no overview sheet"
        raise ValueError(msg)

    layout = input_xlsx["Overview"]
    layout.reset_index(inplace=True)
    # read scan rate from first row in column "Temperature Slope"
    # NOTE without conversion to float scan_rate (even if 1) will be saved to JSON as "null"!
    scan_rate = float(layout["Temperature Slope"].iloc[0])

    layout = layout.reindex(["Capillary", "Sample ID", "dCp"], axis=1)
    layout.rename(columns={"Sample ID": "Condition"}, inplace=True)
    # concatenate A1-H12 and description, then use the "ID" column as the new index
    layout = pd.concat([layout, alphanumeric_index], axis=1)
    layout.set_index("ID", inplace=True)

    # initialize a MoltenProtFitMultiple instance
    if LE:
        output = MoltenProtFitMultipleLE(
            scan_rate=scan_rate,
            layout=layout,
            denaturant="C",
            source=filename,
        )
    else:
        output = MoltenProtFitMultiple(
            scan_rate=scan_rate,
            layout=layout,
            denaturant="C",
            source=filename,
        )

    # cycle through available readouts and add them to MPMultiple
    if refold:
        # Full list of readouts, currently only unfolding can be processed
        # TODO add a class to process refolding data in conjunction with unfolding
        readouts = (
            "Ratio (Unfolding)",
            "330nm (Unfolding)",
            "350nm (Unfolding)",
            "Scattering (Unfolding)",
            "Ratio (Refolding)",
            "330nm (Refolding)",
            "350nm (Refolding)",
            "Scattering (Refolding)",
        )
        output.print_message(
            "Currently refolding data is treated separately from unfolding data",
            "w",
        )
    else:
        readouts = ("Ratio", "330nm", "350nm", "Scattering")

    # NOTE to avoid multiple checks of the scan rate (temp and time scale are the same for all readouts)
    refined_scan_rate = None
    for i in readouts:
        if i in list(input_xlsx.keys()):
            data = input_xlsx[i]
            """
            Convert the read sheet from *.xlsx
            The first column ("Unnamed: 0") contains several rows with NaN values that correspond to one or more columns of the annotations; there is at least one Called Sample ID, and then additional user-defined names. Those have to be removed
            The next row contains the value "u'Time [s]'", and it becomes the first row once the previous operation is done.
            """
            data = data[data.iloc[:, 0].notna()]
            data = data.iloc[1:, :]

            if raw:
                # warn the user that there is a potentially harmful data modification
                output.print_message(
                    "Import of raw data requires interpolation to have all readings on the same temperature scale, i.e. the data gets irreversibly modified",
                    "w",
                )
                # in proc data currently there will be: shared time, shared temp, readings
                # in raw data there are 3 columns for each sample: time, temperature, readings
                # NOTE in some older versions of the raw data the time and temperature are actually the same!
                # care must be taken if scan_rate is determined from such files

                # extract readings, temperatures and times
                readings = data.iloc[:, 2::3].copy()
                temps = data.iloc[:, 1::3]
                times = data.iloc[:, 0::3]
                # the time and temperature of the first sample will be used in the final scale
                time_scale = times.iloc[:, 0].astype(float)
                temp_scale = temps.iloc[:, 0].astype(float)
                # cycle through all samples and perform interpolation
                for col_ix in range(len(readings.columns)):
                    r_col = readings.columns[col_ix]
                    t_col = temps.columns[col_ix]
                    # interpolation function
                    interpolator = interp1d(
                        temps[t_col],
                        readings[r_col],
                        bounds_error=False,
                    )
                    # interpolated values for readings'
                    readings.loc[:, r_col] = interpolator(temp_scale)
                # add time and temperature of the first sample to the output data
                data = pd.concat([time_scale, temp_scale, readings], axis=1)

            # determine true scan rate by running a linear fit of temperature vs time
            if refined_scan_rate is None:
                temp_vs_time = data.iloc[:, :2].astype(float).dropna()
                slope, intercept = np.polyfit(
                    temp_vs_time.iloc[:, 0],
                    temp_vs_time.iloc[:, 1],
                    1,
                )
                refined_scan_rate = slope * 60  # convert degC/sec to degC/min

            # remove the first column with time
            data.drop(data.columns[0], inplace=True, axis=1)
            # rename the first column to "Temperature"
            data.rename(columns={data.columns[0]: "Temperature"}, inplace=True)
            # set Temperature as the index column
            data.set_index("Temperature", inplace=True)
            # convert column names to A1-D12
            data.columns = list(alphanumeric_index.iloc[0 : len(data.columns)])
            # for compatibility with future pandas versions we must make sure that data type is float32
            data = data.apply(pd.to_numeric, errors="coerce")

            # Create a MoltenProtFit instance with this DataFrame as data source
            output.AddDataset(data, i)
        else:
            output.print_message(f"Readout {i} not found", "w")

    if deltaF:
        # check if F330 and F350 are available
        if {"330nm", "350nm"}.issubset(output.GetDatasets()):
            readout_ids = ("330nm", "350nm")
        elif {"330nm_(Unfolding)", "350nm_(Unfolding)"}.issubset(
            output.GetDatasets(),
        ):
            readout_ids = ("330nm_(Unfolding)", "350nm_(Unfolding)")
        else:
            output.print_message(
                "Cannot compute readout deltaF: either F330 or F350 is missing in the input data",
                "w",
            )
            readout_ids = None

        if readout_ids is not None:
            # HACK Temperature in other datasets is already in Kelvins, but denaturant is set to C
            # to prevent adding extra 273.15 to index temporarily set denaturant to K
            output.denaturant = "K"
            output.AddDataset(
                output.datasets[readout_ids[1]].plate
                - output.datasets[readout_ids[0]].plate,
                "deltaF",
            )
            output.denaturant = "C"

    # Check if any datasets could be properly added
    if len(output.datasets) < 1:
        msg = f"Input file {filename} contains no data"
        raise ValueError(msg)

    # assign refined scan_rate
    if abs(refined_scan_rate - output.scan_rate) > 0.2:
        # in range 1-7 degC/min the difference between nominal and true rate is less than 0.2 deg/min
        output.print_message(
            f"The difference between nominal ({output.scan_rate}) and estimated ({refined_scan_rate}) scan rate >0.2 degrees/min",
            "w",
        )
    output.SetScanRate(refined_scan_rate)

    return output


def mp_from_json(input_file):
    """Read a json file and if successful return a ready-to-use MoltenProtFit instance.

    Parameters
    ----------
    input_file
        input file in JSON format

    Notes:
    -----
    BUG column ordering is messed up after JSON I/O
    """
    with open(input_file) as file:
        return json.load(file, object_hook=deserialize)


def mp_to_json(object_inst, output=None):
    """Convert an MoltenProtFit/MPFMultiple instance to a JSON file.

    Parameters
    ----------
    object_inst :
        MP instance to be converted to JSON
    output : string or None
        if None, return a JSON string
        if output=='self', then use object_inst.resultfolder attribute
        otherwise use str as a location where to write

    Returns:
    -------
    string
        only if output parameter is None

    Notes:
    -----
    BUG Column sorting is usually A1 A2..., but after json i/o it is A1 A10...
    """
    if output is None:
        # indent=4 makes the output JSON more human-readable
        return json.dumps(object_inst, default=serialize, indent=4)

    # if output is some kind of string we can use it to write the output
    if output == "self":
        # DELETE only useful for old MPFMultiple
        # a special case is when self.resultfolder is used
        # in all other just use the user-provided string
        output = os.path.join(object_inst.resultfolder, "MP_session.json")

    with open(output, "w") as file:
        json.dump(object_inst, file, default=serialize, indent=4)

    """
    # add compression: output file is smaller, but the process itself is slower
    with gzip.GzipFile(self.resultfolder+'/MP_session.json.gz', 'w') as fout:
        fout.write(json.dumps(self, default=serialize, indent=4))
    """
    return None
