import pyopenms as oms
import polars as pl


def load_spectra(path_to_mzml: str) -> oms.MSExperiment:
    """Load the mzML file.

    :param path_to_mzml: a string representing the path to the mzML file on the system.

    :returns: an oms.MSExperiment with spectra in the mzML file.
    """
    exp = oms.MSExperiment()
    oms.MzMLFile().load(path_to_mzml, exp)
    return exp


def get_precursors(exp: oms.MSExperiment) -> pl.DataFrame:
    """Gets the precursors of each MS2 spectrum.

    :param exp: the experiment for which the precursors are to be collected.

    :returns: a polars.DataFrame with columns ["precursor_mass", "precursor_charge"]
    """
    precursors = [
        spectrum.getPrecursors()[0]
        for spectrum in exp.getSpectra()
        if spectrum.getMSLevel() == 2
    ]
    return pl.DataFrame(
        {
            "weight": [p.getUnchargedMass() for p in precursors],
            "charge": [p.getCharge() for p in precursors],
        }
    )
