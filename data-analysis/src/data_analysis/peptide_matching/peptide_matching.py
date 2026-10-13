import polars as pl
import pyopenms as oms


def find_best_decoy_and_target_alignments(
    precursors_matched_to_candidates: pl.DataFrame,
    theoretical_spectrum_generator: oms.TheoreticalSpectrumGenerator,
    spectrum_alignment: oms.SpectrumAlignment,
) -> pl.DataFrame:
    """For a list of spectra, look for the best alignment for target and decoy candidates.

    :param spectra: a list of observed oms.MSSpectrum.
    :param precursors_macthed_to_candidates: a polars.DataFrame with columns weight, charge, target_matches and decoy_matches.
    :param theoretical_spectrum_generator: a oms.TheoreticalSpectrumGenerator to generate spectra for target and decoy candidates with.
    :param spectrum_alignment: a oms.SpectrumAlignment to align the observed to the theoretical spectra with.
    """

    # memoizer for computed spectra
    computed_spectra = {}

    for spectrum in precursors_matched_to_candidates["observed_spectra"]:
        spectrum.sortByPosition()

    # find the charge to generate candidate spectra with
    precursors_matched_to_candidates = precursors_matched_to_candidates.with_columns(
        pl.min_horizontal(pl.lit(2), pl.col("charge") - 1).alias("min_charge")
    )

    df = find_best_matches(
        precursors_matched_to_candidates,
        theoretical_spectrum_generator,
        spectrum_alignment,
        computed_spectra,
    )

    # drop min_charge because its unnecessary
    precursors_matched_to_candidates.drop("min_charge")
    return df


def find_best_match(
    observed_spectrum: oms.MSSpectrum,
    match_list: list[str],
    min_charge: int,
    theoretical_spectrum_generator: oms.TheoreticalSpectrumGenerator,
    spectrum_alignment: oms.SpectrumAlignment,
    memoizer: dict[tuple[int, str], oms.MSSpectrum] = dict(),
) -> tuple[str | None, list[tuple[int, int]], oms.MSSpectrum | None]:
    """Finds the best match for a given oms.MSSpectrum from a list of sequences.

    :param observed_spectrum: an observed oms.MSSpectrum.
    :param match_list: a list of candidates for a given oms.MSSpectrum.
    :param min_charge: a peptide charge to compute the spectra for.
    :param theoretical_spectrum_generator: a oms.TheoreticalSpectrumGenerator to generate spectra for match_list with.
    :param spectrum_alignment: a oms.SpectrumAlignment to align the observed to the theoretical spectra with.
    :param memoizer: a dictionary with key: (weight, sequence), value: oms.MSSpectrum that stores oms.MSSpectra for potentially already generated weight+sequence combinations.
    """
    best_match = None
    best_alignment: list[tuple[int, int]] = []
    best_spectrum = None
    for candidate in match_list:
        key = (min_charge, candidate)
        if key in memoizer:
            theo_spectrum = memoizer[key]
        else:
            theo_spectrum = oms.MSSpectrum()
            theoretical_spectrum_generator.getSpectrum(
                theo_spectrum, oms.AASequence(candidate), 1, min_charge
            )
            theo_spectrum.sortByPosition()
            memoizer[key] = theo_spectrum
        alignment = spectrum_alignment.getSpectrumAlignment(
            theo_spectrum, observed_spectrum
        )
        if len(alignment) > len(best_alignment):
            best_match = candidate
            best_alignment = alignment
            best_spectrum = theo_spectrum
    return best_match, best_alignment, best_spectrum


def find_best_matches(
    df: pl.DataFrame,
    theoretical_spectrum_generator: oms.TheoreticalSpectrumGenerator,
    spectrum_alignment: oms.SpectrumAlignment,
    memoizer: dict[tuple[int, str], oms.MSSpectrum] = dict(),
) -> pl.DataFrame:
    """find the best matches for a given list of oms.MSSpectra.

    NOTE: this is a concretization of find_best_decoy_and_target_alignments which uses a memoizer, hence the arguments are mostly the same.

    :param memoizer: a dictionary with key: (weight, sequence), value: oms.MSSpectrum that stores oms.MSSpectra for potentially already generated weight+sequence combinations.
    :returns: df with new columns best_target_matches, best_target_alignments, best_decoy_matches, best_decoy_alignments.
    """
    best_targets = [
        find_best_match(
            spectrum,
            m,
            c,
            theoretical_spectrum_generator,
            spectrum_alignment,
            memoizer,
        )
        for m, c, spectrum in zip(
            df["target_matches"], df["min_charge"], df["observed_spectra"]
        )
    ]

    best_decoys = [
        find_best_match(
            spectrum,
            m,
            c,
            theoretical_spectrum_generator,
            spectrum_alignment,
            memoizer,
        )
        for m, c, spectrum in zip(
            df["decoy_matches"], df["min_charge"], df["observed_spectra"]
        )
    ]

    target_matches, target_alignments, target_spectra = zip(*best_targets)
    decoy_matches, decoy_alignments, decoy_spectra = zip(*best_decoys)

    df = df.with_columns(
        pl.Series("best_target_matches", target_matches),
        pl.Series("best_target_alignments", target_alignments),
        pl.Series("best_target_spectra", target_spectra),
        pl.Series("best_decoy_matches", decoy_matches),
        pl.Series("best_decoy_alignments", decoy_alignments),
        pl.Series("best_decoy_spectra", decoy_spectra),
    )

    return df


def filter_most_intense_peaks(exp: oms.MSExperiment, max_n_peaks: int = 200) -> None:
    """In-place filters for max_n_peaks most intense peaks.

    :param exp: oms.MSExperiment to filter.
    :param max_n_peaks: the number of peaks to filter.
    """

    nlargest_filter = oms.NLargest()

    param = oms.Param()
    # Sets the maximum number of peaks to retain.
    param.setValue("n", max_n_peaks)
    nlargest_filter.setParameters(param)

    # Filters the peaks in the PeakMap experiment based on the configured settings.
    nlargest_filter.filterPeakMap(exp)


def normalize(exp: oms.MSExperiment) -> None:
    """Normalizes the peaks.

    :param exp: oms.MSExperiment to normalize.
    """
    normalizer = oms.Normalizer()

    params = normalizer.getParameters()
    params.setValue("method", "to_TIC")
    normalizer.setParameters(params)

    normalizer.filterPeakMap(exp)
