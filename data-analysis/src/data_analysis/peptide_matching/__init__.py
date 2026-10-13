import pyopenms as oms
import argparse as args
import polars as pl
import matplotlib.pyplot as plt
from .precursors import load_spectra, get_precursors
from .database_matching import load_peptide_database, generate_targets_and_decoys
from .peptide_matching import (
    find_best_decoy_and_target_alignments,
    filter_most_intense_peaks,
    normalize,
)
import numpy as np
import scipy


def main() -> None:
    parser = args.ArgumentParser()
    parser.add_argument(
        "-f",
        "--fasta_file",
        default="/home/tauras/pepper/results/dataset/u6_data/u6_uniprot_yeast_reviewed_isoforms_ups1_cra_targets.fasta",
    )
    parser.add_argument(
        "-m",
        "--mzml_file",
        default="/home/tauras/pepper/results/dataset/u6_data/u6_UPS1_50000amol_R2.mzML",
    )
    parser.add_argument("-o", "--outputdir", default=".")
    namespace = parser.parse_args()

    # get precursors of each MS2
    exp = load_spectra(namespace.mzml_file)
    precursors_matched_to_candidates = get_precursors(exp)
    # Schema({'weight': Float64, 'charge': Int64})

    # digest the reference database
    fastas = load_peptide_database(namespace.fasta_file)
    targets, decoys = generate_targets_and_decoys(fastas)

    precursors_matched_to_candidates = match_precursors_to_database(
        precursors_matched_to_candidates, targets, "target"
    )
    # Schema({'weight': Float64, 'charge': Int64, 'target_matches': List(String)})
    precursors_matched_to_candidates = match_precursors_to_database(
        precursors_matched_to_candidates, decoys, "decoy"
    )
    # Schema({'weight': Float64, 'charge': Int64, 'target_matches': List(String), 'decoy_matches': List(String)})

    filter_most_intense_peaks(exp)
    normalize(exp)
    ms2 = [s for s in exp.getSpectra() if s.getMSLevel() == 2]

    spectrum_alignment = oms.SpectrumAlignment()
    param = spectrum_alignment.getParameters()
    param.setValue("tolerance", 500.0)
    param.setValue("is_relative_tolerance", "true")
    spectrum_alignment.setParameters(param)

    theoretical_spectrum_generator = oms.TheoreticalSpectrumGenerator()
    param = oms.Param()
    param.setValue("add_y_ions", "true")
    param.setValue("add_b_ions", "true")
    param.setValue("add_first_prefix_ion", "true")
    param.setValue("add_precursor_peaks", "false")
    param.setValue("add_losses", "false")
    param.setValue("add_metainfo", "true")
    theoretical_spectrum_generator.setParameters(param)

    precursors_matched_to_candidates = precursors_matched_to_candidates.with_columns(
        pl.Series("observed_spectra", ms2)
    )
    # Schema({'weight': Float64, 'charge': Int64, 'target_matches': List(String), 'decoy_matches': List(String), 'observed_spectra': Object, 'min_charge': Int64, 'best_target_matches': String, 'best_target_alignments': List(List(Int64)), 'best_target_spectra': Object, 'best_decoy_matches': String, 'best_decoy_alignments': List(List(Int64)), 'best_decoy_spectra': Object})

    precursors_matched_to_candidates = find_best_decoy_and_target_alignments(
        precursors_matched_to_candidates,
        theoretical_spectrum_generator,
        spectrum_alignment,
    )
    # Schema({'weight': Float64, 'charge': Int64, 'target_matches': List(String), 'decoy_matches': List(String), 'observed_spectra': Object, 'min_charge': Int64, 'best_target_matches': String, 'best_target_alignments': List(List(Int64)), 'best_target_spectra': Object, 'best_decoy_matches': String, 'best_decoy_alignments': List(List(Int64)), 'best_decoy_spectra': Object, 'target_hyperscores': Float64, 'decoy_hyperscores': Float64, 'hyperscores': Float64, 'identified': Boolean})

    plot_alignment_length_histogram(precursors_matched_to_candidates).savefig(
        f"{namespace.outputdir}/peptide_alignment_lengths.png"
    )
    plt.gcf().clear()

    precursors_matched_to_candidates = score_better_alignment(
        precursors_matched_to_candidates
    )
    # Schema({'weight': Float64, 'charge': Int64, 'target_matches': List(String), 'decoy_matches': List(String), 'observed_spectra': Object, 'min_charge': Int64, 'best_target_matches': String, 'best_target_alignments': List(List(Int64)), 'best_target_spectra': Object, 'best_decoy_matches': String, 'best_decoy_alignments': List(List(Int64)), 'best_decoy_spectra': Object, 'target_hyperscores': Float64, 'decoy_hyperscores': Float64, 'hyperscores': Float64, 'identified': Boolean})

    plot_pseudo_ROC(precursors_matched_to_candidates).savefig(
        f"{namespace.outputdir}/pseudo-ROC.png"
    )
    precursors_matched_to_candidates = precursors_matched_to_candidates.drop(
        "observed_spectra", "best_target_spectra", "best_decoy_spectra"
    )
    precursors_matched_to_candidates.write_ndjson(
        f"{namespace.outputdir}/data.zst", compression="zstd", compression_level=22
    )
    plt.gcf().clear()


def plot_pseudo_ROC(df: pl.DataFrame) -> plt.Figure:
    """Creates a figure that plots the search engine's Q-value against the number of total peptides.

    polars.DataFrame columns: {hyperscores: polars.Float64, identified: polars.String}.

    :param df: the polars.DataFrame to plot the data from.
    """
    df = (
        df.lazy()
        .sort(by="hyperscores", descending=True)
        .with_columns(
            (pl.col("identified")).cum_sum().alias("identified_targets"),
            (~pl.col("identified")).cum_sum().alias("identified_decoys"),
        )
        .with_columns(
            pl.when(pl.col("identified_targets") != 0)
            .then(pl.col("identified_decoys") / pl.col("identified_targets"))
            .otherwise(float("inf"))
            .alias("fdr")
        )
        .with_columns(pl.col("fdr").cum_min(reverse=True).alias("q-value"))
        .collect()
    )
    plt.plot(df["q-value"], df["identified_targets"], "k")
    plt.title("Pseudo-ROC curve")
    plt.ylabel("Identifications")
    plt.xlabel("q-value")
    plt.xlim(-0.01, 0.2)
    plt.ylim(-100, 20000)
    total_identifications = df.select(
        pl.col("identified_targets").filter(pl.col("fdr") < 0.01).last()
    ).item()
    print(f"At q=0.01, {total_identifications} peptides could be identified")
    return plt.gcf()


def count_y_b_ions(
    spectrum: oms.MSSpectrum,
    spectrum_peak_indices: np.typing.NDArray[np.uint64],
):
    """Counts b and y ions in a theoretical spectrum at provided indices.

    :param theoretical_spectrum: a theoretical spectrum to count the number of b and y ions in.
    :param thereotical_spectrum_peak_indices: indices of the theoretical spectrum to count the ions in.
    """
    sda = spectrum.getStringDataArrays()[0]
    b_ions, y_ions = 0, 0

    for i in spectrum_peak_indices:
        ion_type = sda[i]
        if ion_type[0] == "b":
            b_ions += 1
        elif ion_type[0] == "y":
            y_ions += 1
        # what if ion type is not b/y?
    return b_ions, y_ions


def compute_hyperscore(
    observed_spectrum: oms.MSSpectrum,
    alignment: list[tuple[int, int]],
    candidate_spectrum: oms.MSSpectrum,
):
    """Computes the hyperscore for an alignment.

    TODO: write the formula for hyperscore computation.

    :param observed_spectrum: the empirically observed spectrum.
    :param candidate_spectrum: the theoretical candidate spectrum.
    :param alignment: the alignment of observed with candidate spectrum.
    """
    if candidate_spectrum is None:
        return 0
    exp_spectrum_peak_indices = np.array(
        [peak[1] for peak in alignment], dtype=np.uint64
    )
    theo_spectrum_peak_indices = np.array(
        [peak[0] for peak in alignment], dtype=np.uint64
    )
    # retrieves intensities of peaks in a spectrum
    matched_peaks_intensities = np.take(
        observed_spectrum.get_peaks()[1], exp_spectrum_peak_indices
    )

    b_ions, y_ions = count_y_b_ions(candidate_spectrum, theo_spectrum_peak_indices)

    # computes hyperscore terms
    hyperscore_dp = sum(matched_peaks_intensities)
    log_hyperscore_factorial_part = scipy.special.gammaln(
        y_ions + 1.0
    ) + scipy.special.gammaln(b_ions + 1.0)

    hyperscore = np.log1p(hyperscore_dp) + log_hyperscore_factorial_part
    return hyperscore


def score_better_alignment(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Computes the hyperscores for target and decoy candidates and creates a new column for the better score.

    :param df: a polars.DataFrame with (at least) the schema: {
    observed_spectra: oms.MSSpectrum,
    best_target_alignments: list[tuple[int, int]],
    best_target_spectra: list[tuple[int, int]],
    best_decoy_alignments: list[tuple[int, int]],
    best_decoy_spectra: list[tuple[int, int]],
    }

    """
    target_hyperscores = [
        compute_hyperscore(
            os,
            a,
            ts,
        )
        for os, a, ts in zip(
            df["observed_spectra"].to_numpy(),
            df["best_target_alignments"].to_numpy(),
            df["best_target_spectra"],
        )
    ]
    decoy_hyperscores = [
        compute_hyperscore(
            os,
            a,
            ts,
        )
        for os, a, ts in zip(
            df["observed_spectra"].to_numpy(),
            df["best_decoy_alignments"].to_numpy(),
            df["best_decoy_spectra"],
        )
    ]

    df = (
        df.lazy()
        .with_columns(
            pl.Series("target_hyperscores", target_hyperscores, dtype=pl.Float64),
            pl.Series("decoy_hyperscores", decoy_hyperscores, dtype=pl.Float64),
        )
        .with_columns(
            pl.when(pl.col("target_hyperscores") >= pl.col("decoy_hyperscores"))
            .then(pl.col("target_hyperscores"))
            .otherwise(pl.col("decoy_hyperscores"))
            .alias("hyperscores"),
            pl.when(pl.col("target_hyperscores") >= pl.col("decoy_hyperscores"))
            .then(True)
            .otherwise(False)
            .alias("identified"),
        )
        .collect()
    )
    return df


def match_precursors_to_database(
    precursors: pl.DataFrame,
    peptides: pl.DataFrame,
    match_prefix: str,
    relative_tolerance: float = 0.000005,
) -> pl.DataFrame:
    """Matches each precursor to peptides in a database whose weight is close to the precursor's weight.

    Return a DataFrame that is a horizontally expanded precursors DataFrame, containing the matched peptides' sequence.

    :param precursors: a polars.DataFrame containing the precursors' weight and charge.
    :param peptides: a polars.DataFrame containing the peptides' weight and sequence.
    :param match_prefix: a string that will be prefixed to the new column of the DataFrame.
    :param relative_tolerance: the relative tolerance for matching peptides to precursors.
    """
    peptides = peptides.sort(by="weight")
    weights = peptides["weight"]
    sequences = peptides["sequence"]

    low = weights.search_sorted(precursors["weight"] * (1 - relative_tolerance), "left")
    high = weights.search_sorted(
        precursors["weight"] * (1 + relative_tolerance), "right"
    )

    return precursors.with_columns(
        pl.Series(
            f"{match_prefix}_matches",
            [sequences[lo:hi] for (lo, hi) in zip(low, high)],
        )
    )


def plot_alignment_length_histogram(df: pl.DataFrame) -> plt.Figure:
    """Plots a histogram that represents the distribution of alignment lengths for both target and decoy best alignments.

    :param df: a dataframe containing the best decoy and target alignments.
    """
    alignment_lens = (
        df.lazy()
        .select(
            pl.col("best_target_alignments").list.len().alias("target_alignment_lens"),
            pl.col("best_decoy_alignments").list.len().alias("decoy_alignment_lens"),
        )
        .with_columns(
            pl.max_horizontal("decoy_alignment_lens", "target_alignment_lens")
            .max()
            .alias("longest_alignment"),
        )
        .collect()
    )

    plt.hist(
        [
            alignment_lens["target_alignment_lens"],
            alignment_lens["decoy_alignment_lens"],
        ],
        bins=range(
            0,
            alignment_lens[0, "longest_alignment"] + 1,
            2,
        ),
        stacked=True,
        label=["targets", "decoys"],
        color=["k", "grey"],
    )
    plt.xlabel("Number of aligned peaks")
    plt.ylabel("Number of alignments")
    plt.legend()
    return plt.gcf()
