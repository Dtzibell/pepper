import polars as pl
import matplotlib.pyplot as plt
import pyopenms as oms
import numpy as np
import scipy


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


def choose_better_alignment(
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
            .then(
                pl.struct(
                    hyperscores="target_hyperscores",
                    identified=True,
                )
            )
            .otherwise(
                pl.struct(
                    hyperscores="decoy_hyperscores",
                    identified=False,
                )
            )
            .struct.unnest()  # https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.when.html
        )
        .collect()
    )
    return df
