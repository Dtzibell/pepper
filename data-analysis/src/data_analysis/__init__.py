import pyopenms as oms
import argparse as args
import polars as pl
import matplotlib.pyplot as plt
from .empirical import load_spectra, get_precursors
from .database import load_peptide_database, generate_targets_and_decoys
from .matching import (
    find_best_decoy_and_target_matches,
    filter_most_intense_peaks,
    normalize,
    match_precursors_to_database,
)
import numpy as np
import scipy
from .alignment import (
    plot_pseudo_ROC,
    choose_better_alignment,
    plot_alignment_length_histogram,
)


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
        pl.Series("observed_spectra", ms2, dtype=pl.Object),
    )
    # Schema({'weight': Float64, 'charge': Int64, 'target_matches': List(String), 'decoy_matches': List(String), 'observed_spectra': Object, 'min_charge': Int64, 'best_target_matches': String, 'best_target_alignments': List(List(Int64)), 'best_target_spectra': Object, 'best_decoy_matches': String, 'best_decoy_alignments': List(List(Int64)), 'best_decoy_spectra': Object})

    precursors_matched_to_candidates = find_best_decoy_and_target_matches(
        precursors_matched_to_candidates,
        theoretical_spectrum_generator,
        spectrum_alignment,
    )
    # Schema({'weight': Float64, 'charge': Int64, 'target_matches': List(String), 'decoy_matches': List(String), 'observed_spectra': Object, 'min_charge': Int64, 'best_target_matches': String, 'best_target_alignments': List(List(Int64)), 'best_target_spectra': Object, 'best_decoy_matches': String, 'best_decoy_alignments': List(List(Int64)), 'best_decoy_spectra': Object, 'target_hyperscores': Float64, 'decoy_hyperscores': Float64, 'hyperscores': Float64, 'identified': Boolean})

    plot_alignment_length_histogram(precursors_matched_to_candidates).savefig(
        f"{namespace.outputdir}/peptide_alignment_lengths.png"
    )
    plt.gcf().clear()

    precursors_matched_to_candidates = choose_better_alignment(
        precursors_matched_to_candidates
    )
    precursors_matched_to_candidates.show()
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
