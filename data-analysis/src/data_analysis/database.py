import polars as pl
import pyopenms as oms


def load_peptide_database(path_to_fasta: str) -> list[oms.FASTAEntry]:
    """Load the FASTA file.

    :param path_to_fasta: the path to the mzML file on the system as a string.

    :returns: a list of oms.FASTAEntries in the database
    """
    return oms.FASTAFile().load(path_to_fasta)


def generate_targets_and_decoys(
    fastas: list[oms.FASTAEntry],
    fixed_modifications: list[str] = [],
    variable_modifications: list[str] = [],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generates targets and decoys for a given database.

    :param fastas: a list of pyopenms.FASTAEntry of proteins to search against.
    :param fixed_modifications: a list of strings of modifications that are applied to each amino acid.
    :param variable_modifications: a list of strings of modifications that are possibly applied to amino acids.

    :returns: a tuple of DataFrames (targets, decoys). The DataFrames have columns ["weight", "sequence"].
    # WARNING: The modifications must be available in the pyopenms framework.
    """
    # set up the protease
    protease = oms.ProteaseDigestion()
    protease.setEnzyme("Trypsin")
    protease.setMissedCleavages(0)

    decoy_gen = oms.DecoyGenerator()

    # set up the amino acid modifications
    modified_pep_gen = oms.ModifiedPeptideGenerator()
    fixed_mods = modified_pep_gen.getModifications(fixed_modifications)
    variable_mods = modified_pep_gen.getModifications(variable_modifications)

    target_seqs = [oms.AASequence(entry.sequence) for entry in fastas]
    targets = generate_targets(
        target_seqs,
        protease,
        modified_pep_gen,
        fixed_mods,
        variable_mods,
    )
    decoys = generate_decoys(
        target_seqs, decoy_gen, protease, modified_pep_gen, fixed_mods, variable_mods
    )

    target_weights = [target.getMonoWeight() for target in targets]
    decoy_weights = [decoy.getMonoWeight() for decoy in decoys]

    # strings are enough information here. Dont really need oms.AASequence
    decoys = [seq.toString() for seq in decoys]
    targets = [seq.toString() for seq in targets]

    # write to a DataFrame
    targets = pl.DataFrame({"sequence": targets, "weight": target_weights})
    decoys = pl.DataFrame({"sequence": decoys, "weight": decoy_weights})

    return targets, decoys


def generate_targets(
    target_seqs: list[oms.AASequence],
    protease: oms.ProteaseDigestion,
    mpg: oms.ModifiedPeptideGenerator,
    fixed_modifications: oms.ModifiedPeptideGenerator_MapToResidueType,
    variable_modifications: oms.ModifiedPeptideGenerator_MapToResidueType,
) -> list[oms.AASequence]:
    """Generates a list of targets from a database.

    :param target_seqs: a list of oms.AASequence representing proteins to convert to digested target peptides.
    :param protease: a oms.Protease to digest with.
    :param mpg: a oms.ModifiedPeptideGenerator to modify peptides with.
    :param fixed_modifications: a oms.ModifiedPeptideGenerator_MapToResidueType that are applied on each amino acid of the proteins.
    :param variable_modifications: a oms.ModifiedPeptideGenerator_MapToResidueType that are possibly present on the proteins.

    :returns: a list of oms.AASequence, targets of the database.

    # NOTE: use oms.ModifiedPeptideGenerator::getModifications() to
    retrieve the oms.ModifiedPeptideGenerator_MapToResidueType

    """
    lst = []
    for seq in target_seqs:
        lst.extend(
            apply_modifications(
                digest(seq, protease),
                mpg,
                fixed_modifications,
                variable_modifications,
            )
        )
    return lst


def generate_decoys(
    target_seqs: list[oms.AASequence],
    degen: oms.DecoyGenerator,
    protease: oms.ProteaseDigestion,
    mpg: oms.ModifiedPeptideGenerator,
    fixed_modifications: oms.ModifiedPeptideGenerator_MapToResidueType,
    variable_modifications: oms.ModifiedPeptideGenerator_MapToResidueType,
) -> list[oms.AASequence]:
    """Generates a list of targets from a database.

    :param target_seqs: a list of oms.AASequence representing proteins from which decoys are to be generated.
    :param degen: a oms.DecoyGenerator that reverses the peptides.
    :param protease: a oms.Protease to digest with
    :param mpg: a oms.ModifiedPeptideGenerator to modify peptides with
    :param fixed_modifications: a oms.ModifiedPeptideGenerator_MapToResidueType that are applied on each amino acid of the proteins.
    :param variable_modifications: a oms.ModifiedPeptideGenerator_MapToResidueType that are possibly present on the proteins.

    :returns: a list of oms.AASequence, decoys of the database

    NOTE: use oms.ModifiedPeptideGenerator::getModifications() to
    retrieve the oms.ModifiedPeptideGenerator_MapToResidueType
    """
    lst = []
    for seq in target_seqs:
        lst.extend(
            apply_modifications(
                digest(degen.reversePeptides(seq, protease.getEnzymeName()), protease),
                mpg,
                fixed_modifications,
                variable_modifications,
            )
        )
    return lst


def apply_modifications(
    sequences: list[oms.AASequence],
    mpg: oms.ModifiedPeptideGenerator,
    fixed_mods: oms.ModifiedPeptideGenerator_MapToResidueType,
    var_mods: oms.ModifiedPeptideGenerator_MapToResidueType,
) -> list[oms.AASequence]:
    """Applies fixed and variable modifications to the provided sequences.

    :param sequences: a list of oms.AASequence to apply the modifications on.
    :param mpg: a oms.ModifiedPeptideGenerator to modify peptides with.
    :param fixed_modifications: a oms.ModifiedPeptideGenerator_MapToResidueType that are applied on each amino acid of the proteins.
    :param variable_modifications: a oms.ModifiedPeptideGenerator_MapToResidueType that are possibly present on the proteins.

    :returns: a list of oms.Sequence, modified peptides of the given sequences.
    """
    lst = []
    for seq in sequences:
        mpg.applyFixedModifications(fixed_mods, seq)
        lst.extend(mpg.applyVariableModifications(var_mods, seq, 1, True))
    return lst


def digest(
    sequence: oms.AASequence,
    protease: oms.ProteaseDigestion,
    min_length: int = 6,
    max_length: int = 30,
) -> list[oms.AASequence]:
    """Digests sequences.

    The function excludes digested sequences that contain the amino acid "X".

    :param sequence: a oms.AASequence representing the protein to be digested.
    :param protease: a oms.ProteaseDigestion to digest the protein with.
    :param min_length: the minimum length of a protein for digestion to work.
    :param max_length: the maximum length of a protein for digestion to work.

    :returns: a list of sequences that stem from a given protein.
    """
    lst = protease.digest(sequence, min_length, max_length)
    X = oms.ResidueDB().getResidue("X")
    return [p for p in lst if not p.has(X)]
