#!/usr/bin/env nextflow

process downloadDataset {
    container 'my:pepper'
    debug true
    
    input:
    val dataset_url

    output:
    path "data.zip"

    script:
    """
    curl -fL -o data.zip ${dataset_url}
    """
}

process unzipDataset {
    container 'my:pepper'
    input:
    path zip

    output: 
    path "dataset/**/*.mz{ML,ml}", emit: mzml
    path "dataset/**/*.fasta", emit: fasta

    script:
    """
    uv run python -m zipfile -e ${zip} dataset
    chmod -R u+rwX dataset
    """
}

process matchPrecursorsToDatabase {
    container 'my:pepper'

    input:
    path fastaPath
    path mzmlPath

    output:
    path "peptide_alignment_lengths.png", emit: alignment_length_histogram
    path "pseudo-ROC.png", emit: ROC
    path "data.zst", emit: final_output

    script:
    """
    uv run --no-sync --project /opt/data-analysis match-peptides --fasta ${fastaPath} --mzml ${mzmlPath} --outputdir .
    """
}


workflow {
    main:
    dataset = downloadDataset(params.dataset_url)
    def (mzml, fasta) = unzipDataset(dataset)
    matchPrecursorsToDatabase(fasta, mzml)
    publish:
    zipped_dataset = dataset
    mzml_file = unzipDataset.out.mzml
    fasta_file = unzipDataset.out.fasta
    roc = matchPrecursorsToDatabase.out.ROC
    final_output = matchPrecursorsToDatabase.out.final_output
    alignment_length_histogram = matchPrecursorsToDatabase.out.alignment_length_histogram
}

output {
    zipped_dataset {}
    mzml_file {}
    fasta_file {}
    roc {mode 'copy'}
    final_output {}
    alignment_length_histogram {mode 'copy'}
}
