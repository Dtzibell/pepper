#!/usr/bin/env nextflow
params {
    dataset_url: String = "https://github.com/KohlbacherLab/CPMAssets/releases/download/2026/u6_data.zip"
}

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
    path "preprocessing/peptide_alignment_lengths.png", emit: alignment_length_histogram
    path "preprocessing/pseudo-ROC.png", emit: ROC
    path "preprocessing/data.zst", emit: final_output

    script:
    """
    mkdir -p preprocessing
    uv run --no-sync --project /opt/data-analysis match-peptides --fasta ${fastaPath} --mzml ${mzmlPath} --outputdir preprocessing
    """
}


workflow {
    main:
    downloadDataset(params.dataset_url)

    unzipDataset(downloadDataset.out)

    matchPrecursorsToDatabase(unzipDataset.out.fasta, unzipDataset.out.mzml)

    publish:
    zipped_dataset = downloadDataset.out
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
