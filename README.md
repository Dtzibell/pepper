# Pepper
A Nextflow workflow for a  [pyopenms](https://pyopenms.readthedocs.io/en/latest/index.html) and [polars](https://pola.rs/) based MS2 spectrum search engine. 

### WARNING: this tool is not suitable for actual scientific analysis. There are certain steps that were taken that differ from what a proper tool should be doing.

To run the programme, you must have nextflow and docker installed.

Run `docker build . -t my:pepper`, wait for the image to build, then run `nextflow run pepper.nf` to see the example workflow.


