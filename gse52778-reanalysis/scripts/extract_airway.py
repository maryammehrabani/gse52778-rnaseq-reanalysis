"""Extract exact raw counts and annotations from the official airway package.

Run from any directory with the dependencies in requirements.txt installed.
The downloadable package is cached under source/; no FASTQ reprocessing occurs.
"""
from pathlib import Path
import hashlib
import json
import tarfile
import urllib.request
import warnings
import numpy as np
import pandas as pd
import rdata

ROOT = Path(__file__).resolve().parents[1]
URL = "https://bioconductor.uib.no/packages/3.23/data/experiment/src/contrib/airway_1.32.0.tar.gz"
ARCHIVE_SHA256 = "829f654fc8075ac9d810574e8a8d4672e69e847c45fa2ce598d1ffbc30c7fe19"

def main():
    source = ROOT / "source"
    data = ROOT / "data"
    source.mkdir(exist_ok=True); data.mkdir(exist_ok=True)
    archive = source / "airway_1.32.0.tar.gz"
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=120) as response:
            archive.write_bytes(response.read())
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if ARCHIVE_SHA256 and sha != ARCHIVE_SHA256:
        raise RuntimeError("airway source checksum mismatch")
    names = ["airway/data/airway.RData", "airway/inst/extdata/sample_table.csv",
             "airway/inst/extdata/GSE52778_series_matrix.txt", "airway/DESCRIPTION"]
    with tarfile.open(archive) as tf:
        for name in names:
            (source / Path(name).name).write_bytes(tf.extractfile(name).read())
    # rdata exposes S4 slots as namespaces. It does not execute any R code.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = rdata.read_rda(source / "airway.RData")["airway"]
    (source / "rdata_conversion_warnings.txt").write_text("\n".join(str(w.message) for w in caught))
    gene_data = obj.rowRanges.elementMetadata.listData
    gene_ids = gene_data["gene_id"]
    assert np.array_equal(gene_ids, obj.rowRanges.partitioning.NAMES)
    sample_ids = obj.colData.rownames
    counts = pd.DataFrame(obj.assays.data.listData["counts"], index=gene_ids, columns=sample_ids)
    counts.index.name = "gene_id"
    assert counts.shape == (63677, 8)
    assert counts.index.is_unique and counts.columns.is_unique
    assert (counts.to_numpy() >= 0).all() and np.issubdtype(counts.to_numpy().dtype, np.integer)
    assert counts.iloc[0].tolist() == [679, 448, 873, 408, 1138, 1047, 770, 572]
    original_meta = pd.DataFrame(obj.colData.listData, index=sample_ids)
    csv_meta = pd.read_csv(source / "sample_table.csv", index_col=0)
    for col in ["cell", "dex", "albut", "SampleName", "Run"]:
        assert original_meta[col].astype(str).tolist() == csv_meta.loc[sample_ids, col].astype(str).tolist()
    meta = pd.DataFrame({"sample_id": sample_ids,
                         "donor": original_meta["cell"].astype(str).to_numpy(),
                         "condition": original_meta["dex"].astype(str).map({"untrt": "untreated", "trt": "dexamethasone"}).to_numpy(),
                         "geo_sample": original_meta["SampleName"].astype(str).to_numpy(),
                         "albuterol": original_meta["albut"].astype(str).to_numpy()})
    assert (meta["albuterol"] == "untrt").all()
    assert pd.crosstab(meta.donor, meta.condition).eq(1).all().all()
    annotation = pd.DataFrame({"gene_id": gene_ids, "gene_name": gene_data["gene_name"],
                               "gene_biotype": gene_data["gene_biotype"]})
    counts.to_csv(data / "counts.tsv.gz", sep="\t", compression={"method": "gzip", "mtime": 0})
    meta.to_csv(data / "samples.tsv", sep="\t", index=False)
    annotation.to_csv(data / "gene_annotations.tsv.gz", sep="\t", index=False,
                      compression={"method": "gzip", "mtime": 0})
    provenance = {"accession": "GSE52778", "package": "airway", "version": "1.32.0",
                  "download_url": URL, "archive_sha256": sha,
                  "counts_object": "airway (not Salmon gse)",
                  "dimensions_genes_samples": list(counts.shape),
                  "gene_annotation": "Ensembl release 75 / GRCh37.p13, as distributed in airway",
                  "assay": "raw gene-level fragment counts", "extraction": "rdata 1.1.0, no changes to counts",
                  "upstream": "STAR alignment and summarizeOverlaps Union counting, as documented by airway",
                  "original_study": "Himes et al., PLoS One 2014;9:e99625; doi:10.1371/journal.pone.0099625"}
    (data / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(provenance, indent=2))

if __name__ == "__main__":
    main()
