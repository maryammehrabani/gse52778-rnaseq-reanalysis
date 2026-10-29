"""Executed paired-donor GSE52778 RNA-seq reanalysis using PyDESeq2."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import warnings
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]

def main():
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    raw = pd.read_csv(ROOT / "data/counts.tsv.gz", sep="\t", index_col=0)
    meta = pd.read_csv(ROOT / "data/samples.tsv", sep="\t", index_col=0)
    annot = pd.read_csv(ROOT / "data/gene_annotations.tsv.gz", sep="\t", index_col=0)
    assert raw.index.is_unique and raw.columns.is_unique and meta.index.is_unique
    assert set(raw.columns) == set(meta.index)
    meta = meta.loc[raw.columns]
    assert raw.shape == (63677, 8) and (raw.to_numpy() >= 0).all()
    assert (raw.to_numpy() == np.floor(raw.to_numpy())).all()
    assert pd.crosstab(meta.donor, meta.condition).eq(1).all().all()
    meta["donor"] = pd.Categorical(meta.donor, categories=sorted(meta.donor.unique()))
    meta["condition"] = pd.Categorical(meta.condition, categories=["untreated", "dexamethasone"])
    keep = (raw >= 10).sum(axis=1) >= 4
    counts = raw.loc[keep].T
    inference = DefaultInference(n_cpus=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dds = DeseqDataSet(counts=counts, metadata=meta, design="~ donor + condition",
                            fit_type="parametric", size_factors_fit_type="ratio",
                            refit_cooks=False, inference=inference)
        design = dds.obsm["design_matrix"]
        assert np.linalg.matrix_rank(design.to_numpy()) == design.shape[1] == 5
        assert design.shape[0] - design.shape[1] == 3
        design.to_csv(out / "design_matrix.tsv", sep="\t")
        dds.deseq2()
        dds.var.to_csv(out / "fit_diagnostics.tsv.gz", sep="\t", compression={"method": "gzip", "mtime": 0})
        if "_LFC_converged" in dds.var:
            assert dds.var["_LFC_converged"].all(), "Some LFC estimates did not converge"
        stats = DeseqStats(dds, contrast=["condition", "dexamethasone", "untreated"],
                            alpha=0.05, cooks_filter=True, independent_filter=True,
                            inference=inference)
        stats.summary()
        result = stats.results_df.copy()
        result.index.name = "gene_id"
        result = result.join(annot[["gene_name", "gene_biotype"]])
        result["fdr_significant"] = result.padj.lt(0.05)
        result["selected_posthoc"] = result.fdr_significant & result.log2FoldChange.abs().ge(1)
        result["direction"] = np.select([result.selected_posthoc & result.log2FoldChange.gt(0),
                                          result.selected_posthoc & result.log2FoldChange.lt(0)],
                                         ["up", "down"], default="not_selected")
        result["test_status"] = np.select([result.pvalue.isna(), result.padj.isna()],
                                           ["pvalue_unavailable", "independent_filter"], default="tested")
        result.sort_values(["padj", "pvalue"]).to_csv(out / "differential_expression.tsv.gz", sep="\t",
                    compression={"method": "gzip", "mtime": 0})
        all_genes = annot.join(result.drop(columns=["gene_name", "gene_biotype"]), how="left")
        all_genes["prefilter_pass"] = keep
        all_genes.loc[~keep, "test_status"] = "removed_low_count"
        all_genes.to_csv(out / "all_input_genes.tsv.gz", sep="\t",
                    compression={"method": "gzip", "mtime": 0})
        normalized = pd.DataFrame(dds.layers["normed_counts"], index=meta.index, columns=counts.columns).T
        normalized.index.name = "gene_id"
        normalized.to_csv(out / "normalized_counts.tsv.gz", sep="\t",
                    compression={"method": "gzip", "mtime": 0})
        qc = meta.copy()
        qc["raw_library_size"] = raw.sum(axis=0)
        qc["size_factor"] = dds.obs["size_factors"]
        qc.to_csv(out / "sample_qc.tsv", sep="\t")
        dds.vst(use_design=True)
        vst = pd.DataFrame(dds.layers["vst_counts"], index=meta.index, columns=counts.columns)
        vst.T.to_csv(out / "vst.tsv.gz", sep="\t", compression={"method": "gzip", "mtime": 0})
        top500 = vst.var(axis=0, ddof=1).sort_values(ascending=False).head(500).index
        pca = PCA(n_components=2, svd_solver="full")
        coords = pca.fit_transform(vst.loc[:, top500])
        pca_data = meta.copy(); pca_data["PC1"] = coords[:, 0]; pca_data["PC2"] = coords[:, 1]
        pca_data.to_csv(out / "pca_coordinates.tsv", sep="\t")
    (out / "warnings.txt").write_text("\n".join(str(w.message) for w in caught)+"\n")
    cris = result.loc[result.gene_name.eq("CRISPLD2")]
    assert len(cris) == 1
    cris_id = cris.index[0]
    cris_row = cris.iloc[0]
    n_up = int((result.direction == "up").sum())
    n_down = int((result.direction == "down").sum())
    versions = {p: importlib.metadata.version(p) for p in
                  ["pydeseq2", "rdata", "numpy", "pandas", "scipy", "scikit-learn", "anndata", "formulaic", "matplotlib"]}
    summary = {"accession": "GSE52778", "analysis_status": "executed", "engine": "PyDESeq2",
      "engine_version": versions["pydeseq2"], "n_samples": 8, "n_donors": 4,
      "donors": [str(v) for v in meta.donor.cat.categories],
      "conditions": ["untreated", "dexamethasone"], "design": "~ donor + condition",
      "contrast": "dexamethasone vs untreated", "design_rank": 5, "residual_df": 3,
      "input_genes": len(raw), "retained_genes": int(keep.sum()), "prefilter_removed": int((~keep).sum()),
      "prefilter": ">= 10 counts in >= 4 samples",
      "wald_null": "log2 fold change = 0", "multiplicity": "Benjamini-Hochberg with independent filtering at alpha 0.05",
      "available_pvalues": int(result.pvalue.notna().sum()),
      "available_padj": int(result.padj.notna().sum()),
      "fdr_significant": int(result.fdr_significant.sum()),
      "posthoc_selected": int(result.selected_posthoc.sum()),
      "posthoc_effect_filter": "absolute unshrunken log2 fold change >= 1",
      "up": n_up, "down": n_down, "lfc_shrinkage": "none",
      "cooks_filter": True, "count_replacement": False,
      "small_sample_note": "Four paired donors; residual df=3. PyDESeq2 warns that dispersion-prior estimation is less stable at low residual degrees of freedom.",
      "pca": {"input": "design-informed VST, top 500 variable retained genes; no batch/donor removal",
              "PC1_percent": float(100*pca.explained_variance_ratio_[0]),
              "PC2_percent": float(100*pca.explained_variance_ratio_[1])},
      "CRISPLD2": {"ensembl_id": str(cris_id), "annotation_source": "airway 1.32.0 gene_name, Ensembl 75",
                    "log2FoldChange": float(cris_row.log2FoldChange), "padj": float(cris_row.padj),
                    "pvalue": float(cris_row.pvalue), "lfcSE": float(cris_row.lfcSE),
                    "fold_change": float(2**cris_row.log2FoldChange)},
      "python": platform.python_version(), "versions": versions,
      "original_study": "Himes et al. 2014; doi:10.1371/journal.pone.0099625",
      "scope": "Independent public-data reanalysis; no clinical validation; not the user's original cohort"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    checksums = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [ROOT/"data/counts.tsv.gz", ROOT/"data/samples.tsv", ROOT/"data/gene_annotations.tsv.gz"]}
    (out/"input_checksums.json").write_text(json.dumps(checksums, indent=2)+"\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
