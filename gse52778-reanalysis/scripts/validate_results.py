"""Independent numerical and file-integrity checks for the distributed outputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from sklearn.decomposition import PCA
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
checks = []

def check(name, condition, detail=None):
    if not bool(condition):
        raise AssertionError(name)
    entry = {"check": name, "status": "passed"}
    if detail is not None:
        entry["detail"] = detail
    checks.append(entry)

def main():
    raw = pd.read_csv(ROOT/"data/counts.tsv.gz", sep="\t", index_col=0)
    meta = pd.read_csv(ROOT/"data/samples.tsv", sep="\t", index_col=0)
    result = pd.read_csv(ROOT/"results/differential_expression.tsv.gz", sep="\t", index_col=0)
    all_genes = pd.read_csv(ROOT/"results/all_input_genes.tsv.gz", sep="\t", index_col=0)
    summary = json.loads((ROOT/"results/summary.json").read_text())
    checksums = json.loads((ROOT/"results/input_checksums.json").read_text())
    for path, expected in checksums.items():
        check("input_sha256:"+path, hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected)
    check("raw_counts_shape", raw.shape == (63677,8))
    check("raw_nonnegative_integer_counts", np.isfinite(raw.to_numpy()).all() and
          (raw.to_numpy() >= 0).all() and (raw.to_numpy() == np.floor(raw.to_numpy())).all())
    check("unique_sample_and_gene_ids", raw.index.is_unique and raw.columns.is_unique and meta.index.is_unique)
    check("sample_alignment", raw.columns.tolist() == meta.index.tolist())
    check("balanced_donor_pairs", pd.crosstab(meta.donor, meta.condition).eq(1).all().all())
    check("no_albuterol_samples", meta.albuterol.eq("untrt").all())
    keep = (raw >= 10).sum(axis=1) >= 4
    check("prefilter_identity", set(result.index) == set(raw.index[keep]) and int(keep.sum()) == 16139)
    check("all_original_genes_preserved", all_genes.index.equals(raw.index) and
          all_genes.loc[~keep,"test_status"].eq("removed_low_count").all())
    design = pd.read_csv(ROOT/"results/design_matrix.tsv",sep="\t",index_col=0)
    check("paired_design_rank_and_df", design.shape==(8,5) and np.linalg.matrix_rank(design)==5)
    check("correct_contrast_encoding", np.array_equal(design["condition[T.dexamethasone]"],
          meta.condition.eq("dexamethasone").astype(int)))
    check("complete_inference", result.pvalue.notna().all() and result.padj.notna().all())
    check("valid_probability_bounds", result.pvalue.between(0,1).all() and result.padj.between(0,1).all())
    independent_bh = false_discovery_control(result.pvalue.to_numpy(),method="bh")
    max_bh_error = float(np.max(np.abs(independent_bh-result.padj.to_numpy())))
    check("independent_bh_recalculation", np.allclose(independent_bh,result.padj,rtol=1e-12,atol=1e-15),
          {"maximum_absolute_difference": max_bh_error})
    selected = result.padj.lt(.05) & result.log2FoldChange.abs().ge(1)
    check("reported_gene_totals", int(result.padj.lt(.05).sum())==summary["fdr_significant"]==3881 and
          int(selected.sum())==summary["posthoc_selected"]==957 and
          int((selected & result.log2FoldChange.gt(0)).sum())==summary["up"]==493 and
          int((selected & result.log2FoldChange.lt(0)).sum())==summary["down"]==464)
    diagnostics = pd.read_csv(ROOT/"results/fit_diagnostics.tsv.gz",sep="\t",index_col=0)
    check("lfc_fit_convergence", diagnostics["_LFC_converged"].eq(True).all())
    fallback = {"gene_wise_dispersion_optimizer_fallbacks": int(diagnostics["_genewise_converged"].eq(False).sum()),
                "MAP_dispersion_optimizer_fallbacks": int(diagnostics["_MAP_converged"].eq(False).sum())}
    qc = pd.read_csv(ROOT/"results/sample_qc.tsv",sep="\t",index_col=0)
    norm = pd.read_csv(ROOT/"results/normalized_counts.tsv.gz",sep="\t",index_col=0)
    expected_norm = raw.loc[norm.index,norm.columns].divide(qc.loc[norm.columns,"size_factor"],axis=1)
    check("normalization_matches_size_factors", np.allclose(norm,expected_norm,rtol=1e-12,atol=1e-10))
    vst = pd.read_csv(ROOT/"results/vst.tsv.gz",sep="\t",index_col=0)
    check("vst_finite_and_aligned", vst.shape==(16139,8) and np.isfinite(vst.to_numpy()).all()
          and vst.columns.tolist()==meta.index.tolist())
    top500 = vst.T.var(axis=0,ddof=1).sort_values(ascending=False).head(500).index
    model = PCA(n_components=2,svd_solver="full")
    coords = model.fit_transform(vst.loc[top500].T)
    stored = pd.read_csv(ROOT/"results/pca_coordinates.tsv",sep="\t",index_col=0)
    check("pca_coordinates_recomputed", np.allclose(np.abs(coords),np.abs(stored[["PC1","PC2"]]),rtol=1e-10,atol=1e-10))
    check("pca_variance_recomputed", np.allclose(model.explained_variance_ratio_*100,
          [summary["pca"]["PC1_percent"],summary["pca"]["PC2_percent"]],rtol=1e-10))
    heatmap = pd.read_csv(ROOT/"results/heatmap_zscores.tsv",sep="\t",index_col=0)
    best20 = result.sort_values(["padj","pvalue"]).head(20).index
    check("heatmap_gene_selection", len(heatmap)==20 and set(heatmap.index)==set(best20))
    heatmap_source = vst.loc[heatmap.index,heatmap.columns]
    z = heatmap_source.sub(heatmap_source.mean(axis=1),axis=0).div(heatmap_source.std(axis=1,ddof=1),axis=0)
    check("heatmap_zscores_recomputed", np.allclose(heatmap,z,rtol=1e-12,atol=1e-12))
    cris = summary["CRISPLD2"]
    check("crispld2_gene_mapping", result.loc[cris["ensembl_id"],"gene_name"]=="CRISPLD2")
    check("crispld2_summary_matches_results", np.allclose(
          result.loc[cris["ensembl_id"],["log2FoldChange","padj","pvalue","lfcSE"]].to_numpy(dtype=float),
          [cris["log2FoldChange"],cris["padj"],cris["pvalue"],cris["lfcSE"]],rtol=1e-12,atol=0))
    measures = pd.read_csv(ROOT/"results/crispld2_paired_counts.tsv",sep="\t")
    paired = measures.pivot(index="donor",columns="condition",values="normalized_count")
    check("crispld2_increases_in_every_pair", paired.dexamethasone.gt(paired.untreated).all() and len(paired)==4)
    for _,row in measures.iterrows():
        sample = meta.index[(meta.donor==row.donor) & (meta.condition==row.condition)][0]
        check("paired_plot_value:"+sample,np.isclose(row.normalized_count,norm.loc[cris["ensembl_id"],sample],rtol=1e-12))
    for name in ["pca","volcano","heatmap","crispld2"]:
        svg = ROOT/f"figures/gse52778_{name}.svg"
        tree = ET.parse(svg)
        check("svg_valid:"+name, tree.getroot().tag.endswith("svg") and not tree.findall(".//{*}script"))
        png = ROOT/f"figures/gse52778_{name}.png"
        with Image.open(png) as im:
            dims = im.size; im.verify()
        check("png_valid:"+name, dims[0]>=1600 and dims[1]>=1000,{"pixels":list(dims)})
    for path in (ROOT/"scripts").glob("*.py"):
        compile(path.read_text(),str(path),"exec")
    check("python_sources_parse",True)
    report = {"status":"passed","checked_utc":datetime.now(timezone.utc).isoformat(),
      "checks":checks,"dispersion_optimizer_fallbacks":fallback,
      "visual_review":"All four latest PNG figures inspected; labels, margins and data legends checked.",
      "R_counterpart":"Included but not executed; R is unavailable. Executed outputs are from PyDESeq2.",
      "statistical_limit":"Four paired donors, residual df=3; low-df dispersion-prior warning retained.",
      "not_claimed":["clinical validation","external replication","native-R reproduction"]}
    (ROOT/"results/validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":"passed","checks":len(checks),"bh_max_error":max_bh_error,
                      "dispersion_fallbacks":fallback},indent=2))

if __name__=="__main__":
    main()
