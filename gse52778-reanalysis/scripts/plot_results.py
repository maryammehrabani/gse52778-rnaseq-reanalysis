"""Make four SVG/PNG figures from executed results, without modifying results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, leaves_list

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"; OUT.mkdir(exist_ok=True)
INK = "#183D42"; TEAL = "#128578"; CORAL = "#CD6549"; BG = "#FCFBF7"
MUTED = "#61797B"; GRID = "#DDE5E1"; GREY = "#BCC9C6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
  "axes.titlesize": 20, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
  "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
  "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": BG,
  "figure.facecolor": BG, "savefig.facecolor": BG, "svg.fonttype": "none"})
s = json.loads((ROOT / "results/summary.json").read_text())
r = pd.read_csv(ROOT / "results/differential_expression.tsv.gz", sep="\t", index_col=0)
meta = pd.read_csv(ROOT / "data/samples.tsv", sep="\t", index_col=0)
donors = s["donors"]
donor_short = {d: f"D{i+1}" for i, d in enumerate(donors)}

def canvas(title, subtitle, figsize=(10, 7.4)):
    fig = plt.figure(figsize=figsize)
    fig.text(.08, .945, "GSE52778  /  PUBLIC-DATA REANALYSIS", fontsize=10.5, color=TEAL, weight="bold")
    fig.text(.08, .88, title, fontsize=24, weight="bold")
    fig.text(.08, .835, subtitle, fontsize=12, color=MUTED)
    fig.text(.08, .038, "Himes et al., 2014  ·  airway 1.32.0  ·  PyDESeq2 0.5.4", fontsize=10, color=MUTED)
    return fig

def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=.7, zorder=0)
    ax.tick_params(length=0, pad=8)
    ax.set_axisbelow(True)

def save(fig, name):
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png", dpi=180)
    plt.close(fig)

# 1. PCA: every dot is an actual sample; joining lines preserve donor pairing.
pca = pd.read_csv(ROOT / "results/pca_coordinates.tsv", sep="\t", index_col=0)
fig = canvas("Treatment response across paired donors", "8 RNA-seq samples · 4 donors · 500 most variable genes after VST")
ax = fig.add_axes([.105, .23, .77, .54]); clean(ax)
markers = ["o", "s", "D", "^"]
for i, d in enumerate(donors):
    pair = pca.loc[pca.donor.eq(d)].set_index("condition").loc[["untreated", "dexamethasone"]]
    ax.plot(pair.PC1, pair.PC2, color=GREY, lw=1.5, zorder=1)
    for condition, color in [("untreated", TEAL), ("dexamethasone", CORAL)]:
        point = pair.loc[condition]
        ax.scatter(point.PC1, point.PC2, c=color, marker=markers[i], s=115,
                   edgecolors=BG, linewidths=1.2, zorder=4)
    mid = pair[["PC1", "PC2"]].mean()
    ax.annotate(donor_short[d], (mid.PC1, mid.PC2), xytext=(0, 8), textcoords="offset points",
                 fontsize=11, ha="center", color=INK,
                 bbox=dict(boxstyle="round,pad=.2", facecolor=BG, edgecolor="none", alpha=.9))
ax.set_xlabel(f"PC1  ·  {s['pca']['PC1_percent']:.1f}% of variance")
ax.set_ylabel(f"PC2  ·  {s['pca']['PC2_percent']:.1f}% of variance")
ax.margins(x=.16, y=.20)
ax.legend(handles=[Line2D([0],[0],marker="o",color="none",markerfacecolor=TEAL,markeredgecolor="none",label="Untreated",markersize=8),
                   Line2D([0],[0],marker="o",color="none",markerfacecolor=CORAL,markeredgecolor="none",label="Dexamethasone",markersize=8)],
          loc="upper left", frameon=False, fontsize=11)
fig.text(.105, .105, "Lines connect matched donors. VST retains donor variation; PCA is descriptive QC.", fontsize=11, color=MUTED)
save(fig, "gse52778_pca")

# 2. Volcano: the two-sided Wald test targets zero LFC; effect cutoff is descriptive.
fig = canvas("A donor-adjusted expression landscape", "Wald test · dexamethasone versus untreated · model: donor + condition")
ax = fig.add_axes([.105, .24, .79, .53]); clean(ax)
valid = r.padj.notna() & np.isfinite(r.log2FoldChange)
v = r.loc[valid].copy()
v["neglog10"] = -np.log10(v.padj.clip(lower=np.finfo(float).tiny))
for label, color, alpha, size in [("not_selected",GREY,.48,7),("down",TEAL,.7,13),("up",CORAL,.7,13)]:
    subset=v.loc[v.direction.eq(label)]
    ax.scatter(subset.log2FoldChange,subset.neglog10,c=color,alpha=alpha,s=size,edgecolors="none",zorder=2)
ax.axhline(-np.log10(.05),color=MUTED,lw=.9,ls="--")
for x in [-1,1]: ax.axvline(x,color=MUTED,lw=.9,ls="--")
ax.set_xlabel("Unshrunken log₂ fold change")
ax.set_ylabel("−log₁₀(BH-adjusted P)")
ax.set_xlim(min(-7,v.log2FoldChange.min()-.5),max(7,v.log2FoldChange.max()+.5))
ax.set_ylim(-3,v.neglog10.max()*1.13)
highlight = ["CRISPLD2", "FKBP5", "TSC22D3", "KLF15"]
offsets = {"CRISPLD2": (30,10), "FKBP5": (20,15), "TSC22D3": (28,0), "KLF15": (-45,20)}
for gene in highlight:
    z=v.loc[v.gene_name.eq(gene)]
    if len(z)==1:
        q=z.iloc[0]
        ax.scatter(q.log2FoldChange,q.neglog10, s=32,edgecolor=INK, facecolor=CORAL,lw=.8,zorder=4)
        ax.annotate(gene,(q.log2FoldChange,q.neglog10),xytext=offsets[gene],textcoords="offset points",fontsize=10.5,
          color=INK,ha="left" if offsets[gene][0]>0 else "right",arrowprops=dict(arrowstyle="-",color=MUTED,lw=.7),
          bbox=dict(boxstyle="round,pad=.15",fc=BG,ec="none",alpha=.9))
ax.text(.025,.95,f"{s['down']:,} DOWN", transform=ax.transAxes,fontsize=13,weight="bold",color=TEAL,va="top")
ax.text(.975,.95,f"{s['up']:,} UP", transform=ax.transAxes,fontsize=13,weight="bold",color=CORAL,va="top",ha="right")
fig.text(.105,.115, "Colored: BH FDR < 0.05 and |log₂FC| ≥ 1 (posthoc effect filter).",fontsize=11,color=MUTED)
fig.text(.105,.083,f"{s['retained_genes']:,} genes tested after count filtering; {s['fdr_significant']:,} meet FDR < 0.05 alone.",fontsize=11,color=MUTED)
save(fig,"gse52778_volcano")

# 3. Display the twenty smallest adjusted p-values, then cluster those rows only.
vst=pd.read_csv(ROOT/"results/vst.tsv.gz",sep="\t",index_col=0)
top=r.dropna(subset=["padj"]).sort_values(["padj","pvalue"]).head(20)
sample_order=[]
for condition in ["untreated","dexamethasone"]:
    sample_order.extend(meta.loc[meta.condition.eq(condition)].reset_index().set_index("donor").loc[donors,"sample_id"].tolist())
values=vst.loc[top.index,sample_order]
z=(values.to_numpy()-values.to_numpy().mean(axis=1,keepdims=True))/values.to_numpy().std(axis=1,ddof=1,keepdims=True)
row_order=leaves_list(linkage(z,method="average",metric="euclidean"))
z=z[row_order]; labels=top.iloc[row_order].gene_name.fillna(pd.Series(top.index,index=top.index)).tolist()
fig=canvas("The strongest expression changes", "Top 20 genes by BH-adjusted P · VST expression as row z-scores",figsize=(10,9.1))
ax=fig.add_axes([.22,.19,.58,.56])
cmap=LinearSegmentedColormap.from_list("teal_cream_coral",[TEAL,"#FCFBF7",CORAL])
lim=max(2,float(np.abs(z).max()))
im=ax.imshow(z,aspect="auto",cmap=cmap,vmin=-lim,vmax=lim,interpolation="nearest")
ax.set_yticks(range(20),labels,fontsize=11)
ax.set_xticks(range(8),[donor_short[d] for d in donors]*2,fontsize=12)
ax.tick_params(length=0,pad=9)
for spine in ax.spines.values():spine.set_visible(False)
ax.axvline(3.5,color=BG,lw=5)
ax.text(1.5,-1.45,"Untreated",color=TEAL,weight="bold",ha="center",fontsize=12)
ax.text(5.5,-1.45,"Dexamethasone",color=CORAL,weight="bold",ha="center",fontsize=12)
cax=fig.add_axes([.835,.36,.022,.22]);bar=fig.colorbar(im,cax=cax)
bar.set_label("Row z-score",fontsize=11,labelpad=9);bar.outline.set_visible(False);bar.ax.tick_params(length=0,labelsize=10)
fig.text(.22,.12,"Same donor order in both conditions; color shows relative expression.",fontsize=11,color=MUTED)
fig.text(.22,.088,"Genes selected in this dataset; this heatmap is not independent validation.",fontsize=10.5,color=MUTED)
save(fig,"gse52778_heatmap")
pd.DataFrame(z,index=top.iloc[row_order].index,columns=sample_order).to_csv(ROOT/"results/heatmap_zscores.tsv",sep="\t")

# 4. All paired measurements for CRISPLD2, using verified package annotation.
norm=pd.read_csv(ROOT/"results/normalized_counts.tsv.gz",sep="\t",index_col=0)
target=s["CRISPLD2"];assert r.loc[target["ensembl_id"],"gene_name"]=="CRISPLD2"
fig=canvas("CRISPLD2: a consistent paired response", "A gene-level view of the paired dexamethasone response")
ax=fig.add_axes([.105,.21,.52,.54]);clean(ax)
measures=[]
label_targets=[]
for i,d in enumerate(donors):
    # Recover sample IDs explicitly after assigning condition as the row index.
    sample_ids=meta.loc[meta.donor.eq(d)].reset_index().set_index("condition").loc[["untreated","dexamethasone"],"sample_id"].tolist()
    y=norm.loc[target["ensembl_id"],sample_ids].to_numpy()
    assert y[1]>y[0]
    ax.plot([0,1],y,color=GREY,lw=1.7,zorder=1)
    ax.scatter([0,1],y,c=[TEAL,CORAL],s=110,marker=markers[i],edgecolors=BG,lw=1.2,zorder=4)
    label_targets.append((float(y[1]),donor_short[d]))
    for j,c in enumerate(["untreated","dexamethasone"]):measures.append({"donor":d,"condition":c,"normalized_count":float(y[j])})
ax.set_xticks([0,1],["Untreated","Dexamethasone"],fontsize=12)
max_y=max(y for y,_ in label_targets);previous=-np.inf
for y,label in sorted(label_targets):
    label_y=max(y,previous+max_y*.065);previous=label_y
    ax.annotate(label,(1,y),xytext=(1.12,label_y),textcoords="data",va="center",fontsize=11,
                arrowprops=dict(arrowstyle="-",color=MUTED,lw=.7))
ax.set_xlim(-.25,1.4);ax.set_ylim(0,max_y*1.18);ax.set_ylabel("Size-factor-normalized counts")
fig.text(.70,.67,f"{target['fold_change']:.2f}×",fontsize=37,weight="bold",color=CORAL)
fig.text(.70,.617,"model-estimated increase",fontsize=10.5,color=MUTED)
fig.text(.70,.535,f"log₂FC  +{target['log2FoldChange']:.2f}",fontsize=15,weight="bold")
fig.text(.70,.48,f"BH FDR  {target['padj']:.2e}",fontsize=11)
fig.text(.70,.42,"Paired model\n4 independent donors",fontsize=11,color=MUTED,linespacing=1.6)
fig.text(.105,.12,"Every point is a measured sample. Lines connect the same donor.",fontsize=11,color=MUTED)
fig.text(.105,.083,f"{target['ensembl_id']} · Gene identity from the airway package annotation.",fontsize=10.5,color=MUTED)
save(fig,"gse52778_crispld2")
pd.DataFrame(measures).to_csv(ROOT/"results/crispld2_paired_counts.tsv",sep="\t",index=False)

captions={
"gse52778_pca":"PCA of design-informed VST values for the 500 most variable retained genes. Each point is one sample; lines connect matched donors. Colors distinguish untreated and dexamethasone-treated cells. Donor variation is retained. PCA is descriptive QC.",
"gse52778_volcano":"Dexamethasone versus untreated in a paired model (~ donor + condition), using PyDESeq2 0.5.4. The y-axis is -log10 BH-adjusted P. Colored genes satisfy BH FDR < 0.05 and an additional descriptive |unshrunken log2FC| >= 1 cutoff; the Wald test itself tests log2FC = 0.",
"gse52778_heatmap":"The 20 genes with the smallest BH-adjusted P-values, ordered by average-linkage clustering of row-standardized VST expression. Columns preserve the same donor order in the two treatment groups. This display uses genes selected from the same dataset and is not external validation.",
"gse52778_crispld2":"All eight CRISPLD2 measurements displayed as size-factor-normalized counts, with lines connecting the four donor pairs. The reported effect and BH-adjusted P come from the genome-wide paired negative-binomial model, not a separate four-pair t-test. CRISPLD2 is a previously reported glucocorticoid-responsive gene in the original study."}
(OUT/"captions.json").write_text(json.dumps({"figures":captions,"donor_labels":donor_short},indent=2)+"\n")
print("Created four SVG/PNG figure pairs and captions.")
