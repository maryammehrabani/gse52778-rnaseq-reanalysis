#!/usr/bin/env Rscript
# Native R counterpart. NOT EXECUTED in the authoring environment.
# Results from this script will differ numerically from the executed PyDESeq2 run.
# Run from the project root, after installing DESeq2 with BiocManager.
if (!requireNamespace("DESeq2", quietly = TRUE)) {
  stop('Install DESeq2 first: BiocManager::install("DESeq2")', call. = FALSE)
}
suppressPackageStartupMessages(library(DESeq2))
counts <- as.matrix(read.delim(gzfile("data/counts.tsv.gz"), row.names = 1,
                               check.names = FALSE))
samples <- read.delim("data/samples.tsv", row.names = 1, check.names = FALSE)
annotation <- read.delim(gzfile("data/gene_annotations.tsv.gz"), row.names = 1,
                         check.names = FALSE)
stopifnot(nrow(counts) == 63677L, ncol(counts) == 8L,
          !anyDuplicated(rownames(counts)), !anyDuplicated(colnames(counts)),
          !anyDuplicated(rownames(samples)), setequal(colnames(counts), rownames(samples)),
          all(is.finite(counts)), all(counts >= 0), all(counts == floor(counts)))
samples <- samples[colnames(counts), , drop = FALSE]
samples$donor <- factor(samples$donor)
samples$condition <- factor(samples$condition, levels = c("untreated", "dexamethasone"))
stopifnot(all(table(samples$donor, samples$condition) == 1L))
design <- model.matrix(~ donor + condition, samples)
stopifnot(qr(design)$rank == 5L, nrow(design) - ncol(design) == 3L)
keep <- rowSums(counts >= 10) >= 4
stopifnot(sum(keep) == 16139L)
dds <- DESeqDataSetFromMatrix(counts[keep, ], samples, ~ donor + condition)
dds <- DESeq(dds, test = "Wald", fitType = "parametric", sfType = "ratio",
              minReplicatesForReplace = Inf, parallel = FALSE)
res <- results(dds, contrast = c("condition", "dexamethasone", "untreated"),
                alpha = 0.05, pAdjustMethod = "BH", independentFiltering = TRUE)
out <- "results_R"
if (dir.exists(out) && length(list.files(out, all.files = TRUE, no.. = TRUE))) {
  stop("results_R is not empty; rename it before rerunning.", call. = FALSE)
}
dir.create(out, showWarnings = FALSE)
tab <- cbind(gene_id = rownames(res), as.data.frame(res),
              annotation[rownames(res), , drop = FALSE])
tab$fdr_significant <- !is.na(tab$padj) & tab$padj < 0.05
tab$selected_posthoc <- tab$fdr_significant & abs(tab$log2FoldChange) >= 1
write.table(tab[order(tab$padj, na.last = TRUE), ], file.path(out, "results.tsv"),
             sep = "\t", quote = FALSE, row.names = FALSE)
vsd <- varianceStabilizingTransformation(dds, blind = FALSE)
saveRDS(dds, file.path(out, "dds.rds")); saveRDS(vsd, file.path(out, "vst.rds"))
write.table(counts(dds, normalized = TRUE), file.path(out, "normalized_counts.tsv"),
             sep = "\t", quote = FALSE)
write.table(assay(vsd), file.path(out, "vst.tsv"), sep = "\t", quote = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
writeLines(capture.output(citation("DESeq2")), file.path(out, "citation.txt"))
message("Native R analysis complete. Review warnings and compare the recorded versions.")
