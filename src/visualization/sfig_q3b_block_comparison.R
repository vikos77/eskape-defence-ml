# Fig 3: Q3b feature block comparison (ARI with ghost bars for unfiltered)
# Reads results/q3b_filtered_results.json + results/q3b_367_results.json directly.
# Output: results/figures/q3b/fig3_q3b_block_comparison.{png,pdf}

library(ggplot2)
library(dplyr)
library(jsonlite)

source("src/visualization/eskape_theme.R")

filt   <- fromJSON("results/q3b_filtered_results.json")
unfilt <- fromJSON("results/q3b_367_results.json")

# Blocks in display order (left to right)
BLOCKS <- c("defence_only", "is_filt", "hmrg_filt", "arg_filt", "defence_is")
LABELS <- c("Defence", "IS elements", "HMRG", "ARGs", "Defence + IS")
COLORS <- c("#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7")

df <- data.frame(
  block  = factor(LABELS, levels = LABELS),
  ari    = c(filt$defence_only$ari_primary, filt$is_filt$ari_primary,
             filt$hmrg_filt$ari_primary,    filt$arg_filt$ari_primary,
             filt$defence_is$ari_primary),
  ci_lo  = c(filt$defence_only$ari_ci95[[1]], filt$is_filt$ari_ci95[[1]],
             filt$hmrg_filt$ari_ci95[[1]],    filt$arg_filt$ari_ci95[[1]],
             filt$defence_is$ari_ci95[[1]]),
  ci_hi  = c(filt$defence_only$ari_ci95[[2]], filt$is_filt$ari_ci95[[2]],
             filt$hmrg_filt$ari_ci95[[2]],    filt$arg_filt$ari_ci95[[2]],
             filt$defence_is$ari_ci95[[2]]),
  color  = COLORS,
  stringsAsFactors = FALSE
)

# Ghost bars: unfiltered ARI for IS, HMRG, ARG blocks only
ghost_df <- data.frame(
  block      = factor(c("IS elements", "HMRG", "ARGs"), levels = LABELS),
  ari_unfilt = c(unfilt$is_only$ari_primary,
                 unfilt$hmrg_only$ari_primary,
                 unfilt$arg_only$ari_primary)
)

p <- ggplot(df, aes(x = block, y = ari, fill = block)) +
  # ghost bars (unfiltered, lighter, drawn behind)
  geom_col(data = ghost_df, aes(x = block, y = ari_unfilt),
           fill = NA, color = "grey60", linewidth = 0.6, linetype = "dashed",
           width = 0.65, inherit.aes = FALSE) +
  # primary bars
  geom_col(width = 0.65, alpha = 0.85, show.legend = FALSE) +
  # error bars (bootstrap 95% CI)
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.18,
                color = "grey20", linewidth = 0.7) +
  # ARI value labels
  geom_text(aes(label = sprintf("%.3f", ari), y = ci_hi + 0.012),
            size = 2.6, color = "grey20") +
  # ghost bar labels
  geom_text(data = ghost_df,
            aes(x = block, y = ari_unfilt + 0.012,
                label = sprintf("%.3f*", ari_unfilt)),
            size = 2.4, color = "grey55", inherit.aes = FALSE) +
  scale_fill_manual(values = setNames(COLORS, LABELS)) +
  scale_y_continuous(limits = c(0, 0.65), expand = c(0, 0),
                     breaks = seq(0, 0.6, 0.1)) +
  labs(
    x       = "Feature block (marker-filtered)",
    y       = "ARI vs species label (K = 6)",
    caption = "*Dashed outline: unfiltered ARI before near-exclusive species marker removal",
    tag     = "A"
  ) +
  theme_eskape(base_size = 9) +
  theme(
    panel.grid.major.x = element_blank(),
    plot.caption       = element_text(size = 7, hjust = 0, color = "grey50")
  )

dir.create("results/figures/q3b", recursive = TRUE, showWarnings = FALSE)
save_fig(p, "results/figures/q3b/fig3_q3b_block_comparison", width = 7.5, height = 4.5)
