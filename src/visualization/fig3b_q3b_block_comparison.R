# Fig 3B: Q3b feature block comparison — 11-condition restructured layout
# Three sections: individual blocks | defence + one block | all combined
# Reads results/q3b_filtered_results.json + results/q3b_367_results.json
# Output: results/figures/q3b/fig3_q3b_block_comparison.{png,pdf}

library(ggplot2)
library(dplyr)
library(jsonlite)

source("src/visualization/eskape_theme.R")

filt   <- fromJSON("results/q3b_filtered_results.json")
unfilt <- fromJSON("results/q3b_367_results.json")

# ── Section 1: individual blocks (5) ─────────────────────────────────────────
# ── Section 2: defence + one block (4) ───────────────────────────────────────
# ── Section 3: all non-defence | all five (2) ────────────────────────────────

BLOCKS <- c(
  "defence_only", "is_filt", "hmrg_filt", "arg_filt", "antidef_filt",
  "defence_is", "defence_hmrg", "defence_arg", "defence_antidef",
  "mobile_all_filt", "all_five_filt"
)
LABELS <- c(
  "Defence", "IS", "HMRG", "ARG", "Anti-\ndefence",
  "Def+IS", "Def+HMRG", "Def+ARG", "Def+\nAntiDef",
  "IS+HMRG\n+ARG\n+AntiDef", "All five"
)
COLORS <- c(
  "#0072B2",  # Defence
  "#009E73",  # IS
  "#E69F00",  # HMRG
  "#D55E00",  # ARG
  "#882255",  # Anti-defence
  "#CC79A7",  # Def+IS  (primary)
  "#56B4E9",  # Def+HMRG
  "#F0C040",  # Def+ARG
  "#AA4499",  # Def+AntiDef
  "#BBBBBB",  # IS+HMRG+ARG+AntiDef
  "#444444"   # All five
)

get_ari <- function(key) filt[[key]]$ari_primary
get_lo  <- function(key) filt[[key]]$ari_ci95[[1]]
get_hi  <- function(key) filt[[key]]$ari_ci95[[2]]

df <- data.frame(
  block = factor(LABELS, levels = LABELS),
  ari   = sapply(BLOCKS, get_ari),
  ci_lo = sapply(BLOCKS, get_lo),
  ci_hi = sapply(BLOCKS, get_hi),
  color = COLORS,
  stringsAsFactors = FALSE
)

# Ghost bars: unfiltered ARI for IS, HMRG, ARG, and Anti-defence individual blocks
ghost_df <- data.frame(
  block      = factor(c("IS", "HMRG", "ARG", "Anti-\ndefence"), levels = LABELS),
  ari_unfilt = c(unfilt$is_only$ari_primary,
                 unfilt$hmrg_only$ari_primary,
                 unfilt$arg_only$ari_primary,
                 unfilt$antidef_only$ari_primary)
)

# Background shading per section
group_rects <- data.frame(
  xmin = c(0.5, 5.5, 9.5),
  xmax = c(5.5, 9.5, 11.5),
  fill = c("#F7F7F7", "#F0ECF5", "#E8E8E8")
)

p <- ggplot(df, aes(x = block, y = ari)) +
  # section backgrounds
  geom_rect(data = group_rects,
            aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = 0.62, fill = fill),
            inherit.aes = FALSE, alpha = 0.5) +
  scale_fill_identity() +
  # ghost bars (unfiltered, dashed outline)
  geom_col(data = ghost_df,
           aes(x = block, y = ari_unfilt),
           fill = NA, color = "grey55", linewidth = 0.5, linetype = "dashed",
           width = 0.65, inherit.aes = FALSE) +
  # primary bars
  geom_col(data = df,
           aes(x = block, y = ari, fill = color),
           width = 0.65, alpha = 0.88, show.legend = FALSE) +
  # CI error bars
  geom_errorbar(data = df,
                aes(x = block, ymin = ci_lo, ymax = ci_hi),
                width = 0.18, color = "grey20", linewidth = 0.65,
                inherit.aes = FALSE) +
  # ARI value labels
  geom_text(data = df,
            aes(x = block, y = ci_hi + 0.014, label = sprintf("%.3f", ari)),
            size = 2.4, color = "grey20", inherit.aes = FALSE) +
  # ghost bar labels
  geom_text(data = ghost_df,
            aes(x = block, y = ari_unfilt + 0.014,
                label = sprintf("%.3f*", ari_unfilt)),
            size = 2.2, color = "grey50", inherit.aes = FALSE) +
  # defence baseline
  geom_hline(yintercept = filt$defence_only$ari_primary,
             linetype = "dashed", color = "#0072B2", linewidth = 0.6, alpha = 0.7) +
  annotate("text", x = 0.6, y = filt$defence_only$ari_primary + 0.022,
           label = "Defence baseline", hjust = 0, size = 2.2, color = "#0072B2") +
  # section labels
  annotate("text", x = 3.0,  y = 0.59, label = "Individual blocks",
           size = 2.5, color = "grey40", fontface = "italic") +
  annotate("text", x = 7.5,  y = 0.59, label = "Defence + one block",
           size = 2.5, color = "grey40", fontface = "italic") +
  annotate("text", x = 10.5, y = 0.59, label = "All combined",
           size = 2.5, color = "grey40", fontface = "italic") +
  scale_x_discrete(limits = LABELS) +
  scale_y_continuous(limits = c(0, 0.63), expand = c(0, 0),
                     breaks = seq(0, 0.6, 0.1)) +
  labs(
    x       = "Feature block (spec_score >= 0.70 marker filter applied to all blocks)",
    y       = "ARI vs species label (K = 6, dereplicated)",
    caption = "*Dashed outline: unfiltered ARI before near-exclusive species marker removal (IS, HMRG, ARG, Anti-defence)"
  ) +
  theme_eskape(base_size = 9) +
  theme(
    panel.grid.major.x = element_blank(),
    axis.text.x        = element_text(size = 7.5, lineheight = 1.05),
    plot.caption       = element_text(size = 7, hjust = 0, color = "grey50")
  )

dir.create("results/figures/q3b", recursive = TRUE, showWarnings = FALSE)
save_fig(p, "results/figures/q3b/fig3_q3b_block_comparison", width = 10.5, height = 4.8)
