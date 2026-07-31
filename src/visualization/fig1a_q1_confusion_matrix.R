# Fig 1A: Q1 out-of-fold confusion matrix (row-normalised)
# Requires: results/fig1a_cm_for_r.csv, results/fig1a_stats_for_r.json
# Run export_fig_data_for_r.py first.
# Output: results/figures/rf/fig1a_q1_confusion_matrix.{png,pdf}

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(jsonlite)
library(patchwork)

source("src/visualization/eskape_theme.R")

cm_raw <- read_csv("results/fig1a_cm_for_r.csv", show_col_types = FALSE)
stats  <- fromJSON("results/fig1a_stats_for_r.json")

SP_ORDER <- c("AB", "EC", "EF", "KP", "PA", "SA")

cm_long <- cm_raw %>%
  rename(true_sp = true_species) %>%
  pivot_longer(cols = all_of(SP_ORDER), names_to = "pred_sp", values_to = "prop") %>%
  mutate(
    true_sp = factor(true_sp, levels = rev(SP_ORDER)),
    pred_sp = factor(pred_sp, levels = SP_ORDER),
    label   = sprintf("%.2f", prop),
    dark    = prop > 0.50
  )

ba_label <- sprintf("BA = %.3f [%.3f–%.3f]",
                    stats$mean_ba, stats$ci_lo, stats$ci_hi)

p <- ggplot(cm_long, aes(x = pred_sp, y = true_sp, fill = prop)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = label, color = dark), size = 2.8, fontface = "bold") +
  scale_fill_gradientn(
    colors = c("white", "#DEEBF7", "#9ECAE1", "#3182BD", "#08306B"),
    limits = c(0, 1),
    name   = "Proportion\ncorrect"
  ) +
  scale_color_manual(values = c("TRUE" = "white", "FALSE" = "grey20"), guide = "none") +
  scale_x_discrete(position = "top") +
  labs(
    x      = "Predicted species",
    y      = "True species",
    caption = ba_label,
    tag    = "A"
  ) +
  theme_eskape(base_size = 9) +
  theme(
    axis.text.x     = element_text(angle = 0, hjust = 0.5, face = "bold"),
    axis.text.y     = element_text(face = "bold"),
    panel.grid      = element_blank(),
    panel.border    = element_rect(color = "grey30", fill = NA, linewidth = 0.5),
    plot.caption    = element_text(size = 8, hjust = 0.5),
    legend.position = "right"
  )

dir.create("results/figures/rf", recursive = TRUE, showWarnings = FALSE)
save_fig(p, "results/figures/rf/fig1a_q1_confusion_matrix", width = 5, height = 4.5)
