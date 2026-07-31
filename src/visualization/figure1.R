# =============================================================================
# Figure 1: Q1 species classification performance
# Panels: A — confusion matrix | B — holdout recall | C — model comparison
# Output: results/figures/final/figure1.{png,pdf}
# Run from project root: Rscript src/visualization/figure1.R
# =============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(jsonlite)
library(patchwork)

source("src/visualization/eskape_theme.R")

# -- Shared data ---------------------------------------------------------------

cm_raw <- read_csv("results/fig1a_cm_for_r.csv", show_col_types = FALSE)
stats  <- fromJSON("results/fig1a_stats_for_r.json")
d      <- fromJSON("results/fig1b_holdout_for_r.json")

SP_ORDER <- c("AB", "EC", "EF", "KP", "PA", "SA")
SP_KEYS  <- c("abaumannii", "ecloaceae", "efaecium",
               "kpneumoniae", "paeruginosa", "saureus")


# =============================================================================
# Section 1: Panel A — Row-normalised confusion matrix (OOF predictions)
# =============================================================================

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

panel_a <- ggplot(cm_long, aes(x = pred_sp, y = true_sp, fill = prop)) +
  geom_tile(color = "grey72", linewidth = 0.35) +
  geom_text(aes(label = label, color = dark), size = 3.2, fontface = "bold") +
  scale_fill_gradientn(
    colors = c("white", "#DEEBF7", "#9ECAE1", "#3182BD", "#08306B"),
    limits = c(0, 1),
    name   = "Proportion\ncorrect"
  ) +
  scale_color_manual(
    values = c("TRUE" = "white", "FALSE" = "grey20"),
    guide  = "none"
  ) +
  scale_x_discrete(position = "top") +
  labs(
    x       = "Predicted species",
    y       = "True species",
    caption = ba_label,
    tag     = "A"
  ) +
  theme_eskape(base_size = 10) +
  theme(
    axis.text.x     = element_text(angle = 0, hjust = 0.5, face = "bold", size = 10),
    axis.text.y     = element_text(face = "bold", size = 10),
    axis.title      = element_text(size = 10),
    panel.grid      = element_blank(),
    panel.border    = element_blank(),
    plot.caption    = element_text(size = 9, hjust = 0.5, margin = margin(t = 6)),
    legend.position = "right"
  )


# =============================================================================
# Section 2: Panel B — Per-species recall on external holdout set (n = 180)
# =============================================================================

recall_df <- data.frame(
  species = factor(SP_ORDER, levels = SP_ORDER),
  recall  = unlist(d$recalls[SP_KEYS])
)

panel_b <- ggplot(recall_df, aes(x = species, y = recall)) +
  geom_hline(yintercept = 1.0, linetype = "dotted",
             color = "grey78", linewidth = 0.55) +
  geom_point(aes(fill = species), size = 5.5, shape = 21,
             color = "white", stroke = 0.8, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.3f", recall)),
            vjust = -1.1, size = 2.9, color = "grey20") +
  scale_fill_manual(values = SP_COLORS) +   # Okabe-Ito; colorblind-safe
  scale_y_continuous(
    limits = c(0.78, 1.10),
    breaks = c(0.80, 0.85, 0.90, 0.95, 1.00),
    expand = c(0, 0)
  ) +
  coord_cartesian(clip = "off") +
  labs(y = "Recall (holdout)", x = NULL, tag = "B") +
  theme_eskape(base_size = 10) +
  theme(
    axis.text.x        = element_text(face = "bold", size = 10),
    panel.grid.major.x = element_blank()
  )


# =============================================================================
# Section 3: Panel C — Four-classifier balanced accuracy comparison
# Values from q1_mcnemar_367.json fold BAs + cluster bootstrap CIs (Table 1).
# RF = primary model (highlighted); all McNemar tests vs RF non-significant.
# =============================================================================

model_df <- data.frame(
  model = factor(
    c("Logistic\nRegression", "LightGBM", "XGBoost", "Random\nForest"),
    levels = c("Logistic\nRegression", "LightGBM", "XGBoost", "Random\nForest")
  ),
  ba      = c(0.911, 0.909, 0.904, 0.900),
  lo      = c(0.852, 0.868, 0.861, 0.871),
  hi      = c(0.970, 0.951, 0.947, 0.923),
  primary = c(FALSE, FALSE, FALSE, TRUE)
)

panel_c <- ggplot(model_df, aes(x = ba, y = model)) +
  geom_errorbar(aes(xmin = lo, xmax = hi),
                width = 0.22, linewidth = 0.85, color = "grey45") +
  geom_point(aes(fill = primary), size = 4.8,
             shape = 21, color = "grey35", stroke = 1.2) +
  geom_text(aes(label = sprintf("%.3f", ba)),
            nudge_y = 0.3, size = 2.8, color = "grey15") +
  scale_fill_manual(
    values = c("TRUE" = "#0072B2", "FALSE" = "white"),  # Okabe-Ito blue
    guide  = "none"
  ) +
  scale_x_continuous(
    limits = c(0.82, 1.00),
    breaks = seq(0.85, 1.00, 0.05)
  ) +
  scale_y_discrete(expand = expansion(mult = 0.55)) +
  annotate("text", x = 0.823, y = 0.42,
           label    = "All McNemar vs RF: p > 0.05",
           size     = 2.5, hjust = 0,
           color    = "grey55", fontface = "italic") +
  labs(x = "Balanced accuracy (5-fold CV)", y = NULL, tag = "C") +
  theme_eskape(base_size = 10) +
  theme(
    plot.tag.position  = c(0.28, 0.98),
    panel.grid.major.y = element_blank(),
    axis.text.y        = element_text(lineheight = 0.85)
  )


# =============================================================================
# Section 4: Multipanel composition and export
# =============================================================================

figure1 <- panel_a / (panel_b + panel_c + plot_layout(widths = c(1, 1))) +
  plot_layout(heights = c(1.45, 1)) +
  plot_annotation(theme = theme(plot.margin = margin(6, 6, 6, 6)))

dir.create("results/figures/final", recursive = TRUE, showWarnings = FALSE)
save_fig(figure1, "results/figures/final/figure1", width = 7.5, height = 8.5)
