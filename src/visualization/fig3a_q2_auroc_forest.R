# Fig 2A: Q2 AUROC forest plot (per-species high-ARG classifier)
# Reads results/q2_367_results.json directly — no Python export needed.
# Output: results/figures/q2/fig2a_q2_auroc_forest.{png,pdf}

library(ggplot2)
library(dplyr)
library(jsonlite)

source("src/visualization/eskape_theme.R")

raw <- fromJSON("results/q2_367_results.json")

SP_KEYS  <- c("abaumannii", "ecloaceae", "efaecium", "kpneumoniae", "paeruginosa", "saureus")
SP_ABBR  <- c("AB", "EC", "EF", "KP", "PA", "SA")
SP_FULL  <- c("A. baumannii", "E. cloacae", "E. faecium",
              "K. pneumoniae", "P. aeruginosa", "S. aureus")

df <- data.frame(
  species  = SP_ABBR,
  full     = SP_FULL,
  auroc    = sapply(SP_KEYS, function(k) raw[[k]]$mean_auroc),
  ci_lo    = sapply(SP_KEYS, function(k) raw[[k]]$ci95_auroc[[1]]),
  ci_hi    = sapply(SP_KEYS, function(k) raw[[k]]$ci95_auroc[[2]]),
  p_adj    = sapply(SP_KEYS, function(k) raw[[k]]$p_adj),
  stringsAsFactors = FALSE
) %>%
  mutate(
    sig   = ifelse(p_adj < 0.05, "sig", "ns"),
    color = SIG_COLORS[sig],
    full  = factor(full, levels = full[order(auroc)])
  )

p <- ggplot(df, aes(x = auroc, y = full, color = sig, shape = sig)) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "grey50", linewidth = 0.5) +
  geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi), height = 0.18, linewidth = 0.7) +
  geom_point(size = 3, fill = "white", stroke = 1.5) +
  geom_text(aes(label = sprintf("%.3f", auroc)), nudge_y = 0.35, size = 2.5,
            color = "grey30") +
  scale_color_manual(values = SIG_COLORS,
                     labels = c(sig = "p_adj < 0.05", ns = "Not significant"),
                     name   = NULL) +
  scale_shape_manual(values = c(sig = 21, ns = 22),
                     labels = c(sig = "p_adj < 0.05", ns = "Not significant"),
                     name   = NULL) +
  scale_x_continuous(limits = c(0.4, 1.0), breaks = seq(0.4, 1.0, 0.1)) +
  scale_y_discrete(labels = function(x) parse(text = paste0("italic('", x, "')"))) +
  labs(x = "AUROC (95% CI)", y = NULL, tag = "A") +
  theme_eskape(base_size = 9) +
  theme(
    panel.grid.major.y = element_blank(),
    legend.position    = c(0.78, 0.15),
    legend.background  = element_rect(fill = "white", color = NA)
  )

dir.create("results/figures/q2", recursive = TRUE, showWarnings = FALSE)
save_fig(p, "results/figures/q2/fig2a_q2_auroc_forest", width = 6.5, height = 3.8)
