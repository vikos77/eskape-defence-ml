# Fig 2B: Q2 within-phylogroup-robust driver dotplot (4 significant species)
# Reads results/q2_367_results.json directly — no Python export needed.
# Output: results/figures/q2/fig2b_q2_driver_dotplot.{png,pdf}

library(ggplot2)
library(dplyr)
library(jsonlite)

source("src/visualization/eskape_theme.R")

raw <- fromJSON("results/q2_367_results.json")

SIG_SPECIES <- c("ecloaceae", "efaecium", "kpneumoniae", "paeruginosa")
SIG_ABBR    <- c("EC", "EF", "KP", "PA")
SIG_FULL    <- c("E. cloacae", "E. faecium", "K. pneumoniae", "P. aeruginosa")
MAX_FEATS   <- 8

strip_prefix <- function(x) {
  x <- sub("^dp_", "", x)
  x <- sub("^(df|padloc)_", "", x)
  x
}

all_rows <- list()
for (i in seq_along(SIG_SPECIES)) {
  sp_key  <- SIG_SPECIES[[i]]
  sp_abbr <- SIG_ABBR[[i]]
  d       <- raw[[sp_key]]$top_drivers
  d       <- d[seq_len(min(MAX_FEATS, nrow(d))), ]

  all_rows <- c(all_rows, list(data.frame(
    species  = sp_abbr,
    sp_full  = SIG_FULL[[i]],
    feature  = d$feature,
    fl       = strip_prefix(d$feature),
    rho      = d$rho_within_pg,
    named    = as.logical(d$is_named),
    survives = as.logical(d$survives_robustness),
    rank     = seq_len(nrow(d)),
    stringsAsFactors = FALSE
  )))
}

plot_df <- bind_rows(all_rows) %>%
  mutate(
    category  = ifelse(named, "named", "uncharacterised"),
    sp_full   = factor(sp_full, levels = SIG_FULL),
    # Combined key preserves per-panel rank ordering; labels() decodes back to fl
    key       = paste(species, formatC(rank, width = 2, flag = "0"), sep = "~~")
  )

# Named lookup: combined key -> clean feature label, for scale_y_discrete
label_map <- setNames(plot_df$fl, plot_df$key)
# Factor levels: within each panel, rank 1 = highest (top), so reverse
key_levels <- rev(unique(plot_df$key[order(plot_df$species, plot_df$rank)]))
plot_df$key <- factor(plot_df$key, levels = key_levels)

p <- ggplot(plot_df, aes(x = rho, y = key, color = category, shape = survives)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.5) +
  geom_segment(aes(x = 0, xend = rho, y = key, yend = key),
               linewidth = 0.7, alpha = 0.55) +
  geom_point(aes(fill = ifelse(survives, category, "white")),
             size = 2.8, stroke = 1.3) +
  facet_wrap(~ sp_full, scales = "free_y", ncol = 2) +
  scale_color_manual(values = DRIVER_COLORS,
                     labels = c(named = "Named system", uncharacterised = "Uncharacterised"),
                     name   = "System type") +
  scale_shape_manual(values = c("TRUE" = 21, "FALSE" = 24),
                     labels = c("TRUE" = "Survives phylo. correction",
                                "FALSE" = "Attenuated"),
                     name   = "Robustness") +
  scale_fill_manual(values = c(named = "#0072B2", uncharacterised = "#E69F00",
                                white = "white"), guide = "none") +
  scale_y_discrete(labels = function(x) label_map[x]) +
  scale_x_continuous(breaks = c(-0.3, -0.15, 0, 0.15, 0.3, 0.45)) +
  labs(x = expression(rho ~ "(within phylogroup)"), y = NULL, tag = "B") +
  theme_eskape(base_size = 8.5) +
  theme(
    strip.text          = element_text(face = "italic", size = 8),
    panel.grid.major.y  = element_blank(),
    legend.position     = "bottom",
    legend.box          = "horizontal",
    legend.margin       = margin(t = 0)
  )

dir.create("results/figures/q2", recursive = TRUE, showWarnings = FALSE)
save_fig(p, "results/figures/q2/fig2b_q2_driver_dotplot", width = 9, height = 7)
