# Shared constants and theme for all ESKAPE figure scripts.
# Source at top of each figure script: source("src/visualization/eskape_theme.R")

library(ggplot2)

# Okabe-Ito colorblind-safe species palette
SP_COLORS <- c(
  AB = "#D55E00",
  EC = "#009E73",
  EF = "#CC79A7",
  KP = "#0072B2",
  PA = "#E69F00",
  SA = "#56B4E9"
)

SP_LABELS <- c(
  AB = expression(italic("A. baumannii")),
  EC = expression(italic("E. cloacae")),
  EF = expression(italic("E. faecium")),
  KP = expression(italic("K. pneumoniae")),
  PA = expression(italic("P. aeruginosa")),
  SA = expression(italic("S. aureus"))
)

SIG_COLORS <- c(sig = "#0072B2", ns = "#969696")

DRIVER_COLORS <- c(named = "#0072B2", uncharacterised = "#E69F00")

SHAP_COLORS <- c(low = "#0072B2", mid = "white", high = "#D55E00")

theme_eskape <- function(base_size = 9) {
  theme_bw(base_size = base_size) +
    theme(
      panel.grid.minor    = element_blank(),
      panel.grid.major    = element_line(color = "grey92", linewidth = 0.3),
      axis.title          = element_text(size = base_size),
      axis.text           = element_text(size = base_size - 1, color = "black"),
      strip.background    = element_rect(fill = "grey95", color = "grey70"),
      strip.text          = element_text(size = base_size, face = "italic"),
      legend.key.size     = unit(0.4, "cm"),
      legend.text         = element_text(size = base_size - 1),
      legend.title        = element_text(size = base_size - 1),
      plot.title          = element_text(size = base_size, face = "bold"),
      plot.tag            = element_text(size = base_size + 1, face = "bold")
    )
}

save_fig <- function(plot, path_noext, width, height, dpi = 300) {
  # ragg::agg_png handles Unicode (en-dashes, etc.) correctly on all platforms
  ggsave(paste0(path_noext, ".png"), plot = plot, width = width, height = height,
         units = "in", dpi = dpi, device = ragg::agg_png)
  ggsave(paste0(path_noext, ".pdf"), plot = plot, width = width, height = height,
         units = "in", device = "pdf")
  message("Saved: ", path_noext, ".png/.pdf")
}
