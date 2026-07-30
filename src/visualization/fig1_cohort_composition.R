#!/usr/bin/env Rscript
# fig1_cohort_composition.R — Figure 1
# Three vertical stacked-bar panels in one row:
#   A: country of origin   B: release-year bin   C: isolation source
# Run from project root: Rscript src/visualization/fig1_cohort_composition.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(tidyr)
  library(jsonlite)
  library(readr)
  library(scales)
  library(purrr)
  library(forcats)
  library(countrycode)
})

setwd("/Users/Vicky/Acinetobacter_ML_2/eskape-defence-ml")

# ── Species catalogue ─────────────────────────────────────────────────────────
SP_KEYS   <- c("abaumannii","ecloaceae","efaecium",
               "kpneumoniae","paeruginosa","saureus")
SP_LABELS <- c("A. baumannii","E. cloacae complex","E. faecium",
               "K. pneumoniae","P. aeruginosa","S. aureus")

# x-axis display labels — italic via expression hack using plotmath won't work
# cleanly in discrete axis; we handle italics with element_text face="italic"
SP_SHORT  <- c("A.\nbaumannii", "E. cloacae\ncomplex", "E.\nfaecium",
               "K.\npneumoniae", "P.\naeruginosa", "S.\naureus")
names(SP_SHORT) <- SP_LABELS

# ── Shared ggplot2 theme ──────────────────────────────────────────────────────
bar_theme <- function(legend_cols = 1) {
  theme_classic(base_size = 10) +
  theme(
    axis.line         = element_line(colour = "#555555", linewidth = 0.4),
    axis.ticks        = element_line(colour = "#555555", linewidth = 0.3),
    axis.text.x       = element_text(face = "italic", size = 8.5,
                                     lineheight = 0.9, margin = margin(t = 3)),
    axis.text.y       = element_text(size = 8),
    axis.title.y      = element_text(size = 9, margin = margin(r = 5)),
    axis.title.x      = element_blank(),
    panel.grid.major.y = element_line(colour = "#e8e8e8", linewidth = 0.35),
    panel.grid.major.x = element_blank(),
    legend.position   = "bottom",
    legend.title      = element_text(size = 8, face = "bold"),
    legend.text       = element_text(size = 7.5),
    legend.key.size   = unit(0.38, "cm"),
    legend.spacing.x  = unit(0.15, "cm"),
    plot.title        = element_text(face = "bold", size = 11,
                                     margin = margin(b = 6)),
    plot.margin       = margin(6, 10, 4, 10)
  )
}

# In-bar label helper
# IMPORTANT: pass full df (not a filtered subset) so position_stack can compute
# correct cumulative positions for every segment. Small segments get "" label.
bar_label <- function(df, threshold = 5) {
  geom_text(
    data = df,
    aes(label = ifelse(pct >= threshold, paste0(round(pct), "%"), "")),
    position = position_stack(vjust = 0.5),
    colour = "white", fontface = "bold", size = 2.6
  )
}

# ── Load metadata ─────────────────────────────────────────────────────────────
all_meta <- map2_dfr(SP_KEYS, SP_LABELS, function(key, lbl) {
  d <- fromJSON(paste0("data/interim/", key, "_accession_metadata.json"))
  d$species <- lbl
  d
}) %>%
  mutate(species = factor(species, levels = SP_LABELS))

# ── PANEL A: Geographic region ────────────────────────────────────────────────
# Use countrycode continent mapping (Africa, Americas, Asia, Europe, Oceania)
CONT_LEVELS <- c("Asia", "Europe", "Americas", "Oceania", "Africa", "Unknown")
CONT_COLORS <- c(
  "Asia"     = "#e07b39",   # warm orange
  "Europe"   = "#2980b9",   # blue
  "Americas" = "#27ae60",   # green
  "Oceania"  = "#8e44ad",   # purple
  "Africa"   = "#f1c40f",   # yellow
  "Unknown"  = "#c8c8c8"    # grey
)

country_data <- all_meta %>%
  mutate(country_clean = trimws(sub(":.*", "", country))) %>%
  filter(!is.na(country_clean)) %>%
  mutate(
    iso3      = suppressWarnings(
                  countrycode(country_clean, "country.name", "iso3c")),
    continent = suppressWarnings(
                  countrycode(iso3, "iso3c", "continent")),
    continent = case_when(
      is.na(continent) ~ "Unknown",
      TRUE             ~ continent
    ),
    continent = factor(continent, levels = CONT_LEVELS)
  ) %>%
  count(species, continent) %>%
  group_by(species) %>%
  mutate(pct = 100 * n / sum(n)) %>%
  ungroup()

panel_a <- ggplot(country_data,
                  aes(x = species, y = pct, fill = continent)) +
  geom_col(width = 0.62, colour = "white", linewidth = 0.3) +
  bar_label(country_data) +
  scale_fill_manual(values = CONT_COLORS, name = "Continent",
                    drop = FALSE,
                    guide = guide_legend(title.position = "top",
                                         ncol = 2,
                                         keywidth  = unit(0.4, "cm"),
                                         keyheight = unit(0.35, "cm"))) +
  scale_x_discrete(labels = SP_SHORT) +
  scale_y_continuous(labels = percent_format(scale = 1),
                     expand  = expansion(mult = c(0, 0.02)),
                     limits  = c(0, 100)) +
  labs(title = "A", y = "Percentage of genomes") +
  bar_theme()

# ── PANEL B: Release-year distribution ───────────────────────────────────────
YEAR_LEVELS <- c("pre-2010", "2010-2015",  "2016-2020",  "2021-present")
YEAR_LABELS <- c("pre-2010", "2010-2015", "2016-2020", "2021-present")
YEAR_COLORS <- c("pre-2010"      = "#d4e6f1",
                 "2010-2015"     = "#7fb3d3",
                 "2016-2020"     = "#2980b9",
                 "2021-present"  = "#1a5276")

year_data <- all_meta %>%
  filter(!is.na(year_bin), year_bin %in% YEAR_LEVELS) %>%
  mutate(year_bin = factor(year_bin, levels = YEAR_LEVELS,
                           labels = YEAR_LABELS)) %>%
  count(species, year_bin) %>%
  group_by(species) %>%
  mutate(pct = 100 * n / sum(n)) %>%
  ungroup()

panel_b <- ggplot(year_data,
                  aes(x = species, y = pct, fill = year_bin)) +
  geom_col(width = 0.62, colour = "white", linewidth = 0.3) +
  bar_label(year_data) +
  scale_fill_manual(values = YEAR_COLORS, name = "Release year",
                    guide = guide_legend(title.position = "top",
                                         ncol = 2,
                                         keywidth  = unit(0.4, "cm"),
                                         keyheight = unit(0.35, "cm"))) +
  scale_x_discrete(labels = SP_SHORT) +
  scale_y_continuous(labels = percent_format(scale = 1),
                     expand  = expansion(mult = c(0, 0.02)),
                     limits  = c(0, 100)) +
  labs(title = "B", y = "Percentage of genomes") +
  bar_theme()

# ── PANEL C: Isolation source ─────────────────────────────────────────────────
ISO_MAP <- c(
  clinical_human      = "Clinical (human)",
  clinical_animal     = "Clinical (animal)",
  environmental_water = "Environmental",
  environmental_soil  = "Environmental",
  food                = "Food",
  other               = "Other / unspecified",
  unknown             = "Not provided",
  not_fetched         = "Not provided"
)
ISO_LEVELS <- c("Clinical (human)", "Clinical (animal)", "Environmental",
                "Food", "Other / unspecified", "Not provided")
ISO_COLORS <- c(
  "Clinical (human)"    = "#c0392b",
  "Clinical (animal)"   = "#e67e22",
  "Environmental"       = "#27ae60",
  "Food"                = "#f1c40f",
  "Other / unspecified" = "#95a5a6",
  "Not provided"        = "#dcdde1"
)

# Species name map for the CSV (full Linnaean → short label)
SP_NAME_MAP <- c(
  "Acinetobacter baumannii"      = "A. baumannii",
  "Enterobacter cloacae complex" = "E. cloacae complex",
  "Enterococcus faecium"         = "E. faecium",
  "Klebsiella pneumoniae"        = "K. pneumoniae",
  "Pseudomonas aeruginosa"       = "P. aeruginosa",
  "Staphylococcus aureus"        = "S. aureus"
)

iso_data <- read_csv("data/interim/isolation_source_all.csv",
                     show_col_types = FALSE) %>%
  mutate(
    species   = recode(species, !!!SP_NAME_MAP),
    species   = factor(species, levels = SP_LABELS),
    iso_group = recode(iso_class, !!!ISO_MAP, .default = "Not provided"),
    iso_group = factor(iso_group, levels = ISO_LEVELS)
  ) %>%
  count(species, iso_group) %>%
  group_by(species) %>%
  mutate(pct = 100 * n / sum(n)) %>%
  ungroup()

panel_c <- ggplot(iso_data,
                  aes(x = species, y = pct, fill = iso_group)) +
  geom_col(width = 0.62, colour = "white", linewidth = 0.3) +
  bar_label(iso_data) +
  scale_fill_manual(values = ISO_COLORS, name = "Isolation source",
                    guide = guide_legend(title.position = "top",
                                         ncol = 2,
                                         keywidth  = unit(0.4, "cm"),
                                         keyheight = unit(0.35, "cm"))) +
  scale_x_discrete(labels = SP_SHORT) +
  scale_y_continuous(labels = percent_format(scale = 1),
                     expand  = expansion(mult = c(0, 0.02)),
                     limits  = c(0, 100)) +
  labs(title = "C", y = "Percentage of genomes") +
  bar_theme()

# ── Assemble ──────────────────────────────────────────────────────────────────
fig1 <- panel_a | panel_b | panel_c

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir <- "results/figures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ggsave(file.path(out_dir, "fig1_cohort_composition.png"),
       fig1, width = 15, height = 7, dpi = 300, bg = "white")

pdf(file.path(out_dir, "fig1_cohort_composition.pdf"),
    width = 15, height = 7)
print(fig1)
invisible(dev.off())

cat("Saved:\n  results/figures/fig1_cohort_composition.png\n")
cat("  results/figures/fig1_cohort_composition.pdf\n")
