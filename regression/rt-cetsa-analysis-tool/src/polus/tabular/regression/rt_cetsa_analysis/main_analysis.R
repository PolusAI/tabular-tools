# #  @@@@@@@   @@@@@@@              @@@@@@@  @@@@@@@@  @@@@@@@   @@@@@@    @@@@@@
# #  @@@@@@@@  @@@@@@@             @@@@@@@@  @@@@@@@@  @@@@@@@  @@@@@@@   @@@@@@@@
# #  @@!  @@@    @@!               !@@       @@!         @@!    !@@       @@!  @@@
# #  !@!  @!@    !@!               !@!       !@!         !@!    !@!       !@!  @!@
# #  @!@!!@!     @!!    @!@!@!@!@  !@!       @!!!:!      @!!    !!@@!!    @!@!@!@!
# #  !!@!@!      !!!    !!!@!@!!!  !!!       !!!!!:      !!!     !!@!!!   !!!@!!!!
# #  !!: :!!     !!:               :!!       !!:         !!:         !:!  !!:  !!!
# #  :!:  !:!    :!:               :!:       :!:         :!:        !:!   :!:  !:!
# #  ::   :::     ::                ::: :::   :: ::::     ::    :::: ::   ::   :::
# # NonParametric Multiparameter Analysis of CETSA/RT-CETSA Experimental Sets
# #
# # Written by: Michael Ronzetti {NIH/NCATS/UMD}
# # Patents: PCT/US21/45184, HHS E-022-2022-0-US-01
# # Main Analysis

library(tidyverse)
library(readxl)
library(stringr)
library(drc)
library(ggthemes)
library(cowplot)
library(hrbrthemes)
library(ggpubr)
library(MESS)
library(devtools)

# read in input data
full_df <- read_csv(data, show_col_types = FALSE)

# BECAUSE OF BUG
pdf(file = NULL)

# EXPERIMENTAL PARAMETERS AND SETUP
#
# Input experiment parameters here
startTemp <- 37
endTemp <- 95
plate_format <- 384
control <- 'vehicle'
pc <- 'control'

source('./functions.R')

full_df <- calculate_auc(full_df)

# Perform some preliminary control group analysis of variability
control_df <-
  control_grouping(full_df, control, pc) # Pull out control compound datapoints
control_var <-
  control_variability(control_df) # Read out the control group variability
controlPlot <-
  control_analysis(
    full_df,
    nc = 'vehicle',
    pc = 'control',
    output = 'plot',
    controlDF = control_var
  )

#Calculate melting parameter difference for each well from MoltenProt
# full_df <- calculate_meltingparams(full_df) %>%
#   calculate_zscore() %>%
#   convert_zscore

#Derive RSS values for null and alternate model for each compound from full_df
rss <- compute.rss.models(full_df, rssPlot = FALSE, drPlot = FALSE, plotModel = FALSE)

#Perform dose-response for each thermogram parameter
parameters <- compute_parameter.rssmodel(full_df, plotModel = FALSE)

#Merge these plots for further analysis
signif.df <- merge(rss, parameters)
colnames(signif.df)[9] <- 'mannwhit.pval'
signif.df <- determineSig(signif.df)
signif.df <- rankOrder(signif.df)

# Volcano plots comparing the different parameters of analysis against the NPARC RSS Difference
# Colored by significance test and whether the compound passes any.
plot_volcanos(signif.df, save = FALSE)

# Plot of RSS Differences vs. p-values for NPARC
rss.pval.plot(signif.df, savePlot = TRUE)

#Heatmap of compounds vs. different measurement styles.
parameter_heatmaps(signif.df, plotHeat = TRUE)

#Write out signif.df and full_df
write.csv(x = full_df, file = paste(outdir,'full_df.csv',sep="/"))
write.csv(x = signif.df, file = paste(outdir,'signif_df.csv',sep="/"))
