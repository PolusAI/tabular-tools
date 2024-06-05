suppressWarnings(library(logging))
library(tidyverse)


# Initialize the logger
basicConfig()

args = commandArgs(trailingOnly=TRUE)

data <- args[1]
outdir <- args[2]

loginfo('data = %s', data)
loginfo('outdir = %s', outdir)

full_df <- read_csv(data,
show_col_types = FALSE
)

source('./main_analysis.R')
