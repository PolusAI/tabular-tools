suppressWarnings(library(argparse))
suppressWarnings(library(logging))


# Initialize the logger
basicConfig()

# Setup the argument parsing
addFitParams <- function(parser) {
  print("addFitParams")
  parser$add_argument("--params", type = "character",help="Fit params csv file")
  invisible(NULL)
}
addBaselineCorrected <- function(parser) {
  parser$add_argument("--values", type = "character",help="Baseline corrected csv file")
  invisible(NULL)
}
addPlateMap <- function(parser) {
  parser$add_argument("--platemap", type = "character",help="platemap excel file")
  invisible(NULL)
}
addOutputArgs <- function(parser) {
  parser$add_argument("--outdir", type = "character",help="Output csv file")
  invisible(NULL)
}
getAllParser <- function() {
  parser <- ArgumentParser(description="ALL PARSER")
  addFitParams(parser)
  addBaselineCorrected(parser)
  addPlateMap(parser)
  addOutputArgs(parser)
  return(parser)
}

# Parse the arguments
parser <- getAllParser()
args <- parser$parse_args()


#Path to csvfile directory
params <- args$params
loginfo('params = %s', params)

values <- args$values
loginfo('values = %s', values)

plate_map <- args$platemap
loginfo('platemap = %s', plate_map)

outdir <- args$outdir
loginfo('outdir = %s', outdir)

loginfo('params (fit params) = %s', params)
loginfo('values (baseline corrected): %s', values)
loginfo('platemap file (plate metadata): %s', plate_map)
loginfo('outdir (output directory): %s', outdir)

source('./prepare_data.R')
source('./main_analysis.R')
