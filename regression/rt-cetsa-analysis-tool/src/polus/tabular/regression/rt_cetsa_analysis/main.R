suppressWarnings(library(argparse))
suppressWarnings(library(logging))


# Initialize the logger
basicConfig()

# Setup the argument parsing
addFitParams <- function(parser) {
  parser$add_argument("--params", type = "character",help="Fit params csv file")
  invisible(NULL)
  print("params arg ok ")
}
addBaselineCorrected <- function(parser) {
  parser$add_argument("--values", type = "character",help="Baseline corrected csv file")
  invisible(NULL)
  print("values arg ok ")
}
addPlateMap <- function(parser) {
  parser$add_argument("--platemap", type = "character",help="platemap excel file")
  invisible(NULL)
  print("platemap arg ok ")
}
addOutputArgs <- function(parser) {
  parser$add_argument("--outdir", type = "character",help="Output directory")
  invisible(NULL)
  print("outdir arg ok ")
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
loginfo('collect all parsing routines...')
parser <- getAllParser()
loginfo('parse data...')

print(parser)

tryCatch(
  args <- parser$parse_args(),
  error = function(e){
    print("There was an error: ")
    print(e)
  }
)




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

# source('./prepare_data.R')
full_df <- read_csv(data,
show_col_types = FALSE
)
source('./main_analysis.R')
