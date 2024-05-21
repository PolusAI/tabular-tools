
suppressWarnings(library(logging))


# Initialize the logger
basicConfig()

args = commandArgs(trailingOnly=TRUE)
print(args[1])
print(args[2])
print(args[3])
print(args[4])

params <- args[1]
values <- args[2]
plate_map <- args[3]
outdir <- args[4]

loginfo('params = %s', params)
loginfo('values = %s', values)
loginfo('platemap = %s', plate_map)
loginfo('outdir = %s', outdir)

loginfo('params (fit params) = %s', params)
loginfo('values (baseline corrected): %s', values)
loginfo('platemap file (plate metadata): %s', plate_map)
loginfo('outdir (output directory): %s', outdir)

source('./prepare_data.R')
source('./main_analysis.R')
