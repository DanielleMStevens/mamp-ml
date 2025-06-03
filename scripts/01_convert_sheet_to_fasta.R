#-----------------------------------------------------------------------------------------------
# Krasileva Lab - Plant & Microbial Biology Department UC Berkeley
# Author: Danielle M. Stevens
# Last Updated: 07/06/2020
# Script Purpose: 
# Inputs: 
# Outputs: 
#-----------------------------------------------------------------------------------------------

library(readxl)
library(tidyverse)

######################################################################
#  function to turn dataframe (where one column is the name and one column is the sequence)
#   into a fasta file
######################################################################

writeFasta <- function(data, filename){
  fastaLines = c()
  for (rowNum in 1:nrow(data)){
    fastaLines = c(fastaLines, data[rowNum,1])
    fastaLines = c(fastaLines,data[rowNum,2])
  }
  fileConn<-file(filename)
  writeLines(fastaLines, fileConn)
  close(fileConn)
}

######################################################################
# filter through blast results, filter by annotation, and put into distict fasta files
######################################################################


args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("   Example: Rscript scripts/01_convert_sheet_to_fasta.R input_file.xlsx")
}

# --- load input file ---
load_data <- data.frame(readxl::read_xlsx(path = args[1], sheet = "Sheet1"))

receptor_full_length <- data.frame("Locus_Tag_Name" = character(0), "Sequence" = character(0))
for (k in 1:nrow(load_data)){
  receptor_full_length <- rbind(receptor_full_length, data.frame(
        "Locus_Tag_Name" = paste(paste(">",  load_data$plant_species[k], sep=""), 
                                 load_data$locus_id[k], 
                                 load_data$receptor[k], 
                                 sep = "|"),
        "Sequence" = load_data$receptor_sequence[k])
      )
}
  
receptor_full_length <- receptor_full_length %>% distinct(Sequence, .keep_all = TRUE)
writeFasta(receptor_full_length, "./intermediate_files/receptor_full_length.fasta")


