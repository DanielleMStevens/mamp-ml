# Load required packages
library(tidyverse)
library(Peptides)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("   Example: Rscript 06_scripts_ml/05_chemical_conversion.R bulkiness test.csv\n",
       "   Available features: bulkiness, charge, hydrophobicity, all")
}

# --- Define Available Features ---
selected_features <- args

# Read the training data
data <- read.csv(paste("test_data_set/", args[2], sep = ""))

# Function to convert sequence to bulkiness values
sequence_to_bulkiness <- function(sequence) {
  amino_acids <- strsplit(sequence, "")[[1]]
  bulkiness_data <- c(A = 11.50, R = 14.28, N = 12.82, D = 11.68, C = 13.46, Q = 14.45, E = 13.57, G = 3.40, H = 13.69, 
      I = 21.40, L = 21.40, K = 15.71, M = 16.25, F = 19.80, P = 17.43, S = 9.47, T = 15.77, W = 21.67, Y = 18.03, V = 21.57)
  bulkiness_values <- sapply(amino_acids, function(aa) {
  if (aa %in% names(bulkiness_data)) {return(bulkiness_data[aa])}})  
  return(paste(bulkiness_values, collapse = ","))
}

# Function to convert sequence to charge values
sequence_to_charge <- function(sequence) {
  amino_acids <- strsplit(sequence, "")[[1]]
  charge_data <- c(A = 0, R = 1, N = 0, D = -1, C = 0, Q = 0, E = -1, G = 0, H = 0.1,
   I = 0, L = 0, K = 1, M = 0, F = 0, P = 0, S = 0, T = 0, W = 0, Y = 0, V = 0)
  charge_values <- sapply(amino_acids, function(aa) {
  if (aa %in% names(charge_data)) {return(charge_data[aa])}})  
  return(paste(charge_values, collapse = ","))
}

# Function to convert sequence to charge values
sequence_to_hydrophobicity <- function(sequence) {
  amino_acids <- strsplit(sequence, "")[[1]]
  hydro_data <- c(A = 0.61, R = 0.00, N = 0.06, D = 0.06, C = 1.07, Q = 0.00, E = 0.01, G = 0.74, H = 0.61, 
      I = 2.22, L = 1.53, K = 0.28, M = 1.18, F = 2.02,P = 1.95, S = 0.46, T = 0.45, W = 2.65, Y = 1.88, V = 1.32)
  hydro_values <- sapply(amino_acids, function(aa) {
  if (aa %in% names(hydro_data)) {return(hydro_data[aa])}})  
  return(paste(hydro_values, collapse = ","))
}

# determine which chemical features to convert from sequence
if (args[1] == "bulkiness") {
  # Convert both Sequence and Receptor.Sequence to bulkiness values
  data$Sequence_Bulkiness <- sapply(data$Sequence, sequence_to_bulkiness)
  data$Receptor_Bulkiness <- sapply(data$Receptor.Sequence, sequence_to_bulkiness)

  # update colnames
  colnames_chemical_names <-  c("Plant species","Receptor","Locus ID/Genbank","Epitope",
  "Sequence","Known Outcome","Receptor Name","Receptor Sequence","Sequence_Bulkiness","Receptor_Bulkiness")
  colnames(data) <- colnames_chemical_names
}
if (args[1] == "charge") {
  # Convert both Sequence and Receptor.Sequence to charge values
  data$Sequence_Charge <- sapply(data$Sequence, sequence_to_charge)
  data$Receptor_Charge <- sapply(data$Receptor.Sequence, sequence_to_charge)

  colnames_chemical_names <-  c("Plant species","Receptor","Locus ID/Genbank","Epitope",
  "Sequence","Known Outcome","Receptor Name","Receptor Sequence","Sequence_Charge","Receptor_Charge")
  colnames(data) <- colnames_chemical_names
}
if (args[1] == "hydrophobicity") {
  # Convert both Sequence and Receptor.Sequence to hydrophobicity values
  data$Sequence_Hydrophobicity <- sapply(data$Sequence, sequence_to_hydrophobicity)
  data$Receptor_Hydrophobicity <- sapply(data$Receptor.Sequence, sequence_to_hydrophobicity)

  colnames_chemical_names <-  c("Plant species","Receptor","Locus ID/Genbank","Epitope",
  "Sequence","Known Outcome","Receptor Name","Receptor Sequence","Sequence_Hydrophobicity","Receptor_Hydrophobicity")
  colnames(data) <- colnames_chemical_names
}
if (args[1] == "all") {
  # Convert both Sequence and Receptor.Sequence to bulkiness values
  data$Sequence_Bulkiness <- sapply(data$Sequence, sequence_to_bulkiness)
  data$Receptor_Bulkiness <- sapply(data$Receptor.Sequence, sequence_to_bulkiness)

  # Convert both Sequence and Receptor.Sequence to charge values
  data$Sequence_Charge <- sapply(data$Sequence, sequence_to_charge)
  data$Receptor_Charge <- sapply(data$Receptor.Sequence, sequence_to_charge)

  # Convert both Sequence and Receptor.Sequence to hydrophobicity values
  data$Sequence_Hydrophobicity <- sapply(data$Sequence, sequence_to_hydrophobicity)
  data$Receptor_Hydrophobicity <- sapply(data$Receptor.Sequence, sequence_to_hydrophobicity)

  # update colnames
  colnames_chemical_names <-  c("Plant species","Receptor","Locus ID/Genbank","Epitope",
  "Sequence","Known Outcome","Receptor Name","Receptor Sequence","Sequence_Bulkiness","Receptor_Bulkiness",
  "Sequence_Charge","Receptor_Charge","Sequence_Hydrophobicity","Receptor_Hydrophobicity")
  colnames(data) <- colnames_chemical_names
}


# Save the processed data
write.csv(data, paste0("test_data_set/data_validation_", args[1], ".csv"), row.names = FALSE)
