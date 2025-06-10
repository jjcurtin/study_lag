# Training controls for lag study

# NOTES------------------------------
# V1 features - 6 hour fence, 1 week window
# xgboost 0 lead (done)
# xgboost 24 lead (done)
# xgboost 72 lead (done)
# xgboost 168 lead (done)
# xgboost 336 lead (done)

# glmnet 0 lead (done)
# glmnet 24 lead (done)
# glmnet 72 lead (done)
# glmnet 168 lead (done)
# glmnet 336 lead (done)

# rda 0 lead (done)
# rda 24 lead (done)
# rda 72 lead (done)
# rda 168 lead (done)
# rda 336 lead (done)

# neural 0 lead (done)
# neural 24 lead (done)
# neural 72 lead (batch made)
# neural 168 lead (batch made)
# neural 336 lead (batch made)

# Xgboost V1 features - 24 hour fence, 1 week window
# xgboost 0 lead (done)
# xgboost 24 lead (done)
# xgboost 72 lead (done)
# xgboost 168 lead (done)
# xgboost 336 lead (done)

# Xgboost V3 features - 24 hour fence, 1 day window
# xgboost 0 lead (KW running)
# xgboost 24 lead (batch made)
# xgboost 72 lead (batch made)
# xgboost 168 lead (batch made)
# xgboost 336 lead (batch made)

# FORMAT PATH FUNCTION------
library(stringr)
library(dplyr)
source("https://github.com/jjcurtin/lab_support/blob/main/format_path.R?raw=true")

# SET GLOBAL PARAMETERS--------------------
study <- "lag"
window <- "1day"
lead <- 0
version <- "v3" #feature version (v1 = 24 hour fence, v2 = 6 hour fence, v3 = 1day/24 hour fence)
algorithm <- "xgboost"
model <- "strat_lh_final"

feature_set <- c("all") # EMA Features set names
data_trn <- str_c("features_", lead, "lag_", version, ".csv")  

seed_splits <- 102030

ml_mode <- "classification"   # regression or classification
configs_per_job <- 25 # number of model configurations that will be fit/evaluated within each CHTC

# RESAMPLING FOR OUTCOME-----------------------------------
# note that ratio is under_ratio, which is used by downsampling as is
# It is converted to  overratio (1/ratio) for up and smote
resample <- c("none", "up_1", "up_2", "up_3", "up_4", "up_5", 
              "down_1", "down_2", "down_3", "down_4", "down_5") # only sample 1-2 and none for week (lapse base rate = 25%)
# Note: I started testing SMOTE but the disk and memory requirements were so drastically
# different these will need to be run as separate batch IF we want to look at performance with SMOTE

# CHTC SPECIFIC CONTROLS----------------------------
username <- "kpaquette2" # for setting staging directory (until we have group staging folder)
stage_data <- FALSE # If FALSE .sif will still be staged, just not data_trn
max_idle <- 1000
request_cpus <- 1 
request_memory <- "45000MB"
request_disk <- "1600MB"
want_campus_pools <- TRUE # previously flock
want_ospool <- TRUE # previously glide

# OUTCOME-------------------------------------
y_col_name <- "lapse" 
y_level_pos <- "yes" 
y_level_neg <- "no"


# CV SETTINGS---------------------------------
cv_resample_type <- "kfold" # can be boot, kfold, or nested
cv_resample = "2_x_5" # can be repeats_x_folds (e.g., 1_x_10, 10_x_10) or number of bootstraps (e.g., 100)
cv_inner_resample <- NULL # can also be a single number for bootstrapping (i.e., 100)
cv_outer_resample <- NULL # outer resample will always be kfold
cv_group <- "subid" # set to NULL if not grouping
cv_strat <- TRUE

cv_name <- if_else(cv_resample_type == "nested",
                   str_c(cv_resample_type, "_", cv_inner_resample, "_",
                         cv_outer_resample),
                   str_c(cv_resample_type, "_", cv_resample))

# STUDY PATHS----------------------------
# the name of the batch of jobs to set folder name
name_batch <- str_c("train_", algorithm, "_", window, "_", lead, "lag_", cv_name, "_", version, "_", model) 
# the path to the batch of jobs to put the folder name
path_batch <- format_path(str_c("studydata/risk/chtc/", study, "/", name_batch)) 
# location of data set
path_data <- format_path(str_c("studydata/risk/data_processed/shared")) 

# ALGORITHM-SPECIFIC HYPERPARAMETERS-----------
hp1_glmnet <- c(0.05, seq(.1, 1, length.out = 10)) # alpha (mixture)
hp2_glmnet_min <- -8 # min for penalty grid - will be passed into exp(seq(min, max, length.out = out))
hp2_glmnet_max <- 2 # max for penalty grid
hp2_glmnet_out <- 200 # length of penalty grid

hp1_knn <- seq(5, 255, length.out = 26) # neighbors (must be integer)

hp1_rf <- c(2, 10, 20, 30, 40) # mtry (p/3 for reg or square root of p for class)
hp2_rf <- c(2, 15, 30) # min_n
hp3_rf <- 1500 # trees (10 x's number of predictors)

hp1_xgboost <- c(0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, .4)  # learn_rate
hp2_xgboost <- c(1, 2, 3, 4) # tree_depth
hp3_xgboost <- c(20, 30, 40, 50)  # mtry
# trees = 500
# early stopping = 20

hp1_rda <- seq(0.1, 1, length.out = 10)  # frac_common_cov: Fraction of the Common Covariance Matrix (0-1; 1 = LDA, 0 = QDA)
hp2_rda <- seq(0.1, 1, length.out = 10) # frac_identity: Fraction of the Identity Matrix (0-1)
 
hp1_nnet <- seq(10, 50, length.out = 5)  # epochs
hp2_nnet <- seq(0, 0.1, length.out = 15) # penalty
hp3_nnet <- seq(5, 30, length.out = 5) # hidden units

# FORMAT DATA------
format_data <- function (df){
  
  df <- df |> 
    rename(y = !!y_col_name) |> 
    # set pos class first
    mutate(y = factor(y, levels = c(!!y_level_pos, !!y_level_neg)), 
           across(where(is.character), factor)) |>
    select(-c(dttm_label)) 
  
  return(df)
}

# BUILD RECIPE---------------------------------------
# Script should have a single build_recipe function to be compatible with fit script. 
build_recipe <- function(d, config) {
  # d: (training) dataset from which to build recipe
  # job: single-row job-specific tibble
  
  # get relevant info from job (algorithm, feature_set, resample, under_ratio)
  algorithm <- config$algorithm
  
  if (config$resample == "none") {
    resample <- config$resample
  } else {
    resample <- str_split(config$resample, "_")[[1]][1]
    ratio <- as.numeric(str_split(config$resample, "_")[[1]][2])
  }
  
  # Set recipe steps generalizable to all model configurations
  rec <- recipe(y ~ ., data = d) |>
    step_rm(subid, label_num) 
  
  if(cv_strat) {
    rec <- rec |> 
      step_rm(strat) # remove strat variable
  }
  
  rec <- rec |>  # needed to retain until now for grouped CV in splits
    step_impute_median(all_numeric_predictors()) |> 
    step_impute_mode(all_nominal_predictors()) |> 
    step_dummy(all_factor_predictors()) |> 
    step_normalize(all_numeric_predictors())  |>            
    step_zv(all_predictors())  |> 
    step_select(where(~ !any(is.na(.)))) |>
    step_nzv(all_predictors())
    
  
 
  
  # resampling options for unbalanced outcome variable
  if (resample == "down") {
    rec <- rec |> 
      # ratio is equivalent to tidymodels under_ratio
      themis::step_downsample(y, under_ratio = ratio, seed = 10) 
  }
  
  
  if (resample == "smote") {
    ratio <- 1 / ratio # correct ratio to over_ratio
    rec <- rec |> 
      themis::step_smote(y, over_ratio = ratio, seed = 10) 
  }
  
  if (resample == "up") {
    ratio <- 1 / ratio # correct ratio to over_ratio
    rec <- rec |> 
      themis::step_upsample(y, over_ratio = ratio, seed = 10)
  }
  
  return(rec)
}

