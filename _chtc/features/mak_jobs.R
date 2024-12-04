# setup jobs to make EMA features on CHTC 

# Constants: EDIT
# MUST EDIT IN mak_features_chtc.R as well
lead_hours <- 24 # considering 0, 24, 72, 168 (1 week), and 336(2 weeks)
version <- "v2" 
# Version 1 - uses features from v5 of ema study
# Version 2 - uses features from v5 of ema study but with 6 hour exclusions instead of 24 hour

# load packages
library(tidyverse)

# Paths and filenames
path_jobs <- str_c("P:/studydata/risk/chtc/lag")
name_job <- str_c("features_", lead_hours, "_", version)

path_processed <- "P:/studydata/risk/data_processed/lag" 
name_ema <- "ema.csv"
name_labels <- str_c("labels_1week_", lead_hours, "lag.csv")
name_study_dates <- "study_dates.csv"

path_shared <- "P:/studydata/risk/data_processed/shared" 
name_lapses <- "lapses.csv"

name_fun <- "fun_features.R"
name_script <- "mak_features_chtc.R"

# create new job directory (if it does not already exist) 
if (!dir.exists(here::here(path_jobs, name_job))) {
  dir.create(here::here(path_jobs, name_job))
  dir.create(here::here(path_jobs, name_job, "input"))
  dir.create(here::here(path_jobs, name_job, "output"))
} else {
  stop("Job folder already exists. No new folders created.")
}

# save out jobs csv file for queue
n_jobs <- nrow(read_csv(here::here(path_processed, name_labels), show_col_types = FALSE))
labels_per_job <- 300
job_start <- seq(1, n_jobs, by = labels_per_job) 
job_stop <- c(seq(job_start[2] - 1, n_jobs, by = labels_per_job), n_jobs)
tibble(job_start, job_stop) |> 
  write_csv(here::here(path_jobs, name_job, "input", "jobs.csv"), 
              col_names = FALSE, append = FALSE)

# select and format relevant variables and then copy ema
read_csv(here::here(path_processed, name_ema), show_col_types = FALSE) |> 
  select(-ema_type, -ema_1_1, -ema_1_2, -ema_1_3, -ema_1_4, -ema_1_5, -ema_1_6) |> 
  arrange(subid, dttm_obs) |> 
  write_csv(here::here(path_jobs, name_job, "input", "ema.csv"))

# select and format relevant variables and then copy lapses
read_csv(here::here(path_shared, name_lapses), show_col_types = FALSE) |> 
  filter(!exclude) |> 
  select(subid, dttm_obs = lapse_start) |> 
  arrange(subid, dttm_obs) |> 
  mutate(count = "lapse") |> 
  write_csv(here::here(path_jobs, name_job, "input", "lapses.csv"))

# copy over other data files verbatim
file.copy(from = here::here(path_processed, name_labels),
          to = here::here(path_jobs, name_job, "input", "labels.csv")) 
file.copy(from = here::here(path_processed, name_study_dates),
          to = here::here(path_jobs, name_job, "input", "study_dates.csv")) 
file.copy(from = here::here(path_shared, "screen.csv"),
          to = here::here(path_jobs, name_job, "input", "screen.csv")) 

# copy over function script
file.copy(from = here::here("shared", name_fun),
          to = here::here(path_jobs, name_job, "input", name_fun))

# copy R script
file.copy(from = here::here("lag", "chtc", "features", name_script),
          to = here::here(path_jobs, name_job, "input", name_script))

# copy over condor files(sh and sub)
file.copy(from = here::here("lag", "chtc", "features", "condor", "features.sh"),
          to = here::here(path_jobs, name_job, "input", "features.sh")) 
file.copy(from = here::here("lag", "chtc", "features", "condor", "features.sub"),
          to = here::here(path_jobs, name_job, "input", "features.sub")) 
