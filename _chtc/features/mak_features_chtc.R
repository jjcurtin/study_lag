# Script to engineer EMA features on CHTC for lag study
# Version 1 - uses features from v5 of ema study
# Version 2 - uses features from v5 of ema study but with 6 hour exclusions instead of 24 hour

# Constants: EDIT
# MUST EDIT IN mak_jobs as well
lead_hours <- 24  # considering 0, 24, 72, 168 (1 week), and 336(2 weeks)
version <- "v2" 

period_durations_morning <- c(48, 72, 168) # feature duration window for items 8-10
period_durations_later <- c(12, 24, 48, 72, 168) # feature duration window for items 2-7 

suppressWarnings(suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(lubridate)
  library(foreach)
  library(tidyr) # for pivot_wider
  source("fun_features.R")
}))

# get chtc process num
args <- commandArgs(trailingOnly = TRUE) 

# for testing
# job_start <- 1
# job_stop <- 10
job_start <- as.numeric(args[1])
job_stop <- as.numeric(args[2])

# read in ema
ema <- read_csv("ema.csv", show_col_types = FALSE) %>% 
  mutate(dttm_obs = with_tz(dttm_obs, tz = "America/Chicago"),
         ema_2 = ema_2,  
         ema_3 = ema_3,
         ema_4 = ema_4,
         ema_5 = ema_5)  %>%   
  select(-ema_1) %>%   # not using ema_1.  Info is in lapse df
  arrange(subid, dttm_obs)

# hack b.c I couldnt get data_type_col_name working with ema_long
ema_count <- ema %>% 
  mutate(count = if_else(is.na(ema_7), NA_character_, "ema")) %>% 
  select(dttm_obs, subid, count)

# pivot longer to allow feature function to loop over EMA items across rows
ema_long <- ema %>% 
  pivot_longer(
    cols = starts_with("ema_"),
    names_to = "ema_num",
    values_to = "response")

lapses <- read_csv("lapses.csv", show_col_types = FALSE) %>% 
  mutate(dttm_obs = with_tz(dttm_obs, tz = "America/Chicago"))
  
labels <- read_csv("labels.csv", show_col_types = FALSE) %>% 
  mutate(dttm_label = with_tz(dttm_label, tz = "America/Chicago")) %>% 
  slice(job_start:job_stop) %>% 
  mutate(label_num = seq(job_start, job_stop, by = 1))

dates <- read_csv("study_dates.csv", show_col_types = FALSE) %>% 
  select(subid, data_start = study_start) %>% 
  mutate(data_start = with_tz(data_start, tz = "America/Chicago"))

demos <- read_csv("screen.csv", show_col_types = FALSE) %>% 
  select(subid,
         demo_age = dem_1,
         demo_sex = dem_2,
         dem_3, dem_4,
         demo_educ = dem_5,
         dem_6,
         demo_income = dem_7,
         demo_marital = dem_8) %>% 
  mutate(demo_race = if_else(str_detect(dem_3, "White/Caucasian"), "White/Caucasian", "Other"),
         demo_race = if_else(str_detect(dem_4, "Yes"), "Other", demo_race),
         demo_educ = str_replace(demo_educ, "High school or GED", "High school or less"),
         demo_educ = str_replace(demo_educ, "Less than high school or GED degree", "High school or less"),
         demo_educ = str_replace(demo_educ, "2-Year degree", "Some college"),
         demo_educ = str_replace(demo_educ, "College degree", "College or more"),
         demo_educ = str_replace(demo_educ, "Advanced degree", "College or more"),
         demo_employ = if_else(str_detect(dem_6, "Employed"), "Employed", "Unemployed"),
         demo_marital = str_replace(demo_marital, "Divorced", "Other"),
         demo_marital = str_replace(demo_marital, "Widowed", "Other"),
         demo_marital = str_replace(demo_marital, "Separated", "Other")) %>% 
  select(-dem_3, -dem_4, -dem_6)

# make features ------------------
# i_label <- 1   # for testing
features <- foreach (i_label = 1:nrow(labels), .combine = "rbind") %do% {
  # message(i_label)  # for testing
  label <- labels %>% slice(i_label)
  subid <- label$subid 
  dttm_label <-  label$dttm_label
  the_label_num <- label$label_num
  
  # day of lapse label
  feature_row <- score_label_day(the_subid = subid, 
                                 the_dttm_label = dttm_label, 
                                 the_tz = "America/Chicago")
  
  # hour of lapse label
  feature_row <- feature_row %>%   
    full_join(score_label_hour(the_subid = subid, 
                               the_dttm_label = dttm_label, 
                               levels = 2, 
                               the_tz = "America/Chicago"),
              by = c("subid", "dttm_label"))
  
  # rate of previous lapses
  feature_row <- feature_row %>%   
    full_join(score_ratecount_value(the_subid = subid, 
                                    the_dttm_label = dttm_label, 
                                    x_all = lapses, 
                                    period_durations  = period_durations_later, # use all durations
                                    lead = lead_hours, 
                                    data_start = dates, 
                                    col_name = "count", 
                                    col_values = "lapse"),
              by = c("subid", "dttm_label"))
  
  
  # rate of previous emas completed,  made count item in ema_count df and set to "ema" when ema was completed through ema_7
  feature_row <- feature_row %>%
    full_join(score_ratecount_value(the_subid = subid,
                                    the_dttm_label = dttm_label,
                                    x_all = ema_count,
                                    period_durations  = period_durations_later, # use all durations
                                    lead = lead_hours,
                                    data_start = dates,
                                    col_name = "count",
                                    col_values = "ema"),
              by = c("subid", "dttm_label"))
  
  # most recent score
  feature_row <- feature_row %>%
    full_join(score_most_recent(the_subid = subid,
                           the_dttm_label = dttm_label,
                           x_all  = ema_long,
                           lead = lead_hours,
                           data_start = dates,
                           col_name = "response",
                           data_type_col_name = "ema_num",
                           data_type_values = str_c("ema_", 2:10)),
              by = c("subid", "dttm_label"))

  # median; use because may be more stable than mean
  # ema_1 not included b/c handled by lapses above
  feature_row <- feature_row %>%
    full_join(score_median(the_subid = subid,
                           the_dttm_label = dttm_label,
                           x_all  = ema_long,
                           period_durations = period_durations_morning,  # use only longer durations for 1x items
                           lead = lead_hours,
                           data_start = dates,
                           col_name = "response",
                           data_type_col_name = "ema_num",
                           data_type_values = str_c("ema_", 8:10)),
              by = c("subid", "dttm_label"))
  
  feature_row <- feature_row %>%
    full_join(score_median(the_subid = subid,
                           the_dttm_label = dttm_label,
                           x_all  = ema_long,
                           period_durations = period_durations_later,
                           lead = lead_hours,
                           data_start = dates,
                           col_name = "response",
                           data_type_col_name = "ema_num",
                           data_type_values = str_c("ema_", 2:7)),  # use all durations for 4x items
              by = c("subid", "dttm_label"))

  # max
  # ema_1 not included b/c handled by lapses above
  feature_row <- feature_row %>%
    full_join(score_max(the_subid = subid,
                        the_dttm_label = dttm_label,
                        x_all  = ema_long,
                        period_durations = period_durations_morning,
                        lead = lead_hours,
                        data_start = dates,
                        col_name = "response",
                        data_type_col_name = "ema_num",
                        data_type_values = str_c("ema_", 8:10)),
              by = c("subid", "dttm_label"))
  
  feature_row <- feature_row %>%
    full_join(score_max(the_subid = subid,
                        the_dttm_label = dttm_label,
                        x_all  = ema_long,
                        period_durations = period_durations_later,
                        lead = lead_hours,
                        data_start = dates,
                        col_name = "response",
                        data_type_col_name = "ema_num",
                        data_type_values = str_c("ema_", 2:7)),
              by = c("subid", "dttm_label"))

  # min
  # ema_1 not included b/c handled by lapses above
  feature_row <- feature_row %>%
    full_join(score_min(the_subid = subid,
                        the_dttm_label = dttm_label,
                        x_all  = ema_long,
                        period_durations = period_durations_morning,
                        lead = lead_hours,
                        data_start = dates,
                        col_name = "response",
                        data_type_col_name = "ema_num",
                        data_type_values = str_c("ema_", 8:10)),
              by = c("subid", "dttm_label"))
  
  feature_row <- feature_row %>%
    full_join(score_min(the_subid = subid,
                        the_dttm_label = dttm_label,
                        x_all  = ema_long,
                        period_durations = period_durations_later,
                        lead = lead_hours,
                        data_start = dates,
                        col_name = "response",
                        data_type_col_name = "ema_num",
                        data_type_values = str_c("ema_", 2:7)),
              by = c("subid", "dttm_label"))
  
  feature_row <- feature_row %>% 
    mutate(label_num = the_label_num)
  
  feature_row
}

# Add demos
features <- features %>% 
  left_join(demos, by = "subid")

# Add outcome label and other info to features
features %>%
  mutate(lapse = labels$lapse) %>% 
  relocate(label_num, subid, dttm_label, lapse) %>% 
  write_csv(str_c("features_", lead_hours, "_", version, "_", 
                    job_start, "_", job_stop, ".csv"))
