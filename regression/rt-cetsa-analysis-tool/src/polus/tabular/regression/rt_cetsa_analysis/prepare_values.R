suppressWarnings(library(logging))
library(tidyverse)

loginfo('loading moltenprot values from : %s', values)


exp_curve_all <- read_csv(values,
show_col_types = FALSE
)

# same, row, col grid.
col_by_row <-
  expand.grid(row = sprintf('%.2d', 1:16), col = sprintf('%.2d', 1:24)) %>%
  arrange(., row)


exp_curve_all <- exp_curve_all %>%
# add prefix
mutate(., Temperature = paste('val_t_', Temperature, sep = ''))

exp_curve_all <- exp_curve_all %>%
  # pivot transform columns into row combinations
  pivot_longer(cols = 2:ncol(exp_curve_all)) %>%
  # pivoting again to get temperature as columns
  pivot_wider(names_from = Temperature) %>%
  # create a id column called well
  rownames_to_column() %>% rename('well' = 'rowname') %>%
  # add the grid coordinates
  bind_cols(col_by_row) %>%
  # remove all unused cols
  dplyr::select(-c('name', 'well', 'row', 'col'))

add_tempheaders <- function(df,
                            start_temp = 37,
                            end_temp = 90) {
  # generate temperature intervals
  temperature_df <-
    seq(start_temp, end_temp, by = ((end_temp - start_temp) / (ncol(df) - 1))) %>%
    round(., digits = 1)
  # rewrite all temperatures!
  # TODO CHECK that, that's quite sketchy. Should convert existing temp
  for (i in 1:ncol(df)) {
    colnames(df)[i] <- paste('t_', temperature_df[i], sep = '')
  }
  message('Temperature assignments changed for ',
          ncol(df),
          ' points.')
  return(df)
}

# NOTE: HARDCODED VALUE INTERVAL
# Also, this should comes from moltenprot
start_temp = 37
end_temp = 90

curve_df <- add_tempheaders(exp_curve_all, start_temp, end_temp)
message('Done preparing fit curves.')
