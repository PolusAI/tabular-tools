# #  @@@@@@@   @@@@@@@              @@@@@@@  @@@@@@@@  @@@@@@@   @@@@@@    @@@@@@
# #  @@@@@@@@  @@@@@@@             @@@@@@@@  @@@@@@@@  @@@@@@@  @@@@@@@   @@@@@@@@
# #  @@!  @@@    @@!               !@@       @@!         @@!    !@@       @@!  @@@
# #  !@!  @!@    !@!               !@!       !@!         !@!    !@!       !@!  @!@
# #  @!@!!@!     @!!    @!@!@!@!@  !@!       @!!!:!      @!!    !!@@!!    @!@!@!@!
# #  !!@!@!      !!!    !!!@!@!!!  !!!       !!!!!:      !!!     !!@!!!   !!!@!!!!
# #  !!: :!!     !!:               :!!       !!:         !!:         !:!  !!:  !!!
# #  :!:  !:!    :!:               :!:       :!:         :!:        !:!   :!:  !:!
# #  ::   :::     ::                ::: :::   :: ::::     ::    :::: ::   ::   :::
# # Isothermal Analysis of CETSA/RT-CETSA Experimental Sets
# #
# # Plate assignment, data cleanup, and functions
# # Patents: PCT/US21/45184, HHS E-022-2022-0-US-01

print("####### loading all functions used in the analysis...")

library(tidyverse)
library(readxl)
library(stringr)
library(drc)
library(ggthemes)
library(cowplot)
library(hrbrthemes)
library(ggpubr)
library(MESS)
library(devtools)
# load_all(".");
#' construct_grid
#' Construct a grid with compatable headers for MoltenProt file prep
#'
#' @param row_num Number of rows in microplate
#' @param col_num Number of columns in microplate
#' @param pad_num Add padding 0 to well address?
#'
#' @return df containing the grid
#' @export
#'
#' @examples
#' construct_grid(16,24)
#' construct_grid(32,48,TRUE)
construct_grid <-
  function(row_num = 16,
           col_num = 24,
           pad_num = FALSE) {
    if (pad_num == FALSE) {
      grid <-
        expand.grid(row = LETTERS[1:(row_num)], col = c(1:(col_num))) %>%
        arrange(row) %>%
        mutate(address = paste(row, col, sep = '')) %>%
        dplyr::select(-c('row', 'col'))
    } else {
      letter <- LETTERS[1:(row_num)]
      number <- c(1:(col_num))
      number <- str_pad(number, 2, pad = '0')
      col_by_row <-
        expand.grid(row = sprintf('%.2d', 1:16),
                    col = sprintf('%.2d', 1:24)) %>%
        arrange(., row)
    }
    return(grid)
  }

#' prepMatLabforMolt
#'
#' Need to provide location of file, usually in ./data/, and sheet location
#'
#' @param file_loc Location of raw Matlab data file
#' @param sheet What sheet in the .xlsx is the data located
#' @param col_names Are there column names in the data sheet
#' @param start_temp Start temp for the experiment
#' @param end_temp End temp for the experiment
#'
#' @return df containing the raw values for the RT-CETSA experiment
#' @export
#'
#' @examples
#' prepMatLabforMolt(file_loc = './data/210318_plate3.xlsx',
#' start_temp = startTemp,
#' end_temp = endTemp)
prepMatLabforMolt <- function(file_loc = './data/rtcetsa_raw.xlsx',
                              sheet = 'Sheet1',
                              col_names = FALSE,
                              start_temp = 37,
                              end_temp = 90) {
  if (file.exists(file_loc) == FALSE) {
    stop('File does not exist at path supplied.')
  }
  df <-
    read_excel(
      path = file_loc,
      sheet = sheet,
      col_names = col_names,
      .name_repair = 'unique'
    )
  if (nrow(df) == 0 || ncol(df) == 0) {
    stop('Imported file is empty. Please navigate to correct RT-CETSA file')
  }
  df <- df %>%
    dplyr::select(-c('...1', '...2')) %>%
    rownames_to_column() %>%
    rename('well' = 'rowname')

  # Construct temperature index (t_n) and pivot around the data to tidy
  tracker <- 1
  for (val in 2:ncol(df) - 1) {
    names(df)[val + 1] <- paste('t_', val, sep = '')
    tracker <- tracker + 1
  }
  df <- df %>%
    pivot_longer(., cols = 2:ncol(df)) %>%
    pivot_wider(names_from = well) %>%
    rename(., 'Temperature' = 'name') %>%
    mutate(., Temperature = as.integer(gsub("[^0-9.]", "", Temperature)))

  #Create temperature index in line with experimental parameters supplied in main script
  temperature_df <-
    seq(start_temp, end_temp, by = ((end_temp - start_temp) / (nrow(df) -
                                                                 1))) %>%
    round(., digits = 1)
  for (i in 1:length(temperature_df))
    df$Temperature[i] <- temperature_df[i]

  # Assemble data for moltenprot analysis by splitting 384-well plate to 96-well plate with appropriate index
  grid_96w <- construct_grid(row_num = 8, col_num = 12)
  q1 <- df %>%
    dplyr::select(., 1, 2:97)
  tracker <- 1
  for (val in 1:nrow(grid_96w)) {
    colnames(q1)[val + 1] <- grid_96w$address[val]
    tracker <- tracker + 1
  }
  q2 <- df %>%
    dplyr::select(., 1, 98:193)
  tracker <- 1
  for (val in 1:nrow(grid_96w)) {
    colnames(q2)[val + 1] <- grid_96w$address[val]
    tracker <- tracker + 1
  }
  q3 <- df %>%
    dplyr::select(., 1, 194:289)
  tracker <- 1
  for (val in 1:nrow(grid_96w)) {
    colnames(q3)[val + 1] <- grid_96w$address[val]
    tracker <- tracker + 1
  }
  q4 <- df %>%
    dplyr::select(., 1, 290:385)
  tracker <- 1
  for (val in 1:nrow(grid_96w)) {
    colnames(q4)[val + 1] <- grid_96w$address[val]
    tracker <- tracker + 1
  }
  write.csv(q1, './data/cleaned_expt1.csv', row.names = FALSE)
  write.csv(q2, './data/cleaned_expt2.csv', row.names = FALSE)
  write.csv(q3, './data/cleaned_expt3.csv', row.names = FALSE)
  write.csv(q4, './data/cleaned_expt4.csv', row.names = FALSE)

  return(df)
}

# Read in MoltenProt readout, with different column identities for different models
retrieveMoltenData <-
  function(model = 'standard',
           plate_format = 384) {
    # Retrieve experimental data from processed file folders
    col_by_row <-
      expand.grid(row = sprintf('%.2d', 1:16), col = sprintf('%.2d', 1:24)) %>%
      arrange(., row)
    if (model == 'standard') {
      exp1_param <-
        read_excel('./data/cleaned_expt1/Signal_resources/Signal_results.xlsx',
                   sheet = 'Fit parameters') %>%
        dplyr::select(-c('Condition'))
      exp2_param <-
        read_excel('./data/cleaned_expt2/Signal_resources/Signal_results.xlsx',
                   sheet = 'Fit parameters') %>%
        dplyr::select(-c('Condition'))
      exp3_param <-
        read_excel('./data/cleaned_expt3/Signal_resources/Signal_results.xlsx',
                   sheet = 'Fit parameters') %>%
        dplyr::select(-c('Condition'))
      exp4_param <-
        read_excel('./data/cleaned_expt4/Signal_resources/Signal_results.xlsx',
                   sheet = 'Fit parameters') %>%
        dplyr::select(-c('Condition'))
      # Reformat ID column in each exp from MoltenProt format (A1, not A01) to arrange
      exp1_param$ID <-
        gsub('([A-Z])(\\d)(?!\\d)', '\\10\\2\\3', exp1_param$ID, perl = TRUE)
      exp1_param <- exp1_param %>% arrange(ID)
      exp2_param$ID <-
        gsub('([A-Z])(\\d)(?!\\d)', '\\10\\2\\3', exp2_param$ID, perl = TRUE)
      exp2_param <- exp2_param %>% arrange(ID)
      exp3_param$ID <-
        gsub('([A-Z])(\\d)(?!\\d)', '\\10\\2\\3', exp3_param$ID, perl = TRUE)
      exp3_param <- exp3_param %>% arrange(ID)
      exp4_param$ID <-
        gsub('([A-Z])(\\d)(?!\\d)', '\\10\\2\\3', exp4_param$ID, perl = TRUE)
      exp4_param <- exp4_param %>% arrange(ID)
      # Combine all experiments and add identifiers
      exp_param_full <-
        exp1_param %>% rbind(., exp2_param, exp3_param, exp4_param) %>%
        rownames_to_column() %>% rename('well' = 'rowname') %>%
        dplyr::select(
          -c(
            'ID',
            'kN_init',
            'bN_init',
            'kU_init',
            'bU_init',
            'dHm_init',
            'Tm_init',
            'kN_fit',
            'bN_fit',
            'kU_fit',
            'bU_fit',
            'S',
            'dCp_component'
          )
        ) %>%
        bind_cols(col_by_row) %>%
        relocate(c('row', 'col'), .after = well) %>%
        dplyr::select(-'well')
      exp_param_full <- well_assignment(exp_param_full, 384)
      return(exp_param_full)
    }
    if (model == 'irrev') {
      exp1_param <-
        read_excel('./data/cleaned_expt1/Signal_resources/Signal_results.xlsx',
                   sheet = 'Fit parameters') %>%
        dplyr::select(-c('Condition'))
      exp2_param <-
        read_excel('./data/cleaned_expt2/Signal_resources/Signal_results.xlsx',
                   sheet = 'Fit parameters') %>%
        dplyr::select(-c('Condition'))
      exp3_param <-
        read_excel('./data/cleaned_expt3/Signal_resources/Signal_results.xlsx',
                   sheet = 'Fit parameters') %>%
        dplyr::select(-c('Condition'))
      exp4_param <-
        read_excel('./data/cleaned_expt4/Signal_resources/Signal_results.xlsx',
                   sheet = 'Fit parameters') %>%
        dplyr::select(-c('Condition'))
      # Reformat ID column in each exp from MoltenProt format (A1, not A01) to arrange
      exp1_param$ID <-
        gsub('([A-Z])(\\d)(?!\\d)', '\\10\\2\\3', exp1_param$ID, perl = TRUE)
      exp1_param <- exp1_param %>% arrange(ID)
      exp2_param$ID <-
        gsub('([A-Z])(\\d)(?!\\d)', '\\10\\2\\3', exp2_param$ID, perl = TRUE)
      exp2_param <- exp2_param %>% arrange(ID)
      exp3_param$ID <-
        gsub('([A-Z])(\\d)(?!\\d)', '\\10\\2\\3', exp3_param$ID, perl = TRUE)
      exp3_param <- exp3_param %>% arrange(ID)
      exp4_param$ID <-
        gsub('([A-Z])(\\d)(?!\\d)', '\\10\\2\\3', exp4_param$ID, perl = TRUE)
      exp4_param <- exp4_param %>% arrange(ID)
      # Combine all experiments and add identifiers
      exp_param_full <-
        exp1_param %>% rbind(., exp2_param, exp3_param, exp4_param) %>%
        rownames_to_column() %>% rename('well' = 'rowname') %>%
        dplyr::select(c(
          'well',
          'Ea_fit',
          'Tf_fit',
          'kN_fit',
          'bN_fit',
          'kU_fit',
          'bU_fit',
          'S'
        )) %>%
        bind_cols(col_by_row) %>%
        relocate(c('row', 'col'), .after = well) %>%
        dplyr::select(-'well')
      exp_param_full <- well_assignment(exp_param_full, 384)
      return(exp_param_full)
    }
  }

# Gather base-line corrected fit curves for the 384-well plate and pivot plate
retrieve_FittedCurves <-
  function(model = 'baseline-fit',
           start_temp = 37,
           end_temp = 90) {
    col_by_row <-
      expand.grid(row = sprintf('%.2d', 1:16), col = sprintf('%.2d', 1:24)) %>%
      arrange(., row)
    if (model == 'baseline-fit') {
      exp1_curve <-
        read_excel('./data/cleaned_expt1/Signal_resources/Signal_results.xlsx',
                   sheet = 'Baseline-corrected')
      exp2_curve <-
        read_excel('./data/cleaned_expt2/Signal_resources/Signal_results.xlsx',
                   sheet = 'Baseline-corrected') %>%
        dplyr::select(-c('Temperature'))
      exp3_curve <-
        read_excel('./data/cleaned_expt3/Signal_resources/Signal_results.xlsx',
                   sheet = 'Baseline-corrected') %>%
        dplyr::select(-c('Temperature'))
      exp4_curve <-
        read_excel('./data/cleaned_expt4/Signal_resources/Signal_results.xlsx',
                   sheet = 'Baseline-corrected') %>%
        dplyr::select(-c('Temperature'))
      exp_curve_all <-
        cbind(
          xp1 = exp1_curve,
          xp2 = exp2_curve,
          xp3 = exp3_curve,
          xp4 = exp4_curve
        ) %>%
        rename(., Temperature = xp1.Temperature) %>%
        mutate(., Temperature = paste('val_t_', Temperature, sep = ''))
      exp_curve_all <- exp_curve_all %>%
        pivot_longer(cols = 2:ncol(exp_curve_all)) %>%
        pivot_wider(names_from = Temperature) %>%
        rownames_to_column() %>% rename('well' = 'rowname') %>%
        bind_cols(col_by_row) %>%
        dplyr::select(-c('name', 'well', 'row', 'col')) %>%
        add_tempheaders(., start_temp, end_temp)
      message('Fit curves retrieved.')
      return(exp_curve_all)
    }
    if (model == 'fit_curves') {
      exp1_curve <-
        read_excel('./data/cleaned_expt1/Signal_resources/Signal_results.xlsx',
                   sheet = 'Fit curves')
      exp2_curve <-
        read_excel('./data/cleaned_expt2/Signal_resources/Signal_results.xlsx',
                   sheet = 'Fit curves') %>%
        dplyr::select(-c('Temperature'))
      exp3_curve <-
        read_excel('./data/cleaned_expt3/Signal_resources/Signal_results.xlsx',
                   sheet = 'Fit curves') %>%
        dplyr::select(-c('Temperature'))
      exp4_curve <-
        read_excel('./data/cleaned_expt4/Signal_resources/Signal_results.xlsx',
                   sheet = 'Fit curves') %>%
        dplyr::select(-c('Temperature'))
      exp_curve_all <-
        cbind(
          xp1 = exp1_curve,
          xp2 = exp2_curve,
          xp3 = exp3_curve,
          xp4 = exp4_curve
        ) %>%
        rename(., Temperature = xp1.Temperature) %>%
        mutate(., Temperature = paste('val_t_', Temperature, sep = ''))
      exp_curve_all <- exp_curve_all %>%
        pivot_longer(cols = 2:ncol(exp_curve_all)) %>%
        pivot_wider(names_from = Temperature) %>%
        rownames_to_column() %>% rename('well' = 'rowname') %>%
        bind_cols(col_by_row) %>%
        dplyr::select(-c('name', 'well', 'row', 'col')) %>%
        add_tempheaders(., start_temp, end_temp)
      message('Fit curves retrieved.')
      return(exp_curve_all)
    }
  }

# Construct full data frame with curve fit and parameters for analysis
bind_fulldf <- function(param_df, curve_df) {
  df <- cbind(param_df, curve_df)
  return(df)
}

#Convert any columns containing Kelvin values from MoltenProt to Celsius
kelToCel <- function(df) {
  df <- df %>%
    mutate(Tm_fit = Tm_fit - 273.15) %>%
    mutate(T_onset = T_onset - 273.15)
}
# #
# ISO-CETSA Functions (new)
# #

# Add temperature headers to df
add_tempheaders <- function(df,
                            start_temp = 37,
                            end_temp = 90) {
  temperature_df <-
    seq(start_temp, end_temp, by = ((end_temp - start_temp) / (ncol(df) - 1))) %>%
    round(., digits = 1)
  for (i in 1:ncol(df)) {
    colnames(df)[i] <- paste('t_', temperature_df[i], sep = '')
  }
  message('Temperature assignments changed for ',
          ncol(df),
          ' points.')
  return(df)
}

# Add row and column to a tidy dataframe (columns are each temperatures, rows are wells/conditions)
add_rowcol <- function(df, well_num) {
  if (well_num == 96) {
    col_by_row <-
      expand.grid(row = sprintf('%.2d', 1:8), col = sprintf('%.2d', 1:12)) %>%
      arrange(., row)
  }
  else if (well_num == 384) {
    col_by_row <-
      expand.grid(row = sprintf('%.2d', 1:16),
                  col = sprintf('%.2d', 1:24)) %>%
      arrange(., row)
  }
  message('Row + Column assignments created for ',
          well_num,
          '-well plate')
  df <- cbind(col_by_row, df)
  return(df)
}

# Add well assignmnets for each plate
well_assignment <- function(df, well_num) {
  if (well_num == 96) {
    letter <- LETTERS[1:8]
    number <- c(1:12)
    number <- str_pad(number, 2, pad = '0')
    tracker <- 1
    temp_df <- tibble(well = c(1:384))
    for (val in letter) {
      for (num in number) {
        temp_df$well[tracker] <- paste(val, num, sep = '')
        tracker <- tracker + 1
      }
    }
  }
  else if (well_num == 384) {
    letter <- LETTERS[1:16]
    number <- c(1:24)
    number <- str_pad(number, 2, pad = '0')
    tracker <- 1
    temp_df <- tibble(well = c(1:384))
    for (val in letter) {
      for (num in number) {
        temp_df$well[tracker] <- paste(val, num, sep = '')
        tracker <- tracker + 1
      }
    }
  }
  message('Well assignments created for ', well_num, '-well plate.')
  df <- cbind(temp_df, df)
  return(df)
}

# Assign compound ids and concentration from platemap
plate_assignment <- function(df, platemap_file) {
  id_df <- read_excel(platemap_file, sheet = 'sample') %>%
    dplyr::select(-1) %>%
    pivot_longer(., cols = 1:ncol(.)) %>%
    rename(ncgc_id = value) %>%
    dplyr::select(-c('name'))
  id_df$ncgc_id <- gsub('empty', 'vehicle', id_df$ncgc_id)
  conc_df <- read_excel(platemap_file, sheet = 'conc') %>%
    dplyr::select(-1) %>%
    pivot_longer(., cols = 1:ncol(.)) %>%
    rename(conc = value) %>%
    dplyr::select(-c('name'))
  df <- cbind(id_df, conc_df, df)
  message('Plate assignment attached to dataframe.')
  df$row <- as.numeric(df$row)
  df$col <- as.numeric(df$col)
  return(df)
}

# Calculate AUC for each well
print("####### loading calculate_auc...")
calculate_auc <- function(df) {
  #Retrieve temperatures to be used for AUC determination.
  auc.df <- df %>%
    dplyr::select(matches('t_\\d'))

  #Initialize the AUC column
  df$auc <- NA

  # Pivot and clean each row for AUC model
  for (i in 1:nrow(auc.df)) {
    curveVals <- auc.df[i,] %>%
      pivot_longer(cols = everything(),
                   names_to = 'temp',
                   values_to = 'response')
    curveVals$temp <- curveVals$temp %>%
      sub('t_', '', .)
    curveVals$temp <- as.numeric(curveVals$temp)
    df$auc[i] <- auc(x = curveVals$temp, y = curveVals$response)
  }
  message('AUC Values calculated for ', nrow(auc.df), ' wells.')
  return(df)
}

control_grouping <- function(df, control = 'DMSO', pc = 'control') {
  control_df <- filter(df, ncgc_id == control | ncgc_id == pc)
  if (nrow(control_df) == 0) {
    message('No control wells found. Review control input to function.')
  } else
    if (nrow(control_df) > 0) {
      control_df <- control_df %>%
        dplyr::select(-'conc')
      return(control_df)
    }
}

control_variability <-
  function(df, nc = 'vehicle', pc = 'control') {
    #Filter out positive and negative controls into their own df
    nc.controls.df <- df %>%
      filter(ncgc_id == nc) %>%
      dplyr::select(-c('ncgc_id', 'well', 'row', 'col'))
    pc.controls.df <- df %>%
      filter(ncgc_id == pc) %>%
      dplyr::select(-c('ncgc_id', 'well', 'row', 'col'))

    #Calculate means, sd, and %CV
    nc.mean.df <-
      apply(nc.controls.df[1:ncol(nc.controls.df)], 2, mean)
    nc.sd.df <- apply(nc.controls.df[1:ncol(nc.controls.df)], 2, sd)
    pc.mean.df <-
      apply(pc.controls.df[1:ncol(pc.controls.df)], 2, mean)
    pc.sd.df <- apply(pc.controls.df[1:ncol(pc.controls.df)], 2, sd)

    #Calculate %CV
    nc.var.df <- tibble(nc.mean = nc.mean.df, nc.sd = nc.sd.df) %>%
      mutate(nc.cv = (nc.sd / nc.mean) * 100)
    pc.var.df <- tibble(pc.mean = pc.mean.df, pc.sd = pc.sd.df) %>%
      mutate(pc.cv = (pc.sd / pc.mean) * 100)
    analysis_method <- colnames(nc.controls.df)
    var_df <- cbind(analysis_method, nc.var.df, pc.var.df)
    message('Control group variability analyzed.')
    return(var_df)
  }

# Returns thermogram with mean/sd of DMSO curve across temps
control_thermogram <- function(df, pcTm, ncTm) {
  subset_df <- subset(df, grepl('t_', analysis_method)) %>%
    mutate(temp = as.numeric(gsub('t_', '', analysis_method))) %>%
    dplyr::select(-'analysis_method')
  therm_plot <- ggplot(subset_df, aes(x = temp)) +
    geom_line(aes(y = nc.mean),
              size = 1.5,
              alpha = 0.75,
              color = '#88CCEE') +
    geom_errorbar(aes(ymin = nc.mean - nc.sd, ymax = nc.mean + nc.sd),
                  size = 0.5,
                  width = 1) +
    geom_point(
      aes(y = nc.mean),
      size = 3.25,
      shape = 21,
      color = 'black',
      fill = '#88CCEE'
    ) +
    geom_line(aes(y = pc.mean),
              size = 1.5,
              alpha = 0.75,
              color = '#882255') +
    geom_errorbar(aes(ymin = pc.mean - pc.sd, ymax = pc.mean + pc.sd),
                  size = 0.5,
                  width = 1) +
    geom_point(
      aes(y = pc.mean),
      size = 3.25,
      shape = 21,
      color = 'black',
      fill = '#EE3377'
    ) +
    theme_minimal() +
    labs(title = 'Control Thermograms',
         x = 'Temperature [C]',
         y = 'Fraction Unfolded')
  print(therm_plot)
  return(therm_plot)
}

# Controls analysis and z' output for groups
# Possible outputs:
# output = 'plot': Cowplot of controls
# output = 'df': Control dataframe
control_analysis <-
  function(df,
           nc = 'vehicle',
           pc = 'control',
           output = '',
           controlDF) {
    controls.df <- df %>%
      filter(ncgc_id == nc | ncgc_id == pc)

    #Calculate Z' from controls for each parameter
    test_params <-
      c('Tm_fit',
        'auc')
    Tm.nc.mean <-
      mean(controls.df$Tm_fit[controls.df$ncgc_id == nc])
    Tm.nc.sd <- sd(controls.df$Tm_fit[controls.df$ncgc_id == nc])
    Tm.pc.mean <-
      mean(controls.df$Tm_fit[controls.df$ncgc_id == pc])
    Tm.pc.sd <- sd(controls.df$Tm_fit[controls.df$ncgc_id == pc])
    Tm.z <-
      1 - (((3 * Tm.pc.sd) + (3 * Tm.nc.sd)) / abs(Tm.pc.mean - Tm.nc.mean))

    message('Z\' for Tm: ', signif(Tm.z))
    auc.nc.mean <- mean(controls.df$auc[controls.df$ncgc_id == nc])
    auc.nc.sd <- sd(controls.df$auc[controls.df$ncgc_id == nc])
    auc.pc.mean <- mean(controls.df$auc[controls.df$ncgc_id == pc])
    auc.pc.sd <- sd(controls.df$auc[controls.df$ncgc_id == pc])
    auc.z <-
      1 - (((3 * auc.pc.sd) + (3 * auc.nc.sd)) / abs(auc.pc.mean - auc.nc.mean))
    message('Z\' for AUC: ', signif(auc.z))

    if (output == 'plot') {
      Tm.plot <-
        ggplot(controls.df, aes(x = ncgc_id, y = Tm_fit, fill = ncgc_id)) +
        geom_boxplot(outlier.alpha = 0, size = 0.75) +
        geom_jitter(shape = 21, size = 3) +
        theme_minimal() +
        scale_fill_hue() +
        labs(title = 'Controls | Tagg',
             subtitle = paste('Z\': ', signif(Tm.z), sep = '')) +
        theme(
          legend.position = 'none',
          axis.title.x = element_blank(),
          axis.text.x = element_text(size = 12, face = 'bold'),
          axis.text.y = element_text(size = 10),
          axis.title.y = element_text(size = 12, face = 'bold'),
          plot.title = element_text(size = 12, face = 'bold')
        )
      auc.plot <-
        ggplot(controls.df, aes(x = ncgc_id, y = auc, fill = ncgc_id)) +
        geom_boxplot(outlier.alpha = 0, size = 0.75) +
        geom_jitter(shape = 21, size = 3) +
        theme_minimal() +
        scale_fill_hue() +
        labs(title = 'Controls | AUC',
             subtitle = paste('Z\': ', signif(auc.z), sep = '')) +
        theme(
          legend.position = 'none',
          axis.title.x = element_blank(),
          axis.text.x = element_text(size = 12, face = 'bold'),
          axis.text.y = element_text(size = 10),
          axis.title.y = element_text(size = 12, face = 'bold'),
          plot.title = element_text(size = 12, face = 'bold')
        )
      right.grid <-
        plot_grid(Tm.plot, auc.plot, ncol = 1)
      control.grid <-
        plot_grid(
          control_thermogram(controlDF, ncTm = Tm.nc.mean, pcTm = Tm.pc.mean),
          right.grid,
          ncol = 2,
          nrow = 1
        )
      out <- paste(outdir,'controls.png',sep="/")
      ggsave(out, dpi = 'retina', scale = 1.5)
      return(control.grid)
    }
    if (output == 'df') {
      means <-
        c(Tm.nc.mean,
          auc.nc.mean)
      parameters <- c('Tm_fit', 'auc')
      output.df <- tibble(parameters, means)
      return(output.df)
    }
  }


# Dose-response curve fit with LL.4 log-logistic fit
# otrace = TRUE; output from optim method is displayed. Good for diagnostic
# trace = TRUE; trace from optim displayed
# robust fitting
#   robust = 'lms': doesn't handle outlier/noisy data well
dr_fit <- function(df) {
  try(expr = {
    drm(
      resp ~ conc,
      data = df,
      type = 'continuous',
      fct = LL.4(),
      control = drmc(
        errorm = FALSE,
        maxIt = 10000,
        noMessage = TRUE,
      )
    )
  })
}

# Deprecated for now...
# Perform drm on each compound at each temperature
dr_analysis <-
  function(df,
           control = 'DMSO',
           export_label = '',
           plot = TRUE) {
    # Construct df from the unique compound ids (less control) with empty analysis parameters
    model_df <-
      tibble(compound = (unique(filter(
        df, ncgc_id != control
      )$ncgc_id))) %>%
      filter(compound != 'control')
    for (i in 6:ncol(df)) {
      col.nm <- colnames(df)[i]
      model_df[, col.nm] <- NA
    }

    # Make a long df with the parameters (colnames above)
    modelfit_df <- tibble(colnames(model_df)[2:ncol(model_df)])
    names(modelfit_df)[1] <- 'analysis'

    # Loop through each column in every row of modelfit_df and create a drm model for each and
    # add statistics and readouts to a temp df that is bound to modelfit_df
    for (i in 1:nrow(model_df)) {
      # Create a working df with the raw data from compound[i]
      temp_df <-
        filter(df, df$ncgc_id == model_df$compound[(i)]) %>%
        dplyr::select(-c('well', 'row', 'col'))
      print(paste('Analyzing: ', model_df$compound[i]), sep = '')

      # This temp df will hold the statistics that we read out from each model, and is reset every time.
      # Parameters to include:
      # ec50: EC50 reading of the curve fit
      # pval: curve fit pvalue
      # noEffect: p-value of the noEffect test of the dose-response
      # hill: LL4 parameter B
      # ec50: LL4 parameter A
      # lowerlim: LL4 parameter C
      # upperlim: LL4 parameter D
      temp_modelfit_df <- modelfit_df[1] %>%
        mutate(
          ec50 = 0,
          noEffect = 0,
          hill = 0,
          lowerlim = 0,
          upperlim = 0,
          ec50 = 0
        )
      # Iterate through columns
      for (n in 3:ncol(temp_df)) {
        #Make df for drm model by selecting concentration and appropriate column
        dr_df <- temp_df %>% dplyr::select(c(2, n))
        colnames(dr_df)[1] <- 'conc'
        colnames(dr_df)[2] <- 'resp'
        temp.model <-
          drm(
            resp ~ conc,
            data = dr_df,
            fct = LL.4(),
            control = drmc(
              errorm = FALSE,
              maxIt = 500,
              noMessage = TRUE
            )
          )
        # Construct fitted curve for plotting
        pred.fit <-
          expand.grid(pr.x = exp(seq(log(min(
            dr_df[1]
          )), log(max(
            dr_df[1]
          )), length = 1000)))
        # Seems necessary to make loop continue through curves that can't be fit... NEED TO STUDY
        if ("convergence" %in% names(temp.model) == FALSE) {
          pm <-
            predict(object = temp.model,
                    newdata = pred.fit,
                    interval = 'confidence')
          pred.fit$p <- pm[, 1]
          pred.fit$pmin <- pm[, 2]
          pred.fit$pmax <- pm[, 3]
          # Plot out dose response curve if conditional met in function
          if (plot == TRUE) {
            dr_plot <- ggplot(dr_df, aes(x = conc, y = resp)) +
              geom_line(
                data = pred.fit,
                aes(x = pr.x, y = p),
                size = 1.5,
                color = 'black'
              ) +
              geom_point(
                size = 4,
                shape = 21,
                fill = 'orange',
                color = 'black'
              ) +
              scale_x_log10() +
              theme_cowplot() +
              labs(
                title = paste(
                  'Analysis of ',
                  model_df$compound[i],
                  ' by ',
                  colnames(temp_df)[n],
                  sep = ''
                ),
                subtitle = paste(
                  'EC50: ',
                  signif(temp.model$coefficients[4], 3),
                  ' nM',
                  '\n',
                  'Significance of noEffect Test: ',
                  signif(noEffect(temp.model)[3], 3),
                  sep = ''
                ),
                x = 'Concentration'
              )
            print(dr_plot)

            out <- paste(outdir,'dr_curves', export_label, sep="/")
            png(
              filename = paste(
                out,
                '_',
                model_df$compound[i],
                colnames(temp_df)[n],
                '.png',
                sep = ''
              ),
              width = 3200,
              height = 1800,
              res = 300
            )
            print(dr_plot)
            dev.off()
          }
          print(n)
          # Extract fit parameters for the dr model
          temp_modelfit_df$ec50[(n - 2)] <-
            signif(temp.model$coefficients[4], 3)
          temp_modelfit_df$noEffect[(n - 2)] <-
            signif(noEffect(temp.model)[3], 3)
          temp_modelfit_df$hill[(n - 2)] <-
            signif(temp.model$coefficients[1], 3)
          temp_modelfit_df$lowerlim[(n - 2)] <-
            signif(temp.model$coefficients[2], 3)
          temp_modelfit_df$upperlim[(n - 2)] <-
            signif(temp.model$coefficients[3], 3)
        }
      }
      modelfit_df <- modelfit_df %>% cbind(., temp_modelfit_df[2:6])
      names(modelfit_df)[names(modelfit_df) == 'ec50'] <-
        paste('ec50_', model_df$compound[i], sep = '')
      names(modelfit_df)[names(modelfit_df) == 'noEffect'] <-
        paste('noEffect_', model_df$compound[i], sep = '')
      names(modelfit_df)[names(modelfit_df) == 'hill'] <-
        paste('hill_', model_df$compound[i], sep = '')
      names(modelfit_df)[names(modelfit_df) == 'lowerlim'] <-
        paste('lowerlim_', model_df$compound[i], sep = '')
      names(modelfit_df)[names(modelfit_df) == 'upperlim'] <-
        paste('upperlim_', model_df$compound[i], sep = '')
    }
    return(modelfit_df)
  }



# Extract residual sum of squares for dmso columns
# Returns df with all dmso values ready for model fit
dmso_rss <- function(df, control = 'DMSO') {
  df_rss <- df %>%
    dplyr::select(starts_with("t_")) %>%
    pivot_longer(cols = everything())
  colnames(df_rss)[1] <- 'conc'
  colnames(df_rss)[2] <- 'resp'
  df_rss$conc <- as.integer(gsub('t_', '', df_rss$conc))
  message('Fitting DMSO thermogram...')
  rss_model <- dr_fit(df_rss)
  rss_dmso <- sum(residuals(rss_model) ^ 2)
  message('DMSO RSS: ', signif(rss_dmso, 6))
  plot(
    rss_model,
    type = 'all',
    cex = 0.5,
    main = paste('DMSO Thermogram Fit\n', 'RSS: ', signif(rss_dmso, 5), sep =
                   ''),
    sub = paste('DMSO RSS: ', signif(rss_dmso, 5), sep = ''),
    xlab = 'Temperature',
    ylab = 'Fraction Unfolded',
    ylim = c(-0.25, 1.25)
  )
  return(df_rss)
}

compare_models <- function(df, dmso.rss.df, plot = FALSE) {
  temp_df <- df %>% dplyr::select(-one_of(
    'Ea_fit',
    'Tf_fit',
    'kN_fit',
    'bN_fit',
    'kU_fit',
    'bU_fit',
    'S'
  )) %>%
    filter(ncgc_id != 'DMSO' & ncgc_id != 'ignore')
  rss_df <- temp_df %>%
    dplyr::select(-starts_with('t_'))
  rss_df$null.rss <- NA
  rss_df$alt.rss <- NA
  rss_df$rss.diff <- NA

  dmso.model <- dr_fit(dmso.rss.df)
  dmso.rss <- sum(residuals(dmso.model) ^ 2)
  for (i in 1:nrow(temp_df)) {
    cmpnd.df <- temp_df[i, ] %>%
      dplyr::select(starts_with('t_')) %>%
      pivot_longer(cols = everything())
    colnames(cmpnd.df)[1] <- 'conc'
    colnames(cmpnd.df)[2] <- 'resp'
    cmpnd.df$conc <- as.integer(gsub('t_', '', cmpnd.df$conc))

    # Fitting the null model
    null.model <- bind_rows(dmso.rss.df, cmpnd.df)
    null.drm <- dr_fit(null.model)
    null.rss <- sum(residuals(null.drm) ^ 2)
    rss_df$null.rss[i] <- null.rss
    message(
      'Null model for ',
      temp_df$ncgc_id[i],
      ' at concentration \' ',
      temp_df$conc[i],
      '\': ',
      signif(null.rss, 6)
    )
    if (plot == TRUE) {
      plot(null.drm,
           type = 'all',
           cex = 0.5,
           main = 'Null model fit')
    }
    # Fitting the alternate model
    cmpnd.drm <- dr_fit(cmpnd.df)
    cmpnd.rss <- sum(residuals(cmpnd.drm) ^ 2)
    alt.rss <- sum(cmpnd.rss, dmso.rss)
    rss_df$alt.rss[i] <- alt.rss
    message (
      'Alternate Model for ',
      temp_df$ncgc_id[i],
      ' at concentration \' ',
      temp_df$conc[i],
      ' \': ',
      signif(alt.rss, 6)
    )
    rss.diff <- null.rss - alt.rss
    message('RSS.0 - RSS.1: ', signif(rss.diff, 6))
    rss_df$rss.diff[i] <- rss.diff
  }
  return(rss_df)
}

# Calculate the traditional melting parameters from the full + rss model and output df
# of the parameters for each compound
calculate_meltingparams <- function (df, control = 'vehicle') {
  #Standard parameters to test:
  test_params <- c('dHm_fit', 'Tm_fit', 'dG_std', 'T_onset')

  #Set up to loop through entire dataframe for each of the above params
  for (i in 1:length(test_params)) {
    #Initialize the column name
    current_param <- test_params[i]
    df[, paste(current_param, '_diff', sep = '')] <- NA

    #First, calculate the mean of control columns
    mean_control <- mean(df[[current_param]][df$ncgc_id == control])

    #Then, subtract this mean value from each well in the plate in a new column.
    #Can't figure out how to mutate with a pasted column name...
    for (i in 1:nrow(df)) {
      df[i, paste(current_param, '_diff', sep = '')] <-
        df[i, current_param] - mean_control
    }

    #Print out mean and stdev of vehicle for each condition
    std_control <- sd(df[[current_param]][df$ncgc_id == control])
    message('Vehicle mean for ',
            current_param,
            ': ',
            mean_control,
            ' (SD: ',
            std_control,
            ')')
  }
  return(df)
}

# Print out the volcano plots for each parameter and RSS vs. p-val
plot_volcanos <- function(df, save = TRUE) {
  test_params <-
    c('Tm_fit.maxDiff',
      'auc.maxDiff')
  test_pval <-
    c('Tm_fit.maxDiff',
      'auc.maxDiff')

  # Plot out RSS Difference(x) vs. Parameter Difference(y)
  # Conditional fill: grey/alpha if not significant in either
  #   grey/alpha if not significant in either #DDDDDD
  #   teal if by parameter only #009988
  #   orange if by NPARC only #EE7733
  #   wine if by both #882255
  # NEED TO CODE THIS BETTER WTF
  for (i in 1:length(test_params)) {
    current_param <- test_params[i]
    current_pval <- test_pval[i]
    plot.df <- df %>%
      dplyr::select(compound,
                    rss.diff,
                    mannwhit.pval,
                    one_of(current_param),
                    one_of(current_pval))
    # Assign significance testing outcomes
    plot.df$sigVal <-
      case_when((plot.df$mannwhit.pval < 0.05 &
                   plot.df[, current_pval] < 0.05) ~ 'Both',
                (plot.df$mannwhit.pval < 0.05 &
                   plot.df[, current_pval] >= 0.05) ~ 'RSS NPARC',
                (plot.df$mannwhit.pval >= 0.05 &
                   plot.df[, current_pval] < 0.05) ~ 'Parameter',
                (plot.df$mannwhit.pval >= 0.05 &
                   plot.df[, current_pval] >= 0.05) ~ 'Insignificant'
      )

    fillvalues <-
      c('Both', 'RSS NPARC', 'Parameter', 'Insignificant')
    colors <- c('#882255', '#EE7733', '#009988', '#DDDDDD')
    volcano_plot <-
      ggplot(plot.df,
             aes(x = rss.diff,
                 y = plot.df[, current_param],
                 label = compound)) +
      geom_point(shape = 21,
                 aes(fill = sigVal),
                 size = 5) +
      theme_minimal() +
      labs(
        title = paste('Residual Variance vs. ', current_param, sep = ''),
        y = paste(current_param, ' Experimental - Vehicle Mean', sep = ''),
        x = 'RSS0 - RSS1 NPARC',
        fill = 'Significance Detected'
      ) +
      scale_fill_manual(breaks = fillvalues, values = colors) +
      theme(legend.position = 'bottom')
    print(volcano_plot)
    out <- paste(outdir,"/",current_param, sep="")
    ggsave(
      paste(out, '_volcano.png', sep = ''),
      dpi = 'retina',
      scale = 1.25
    )
  }
}

# Plot out RSS Difference by p-value for the MannWhitney
rss.pval.plot <- function (df, savePlot = FALSE) {
  plot.df <- df %>%
    dplyr::select(compound, rss.diff, mannwhit.pval, mannwhit.ec50)
  plot.df$mannwhit.pval <- log2(plot.df$mannwhit.pval)

  rss.plot <-
    ggplot(plot.df,
           aes(x = rss.diff, y = mannwhit.pval, fill = mannwhit.ec50)) +
    geom_point(shape = 21, size = 3.5) +
    theme_minimal() +
    scale_fill_gradient(low = '#EE3377',
                        high = '#88CCEE',
                        na.value = 'grey20') +
    labs(
      title = 'RSS vs. Mann Whitney P-val',
      x = 'RSS0 - RSS1',
      y = 'Log2 Mann Whitney P-val',
      fill = 'NPARC EC50'
    )
  print(rss.plot)
  if (savePlot == TRUE) {
    out <- paste(outdir,'/rssPvalcomp.png',sep="")
    ggsave(out,
      dpi = 'retina',
      scale = 1.25
    )
  }
}

#
parameter_doseresponse <-
  function(df, control = 'vehicle', plot = TRUE) {
    #First calculate the mean for the control/vehicle condition, and construct df with this.
    parameter_df <-
      tibble(compound = (unique(filter(
        df, ncgc_id != control
      )$ncgc_id)))
    for (i in 1:nrow(parameter_df)) {

    }
  }

calculate_zscore <-
  function(df, control = 'vehicle', plot = FALSE) {
    test_params <-
      c('Tm_fit')
    for (i in 1:length(test_params)) {
      current_param <- test_params[i]
      mean_control <-
        mean(df[[current_param]][df$ncgc_id == control])
      std_control <- sd(df[[current_param]][df$ncgc_id == control])
      df[, paste(current_param, '_zscore', sep = '')] <- NA
      for (i in 1:nrow(df)) {
        df[i, paste(current_param, '_zscore', sep = '')] <-
          (df[i, current_param] - mean_control) / std_control
      }
    }
    return(df)
  }

convert_zscore <- function(df, control = 'vehicle', plot = FALSE) {
  test_params <-
    c('Tm_fit_zscore')
  for (i in 1:length(test_params)) {
    current_param <- test_params[i]
    mean_control <- mean(df[[current_param]][df$ncgc_id == control])
    std_control <- sd(df[[current_param]][df$ncgc_id == control])
    df[, paste(current_param, '_pval', sep = '')] <- NA

    #Calculate p value for normal distribution from z score for each column.
    for (i in 1:nrow(df)) {
      df[i, paste(current_param, '_pval', sep = '')] <-
        2 * pnorm(-abs(df[i, current_param]))
    }
  }
  return(df)
}

# Fit the null model to a set of conc ~ resp values.
# Returns: RSS values
# Requires; df with concentration and response values
# df[1]: concentration values
# df[2]: response values
fit_nullmodel <- function(df,
                          plot.model,
                          graphTitle = '') {
  null.model <- lm(resp ~ 1, data = df)
  #Plot if TRUE. For diagnostic use mostly...
  if (plot.model == TRUE) {
    out <- paste(outdir,'models', sep="/")
    try(jpeg(filename = paste(out, graphTitle, '.jpg', sep =
                                '')))
    try(plot(
      df$conc,
      df$resp,
      main = graphTitle,
      pch = 21,
      cex = 3,
      col = 'black',
      bg = 'orange',
      lwd = 3
    ))
    try(abline(null.model, col = 'black', lwd = 3))
    try(dev.off()
    )
  }
  #Return squared residuals for null model
  # message('NUll Model RSS: ',
  #         (sum(residuals(null.model) ^ 2)))
  return(sum(residuals(null.model) ^ 2))
}

# Fit the alternate model log logistic to a set of conc ~ resp values.
# Returns: RSS values
# Requires: same as null_model
fit_altmodel <- function(df,
                         plot.model,
                         graphTitle = '') {
  alt.model <- dr_fit(df)
  if (plot.model == TRUE) {
    out <- paste(outdir,'models', sep="/")
    try(jpeg(filename = paste(out, graphTitle, '.jpg', sep =
                                '')))
    try(plot(
      alt.model,
      main = graphTitle,
      pch = 21,
      cex = 3,
      col = 'black',
      bg = 'cyan',
      lwd = 3
    ))
    try(dev.off()
    )
  }
  # message('Alternate Model RSS: ',
  #         (sum(residuals(alt.model) ^ 2)))
  return(sum(residuals(alt.model) ^ 2))
}

# Derive RSS values for null and alternate model for each compound from full_df
compute.rss.models <-
  function(df,
           control = 'DMSO',
           plotModel = TRUE,
           rssPlot = TRUE,
           drPlot = TRUE) {
    #Construct tibble of unique compounds names
    rss.df <- tibble(compound = (unique(
      filter(df, ncgc_id != control | ncgc_id != 'vehicle')$ncgc_id
    ))) %>%
      filter(compound != 'control') %>%
      filter(compound != 'vehicle')
    rss.df$null.model.n <- NA
    rss.df$alt.model.n <- NA
    rss.df$null.model.sum <- NA
    rss.df$alt.model.sum <- NA
    rss.df$null.model.sd <- NA
    rss.df$alt.model.sd <- NA
    rss.df$rss.diff <- NA
    rss.df$mannwhit.pval <- NA
    rss.df$mannwhit.ec50 <- NA

    for (i in 1:nrow(rss.df)) {
      #Construct df for current compound
      fit.df <- df %>% filter(ncgc_id == toString(rss.df[i, 1])) %>%
        dplyr::select(ncgc_id, conc, starts_with('t_')) %>%
        dplyr::select(!contains('onset'))

      #Plot out dose-response thermogram here?
      if (drPlot == TRUE) {
        dr.thermogram(fit.df, target = rss.df$compound[i])
      }

      #Construct a df to hold the rss values until final calculations of mean,sd,N
      cmpnd.fit.df <- fit.df %>%
        dplyr::select(starts_with('t_'))
      cmpnd.fit.df <- tibble(temp = colnames(cmpnd.fit.df))
      cmpnd.fit.df$null <- NA
      cmpnd.fit.df$alt <- NA

      #Iterate through each temperature, construct df, perform rss analysis, and add to cmpnd.fit.df
      for (t in 3:ncol(fit.df)) {
        current.fit.df <- fit.df %>%
          dplyr::select(1:2, colnames(fit.df)[t])
        colnames(current.fit.df)[3] <- 'resp'
        cmpnd.fit.df$null[t - 2] <-
          fit_nullmodel(current.fit.df,
                        plot.model = plotModel,
                        graphTitle = as.character(paste(
                          current.fit.df[1, 1], ' Null Model at ', colnames(fit.df)[t], sep = ''
                        )))
        cmpnd.fit.df$alt[t - 2] <-
          fit_altmodel(current.fit.df,
                       plot.model = plotModel,
                       graphTitle = as.character(paste(
                         current.fit.df[1, 1], ' Alternate Model at ', colnames(fit.df)[t], sep = ''
                        )))
      }
      # RSS0-RSS1
      cmpnd.fit.df <- cmpnd.fit.df %>%
        mutate(diff = null - alt)
      #Now, we calculate and assign rss values for both models in the rss.df for this compound.
      rss.df$null.model.n[i] <- length(na.omit(cmpnd.fit.df$null))
      rss.df$alt.model.n[i] <- length(na.omit(cmpnd.fit.df$alt))
      rss.df$null.model.sum[i] <- sum(cmpnd.fit.df$null)
      rss.df$alt.model.sum[i] <- sum(cmpnd.fit.df$alt)
      rss.df$null.model.sd[i] <- sd(cmpnd.fit.df$null)
      rss.df$alt.model.sd[i] <- sd(cmpnd.fit.df$alt)
      rss.df$rss.diff[i] <-
        sum(cmpnd.fit.df$null) - sum(cmpnd.fit.df$alt)

      #Perform Mann-Whitney iU test on alternative vs. null model dataframe for compound.
      mann.whit <-
        wilcox.test(x = cmpnd.fit.df$null,
                    y = cmpnd.fit.df$alt,
                    exact = TRUE)
      rss.df$mannwhit.pval[i] <- mann.whit$p.value

      #Message out RSS0-RSS1 and p value
      message('RSS Difference for ',
              rss.df[i, 1],
              ': ',
              rss.df$rss.diff[i])
      message('Mann-Whitney U Test p-val: ',
              rss.df$mannwhit.pval[i])

      # Construct drc model and derive ec50 if p-val is significant
      if (rss.df$mannwhit.pval[i] <= 0.05) {
        #Find what temperature is the max point
        rss.max.temp <-
          cmpnd.fit.df$temp[cmpnd.fit.df$diff == max(cmpnd.fit.df$diff)]
        #Construct df ready for drc at max temperature
        rss.drc.df <- fit.df %>%
          dplyr::select(conc, one_of(rss.max.temp))
        colnames(rss.drc.df)[2] <- 'resp'
        rss.drc.model <- dr_fit(rss.drc.df)
        ec50.temp <- rss.drc.model$coefficients[4]
        if (length(ec50.temp != 0)) {
          rss.df$mannwhit.ec50[i] <- signif(ec50.temp, 3)
        }
      }

      #Plot the RSS values across the temperature range if true
      if (rssPlot == TRUE) {
        # First, clean up the temperatures
        cmpnd.fit.df$temp <- sub('t_', '', cmpnd.fit.df$temp)
        cmpnd.fit.df$temp <- as.numeric(cmpnd.fit.df$temp)

        #Plot RSS as
        rss.plot <- ggplot(cmpnd.fit.df, aes(x = temp, y = diff)) +
          geom_point(shape = 21,
                     size = 4,
                     fill = '#AA4499') +
          theme_minimal() +
          labs(
            title = paste(current.fit.df[1, 1], ' RSS Difference', sep = ''),
            subtitle = paste('Mann-Whitney U pval: ', signif(rss.df$mannwhit.pval[i])),
            sep = '',
            y = 'RSS0-RSS1',
            x = 'Temperature [C]'
          )
        print(rss.plot)
        out <- paste(outdir,'models', current.fit.df[1, 1], sep="/")
        ggsave(
          filename = paste(out, '_rss.png', sep =''),
          scale = 1.25,
          dpi = 'retina'
        )
      }
    }
    return(rss.df)
  }

# Compute the rss difference and significance for each of the parameters
compute_parameter.rssmodel <- function(df, plotModel = FALSE) {
  #Test parameters for standard model
  test_params <-
    c('Tm_fit',
      'auc')

  #Construct df of unique compounds and initialize parameter readouts.
  param.rss.df <- tibble(compound = (unique(
    filter(df, ncgc_id != 'control' & ncgc_id != 'vehicle')$ncgc_id
  )))
  param.rss.df$Tm_fit.ec50 <- as.numeric(NA)
  param.rss.df$Tm_fit.pval <- as.numeric(NA)
  param.rss.df$Tm_fit.maxDiff <- as.numeric(NA)
  param.rss.df$auc.ec50 <- as.numeric(NA)
  param.rss.df$auc.pval <- as.numeric(NA)
  param.rss.df$auc.maxDiff <- as.numeric(NA)

  control.means <- control_analysis(df, output = 'df')

  for (i in 1:nrow(param.rss.df)) {
    cmpnd.fit.df <- df %>%
      filter(ncgc_id == param.rss.df$compound[i])
    #Now iterate through columns in test_params
    for (p in 1:length(test_params)) {
      current_param <- test_params[p]
      current.fit.df <- cmpnd.fit.df %>%
        dplyr::select(ncgc_id, conc, I(test_params[p]))
      colnames(current.fit.df)[3] <- 'resp'
      current.model <- dr_fit(current.fit.df)

      #Workaround to avoid drm that can't converge
      if (class(current.model) != 'list') {
        param.rss.df[i, paste(current_param, '.pval', sep = '')] <-
          noEffect(current.model)[3]

        param.rss.df[i, paste(current_param, '.ec50', sep = '')] <-
          summary(current.model)$coefficients[4]

        #Calculate the maximum difference in param and subtract negative control mean from it.
        current.fit.df$absDiff <-
          abs(current.fit.df$resp - control.means$means[control.means$parameters ==
                                                          current_param])
        param.rss.df[i, paste(current_param, '.maxDiff', sep = '')] <-
          current.fit.df$resp[current.fit.df$absDiff == max(current.fit.df$absDiff)] - control.means$means[control.means$parameters ==
                                                                                                             current_param]

        message('Analyzing Compound ', param.rss.df[i, 1], '...')
        message(current_param)
        message('EC50: ', param.rss.df[i, paste(current_param, '.ec50', sep =
                                                  '')])
        message('No Effect ANOVA p-val: ', signif(param.rss.df[i, paste(current_param, '.pval', sep =
                                                                          '')]), 1)
        if (plotModel == TRUE) {
          out <- paste(outdir,'models', param.rss.df[i, 1], sep="/")
          png(
            filename = paste(
              out,
              '_',
              current_param,
              '.png',
              sep = ''
            ),
            bg = 'transparent'
          )
          plot(
            current.model,
            main = paste(
              param.rss.df[i, 1],
              '\n',
              ' NoEffect pval: ',
              signif(param.rss.df[i, paste(current_param, '.pval', sep =
                                             '')]),
              '\n',
              'EC50: ',
              signif(param.rss.df[i, paste(current_param, '.ec50', sep =
                                             '')]),
              '\n',
              current_param
            )
          )
          dev.off()
        }
      }
    }
  }
  return(param.rss.df)
}

# Creates a thermogram with all concentrations of a target for plotting
# Must match exact ncgc_id in well assignment..
dr.thermogram <- function(df, target = '') {
  # Create df with the compound, conc, and temperature columns
  # df <- df %>%
  #   dplyr::select(ncgc_id, conc, matches('t_\\d')) %>%
  #   filter(., ncgc_id == target)
  df <- df %>%
    pivot_longer(cols = 3:ncol(df),
                 names_to = 'temp',
                 values_to = 'resp')
  df$temp <- as.numeric(sub('t_', '', df$temp))

  dr.plot <- ggplot(df, aes(
    y = resp,
    x = temp,
    fill = as.factor(signif(conc)),
    group_by(signif(conc))
  )) +
    geom_line(color = 'black',
              alpha = 0.8,
              size = 1) +
    geom_point(shape = 21, size = 3) +
    theme_minimal() +
    scale_color_viridis_d() +
    labs(
      title = paste('Dose-Response Thermogram for ', target, sep = ''),
      x = 'Temperature [C]',
      y = 'Response',
      fill = 'Concentration'
    ) +
    theme()
  print(dr.plot)
  out <- paste(outdir,'models', 'dr_', sep="/")
  ggsave(
    filename = paste(out, target, '.png', sep = ''),
    scale = 1.25,
    dpi = 'retina'
  )
  return(dr.plot)
}

# Export heatmaps of EC50 and P-values across analysis parameters
# Pass in parameters df
parameter_heatmaps <- function(df, plotHeat = FALSE) {
  ec50.heat.df <- df %>%
    dplyr::select(compound, contains('ec50')) %>%
    pivot_longer(cols = !compound,
                 names_to = 'parameter',
                 values_to = 'ec50') %>%
    mutate(ec50 = log10(ec50))
  ec50.heat.df$parameter <- ec50.heat.df$parameter %>%
    sub('.ec50', '', .)

  ec50.heat.plot <-
    ggplot(ec50.heat.df,
           aes(
             x = parameter,
             y = compound,
             fill = ec50,
             label = signif(ec50)
           )) +
    geom_tile(color = 'black') +
    geom_text(alpha = 0.85, size = 2.5) +
    theme_minimal() +
    scale_fill_gradientn(colors = c('#EE3377', '#DDCC77', '#88CCEE'), ) +
    labs(title = 'EC50 Parameter Comparison',
         fill = 'Log EC50') +
    theme(
      axis.title.y = element_blank(),
      axis.title.x = element_blank(),
      axis.text.x = element_text(size = 12, face = 'bold')
    )

  pval.heat.df <- df %>%
    dplyr::select(compound, contains('pval')) %>%
    pivot_longer(cols = !compound,
                 names_to = 'parameter',
                 values_to = 'pval') %>%
    mutate(sigVal = ifelse(pval < (0.05 / length(unique(
      df
    ))), 'Significant', 'Insignificant'))
  pval.heat.df$parameter <- pval.heat.df$parameter %>%
    sub('.pval', '', .)
  pval.heat.plot <-
    ggplot(pval.heat.df,
           aes(
             x = parameter,
             y = compound,
             fill = sigVal,
             label = signif(pval)
           )) +
    geom_tile(color = 'black') +
    geom_text(alpha = 0.85, size = 2.5) +
    theme_minimal() +
    labs(title = 'P-Value Parameter Comparison',
         fill = 'P-Value') +
    theme(
      axis.title.y = element_blank(),
      axis.title.x = element_blank(),
      axis.text.x = element_text(size = 12, face = 'bold'),

    )
  if (plotHeat == TRUE) {
    print(pval.heat.plot)
    out <- paste(outdir, 'pval_heatmap.png' , sep="/")
    ggsave(out,
           dpi = 'retina',
           scale = 1.25)
    print(ec50.heat.plot)
    out <- paste(outdir, 'ec50_heatmap.png' , sep="/")
    ggsave(out,
           dpi = 'retina',
           scale = 1.25)
  }
}

# Mutates a binary variable testing each analysis method for significance
# 0 if insignificant
# 1 if significant
determineSig <- function(df, alpha = 0.05) {
  analysisMethods <- dplyr::select(df, contains('pval'))
  analysisMethods <- colnames(analysisMethods)
  analysisMethodsNames <- sub('pval', 'pval.sig', analysisMethods)
  sigVal <- alpha / nrow(df)
  for (i in 1:length(analysisMethods)) {
    df[, analysisMethodsNames[i]] <-
      ifelse(df[, analysisMethods[i]] < sigVal, 1, 0)
    df[is.na(df)] <- 0
  }
  return(df)
}

# Use after determineSig from above
rankOrder <- function(df) {
  methodSig <- dplyr::select(df, contains('pval.sig'))
  methodSig <- colnames(methodSig)
  methodRank <- sub('pval.sig', 'rankOrder', methodSig)
  methods <- sub('.rankOrder', '', methodRank)
  methodsEC <- paste(methods, '.ec50', sep = '')

  for (i in 1:length(methods)) {
    rank.df <- filter(df, df[, (methodSig[i])] == 1)
    rank.df[, methodRank[i]] <-
      as.integer(rank(rank.df[, methodsEC[i]]))
    df <- left_join(df, rank.df)
  }
  return(df)
}
