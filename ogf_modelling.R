rm(list = ls()) #Clear memory

# R-SCRIPT SHOWING HOW WEIGHTED OGF MODELLING IS IMPLEMENTED FOR THE MANUSCRIPT CALLED
# 'Detection of old-growth forests using airborne laser scanning and other auxiliary 
# data combined with National Forest Inventory field plot data'

# MAKE SURE THAT YOU HAVE MODEL_DATA.TXT AVAILABLE AND OUTPUT DIRECTORY CREATED,
# AND PACKAGES DOWNLOADED

# BY Dr. JANNE TOIVONEN & Dr. PARVEZ RANA @ Natural Resources Institute Finland
# 5TH October 2026

# load needed packages
library(ranger)
library(glmmTMB)
library(xgboost)
library(caTools)
library(caret)
library(mlbench)
library(dplyr)
library(tidyr)
library(data.table)
library(psych)
library(VSURF)
library(readxl)
library(parallel)
library(lme4)
library(themis)
library(tidymodels)
library(vip)
library(pROC)
library(yardstick)

main_seed <- 123
set.seed(main_seed)

start.time <- Sys.time()

# Read model data in
model_data_OGF <- read.delim(".../model_data.txt",
                             header = T, sep = ";")


# change categorical variables into factors
model_data <- transform(
  model_data_OGF,
  oldGrowth1 = as.factor(oldGrowth1),
  oldGrowth2 = as.factor(oldGrowth2),
  oldGrowth4 = as.factor(oldGrowth4),
  MVMI_soil_type = as.factor(MVMI_soil_type),
  MVMI_site_type = as.factor(MVMI_site_type),
  MVMI_main_type = as.factor(MVMI_main_type),
  MVMI_CC_decid  = as.numeric(MVMI_CC_decid),
  MVMI_CC_all = as.numeric(MVMI_CC_all),
  p_drainage = as.factor(p_drainage),
  block = as.factor(block),
  lidar_dens = as.factor(lidar_dens),
  forestZone = as.factor(forestZone)
)

# remove redundant variables
# OGF1 execution
model_data <- model_data[,-c(3,4)]
# # OGF2 execution
# model_data <- model_data[,-c(2,4)]
# # OGF3 execution
# model_data <- model_data[,-c(2,3)]

colnames(model_data)[2] <- "oldGrowth"

# CHECK for NA values
colnames(model_data)[colSums(is.na(model_data)) > 0]


##############################
# REMOVING REDUNDANT VARIABLES

# select only numeric variables
data_p <- model_data[,-c(1:10)]
# Variables to be added later back to df
to_be_added <- model_data[,c(1:10)]

# Remove near zero predictor variables
NZV <- nearZeroVar(data_p, saveMetrics = TRUE)
selected_var <- row.names(NZV[NZV$nzv == F,])

mod_data <- data_p[,selected_var]

# Add aux variables
m_data <- cbind(to_be_added,mod_data)


# make final transformations
m_data <- transform(
  m_data,
  year_scan = as.factor(year_scan),
  block = as.factor(block),
  lidar_dens = as.factor(lidar_dens),
  forestZone = as.factor(forestZone),
  MVMI_site_type = as.factor(MVMI_site_type),
  MVMI_main_type = as.factor(MVMI_main_type),
  MVMI_soil_type = as.factor(MVMI_soil_type)
)


# Prepare data frames for output
result_df_fin <- data.frame()
importance_df_fin_rf <- data.frame()
importance_df_fin_xgboost <- data.frame()
importance_df_fin_glmm <- data.frame()
lme_randeffGRP <- list()
lme_randeffVAR <- list()
lme_imp <- list()
num_var_vec <- c()
best_p_vec <- c()

f1_pr_df <- data.frame()

rf_param_df <- data.frame()
xgb_param_df <- data.frame()

time_vec_rf <- c()
time_vec_xgboost <- c()
time_vec_glmmTMB <- c()

cat_vars <- c("block","year_scan","lidar_dens","MVMI_site_type","MVMI_soil_type",
              "MVMI_main_type","p_drainage","forestZone")


posi <- 1

# OR "1" = baseline using all variables, OR "0.15" = SMOTE 15% OR "0.25" = SMOTE 25%
# CODE IS SLIGHTLY DIFFERENT FOR OTHER MODELLING CONFIGURATIONS THAN "W" = WEIGHTED
# E.G., SOME VARIABLE/COLUMN NAMES DIFFER AND MODELS DO NOT USE WEIGHT ARGUMENTS
p_vector <- c("w")

# for finding optimal "threshold" value for binary prediction
thr_vec <- c(seq(0.01,0.99, length.out = 99))



for (h in 1:1) {
  
  for (p in p_vector) {
    
    
    ###########################
    #  LEAVE  OUT CV 
    ###########################
    
    create_stratified_block_folds <- function(data, block_col, y_col, k = 5) {
      
      # Ensure inputs are vectors
      block <- data[[block_col]]
      y <- data[[y_col]]
      
      # Count per block
      block_df <- aggregate(
        list(size = y, pos = y),
        by = list(block = block),
        FUN = function(x) c(length(x), sum(x == 1))
      )
      
      # Unpack matrix columns
      block_df$size <- block_df$size[,1]
      block_df$pos  <- block_df$pos[,2]
      
      # Sort blocks: prioritize minority information
      block_df <- block_df[order(-block_df$pos, -block_df$size), ]
      
      # Initialize folds
      folds <- vector("list", k)
      fold_sizes <- rep(0, k)
      fold_pos   <- rep(0, k)
      
      # Greedy assignment
      for (i in 1:nrow(block_df)) {
        
        # Score folds: balance both size and positives
        scores <- fold_sizes + 3 * fold_pos   # weight minority higher
        
        fold_id <- which.min(scores)
        
        # Assign block
        folds[[fold_id]] <- c(folds[[fold_id]], as.character(block_df$block[i]))
        
        # Update fold stats
        fold_sizes[fold_id] <- fold_sizes[fold_id] + block_df$size[i]
        fold_pos[fold_id]   <- fold_pos[fold_id]   + block_df$pos[i]
      }
      
      # Convert to row indices (for modeling)
      fold_indices <- lapply(folds, function(b) which(block %in% b))
      
      return(fold_indices)
    }
    
    folds <- create_stratified_block_folds(
      data = m_data,
      block_col = "block",
      y_col = "oldGrowth",
      k = 5
    )
    
    
    # check if fold split is correct
    intersect(folds[[1]], folds[[2]])
    intersect(folds[[2]], folds[[3]])
    intersect(folds[[3]], folds[[4]])
    intersect(folds[[4]], folds[[5]])
    
    # Create empty list
    result_df <- list()
    imp_df_names <- colnames(m_data)[!colnames(m_data) %in% c("plotID","oldGrowth","folds")]
    importance_df_rf <- as.data.frame(imp_df_names)
    colnames(importance_df_rf)[1] <- "metric"
    importance_df_xgboost <- as.data.frame(imp_df_names)
    colnames(importance_df_xgboost)[1] <- "metric"
    importance_df_glmm <- as.data.frame(imp_df_names)
    colnames(importance_df_glmm)[1] <- "metric"
    
    
    
    for (f in 1:length(folds)) {
      
      fold_idx <- folds[[f]]
      
      test <- m_data[fold_idx,]
      train <- m_data[!m_data$block %in% c(test$block),]
      
      train_tba <- train[ , which(names(train) %in% c(names(to_be_added)))]
      train_red <- train[ , -which(names(train) %in% c(names(to_be_added)))]
      
      
      # Variable Selection Using Random Forests
      m_data_VSURF <- VSURF(train_red, train_tba$oldGrowth,
                            parallel = T,
                            ncores = 10,
                            RFimplem = "ranger",
                            verbose = T)
      # DEBUGGING PURPOSES
      #selected_indices <- c(347,353,63,8,10,356,354,40)
      selected_indices <- m_data_VSURF$varselect.pred
      predictors <- train_red
      VSURF_sel_predictors <- predictors[,selected_indices]
      train_final <- cbind(train_tba,VSURF_sel_predictors)
      
      if (length(selected_indices) == 1) {
        name <- colnames(predictors)[selected_indices]
        colnames(train_final)[ncol(train_final)] <- name
      }
      
      drop <- c("plotID","folds")
      train_final = train_final[,!(names(train_final) %in% drop)]
      
      cat("\n")
      print(paste("Number of fixed predictors from VSURF:", length(selected_indices), sep = " "))
      
      # save the number of fixed predictors in the model (= selected from VSURF prediction phase)
      num_var_vec <- append(num_var_vec, length(selected_indices))
      
      NO_CATEG <- !any(cat_vars %in% colnames(train_final))
      
      
      # To check whether MS-NFI variables are included
      if(ncol(m_data) >= 370) {
        tr_value <- 10
        te_value <- 11
      } else {
        tr_value <- 7
        te_value <- 8
      }
      
      
      # Save scaling of training data
      train_num_cols <- tr_value:ncol(train_final)
      train_means <- sapply(train_final[, train_num_cols], mean, na.rm = TRUE)
      train_sds <- sapply(train_final[, train_num_cols], sd, na.rm = TRUE)
      
      # Scale training data
      train_final_s <- train_final
      train_final_s[, train_num_cols] <- scale(train_final[, train_num_cols], center = train_means, scale = train_sds)
      
      # Scale test data according to training data
      test_num_cols <- te_value:ncol(test)
      # Make sure that column names match or use column names
      common_vars <- intersect(colnames(train_final)[train_num_cols], colnames(test))
      for (col in common_vars) {
        test[[col]] <- (test[[col]] - train_means[col]) / train_sds[col]
      }
      
      
      # Compute class weigths n0 = class 0, n1 = class 1
      n0 <- sum(train_final_s$oldGrowth == 0)
      n1 <- sum(train_final_s$oldGrowth == 1)
      
      beta <- 0.999
      e_0 <- (1 - beta) / (1 - beta^n0)
      e_1 <- (1 - beta) / (1 - beta^n1)
      w_pos <- e_1 / e_0
      
      weights_vec <- c("0" = 1, "1" = w_pos)
      weights_vec_glmm <- ifelse(train_final_s$oldGrowth == 1, w_pos, 1)
      
      
      #########################
      ### 1: Random Forest  ###
      #########################
      
      num_p <- ncol(train_final_s[,!(names(train_final_s) %in% c("plotID","oldGrowth","folds"))])
      
      rf <- ranger(oldGrowth ~ ., data=train_final_s, importance = "permutation", class.weights = weights_vec,
                   num.trees = num_p * 10, respect.unordered.factors = T, num.threads = 10,
                   probability = T)
      
      default_brier <- rf$prediction.error
      
      #########################
      # HYPERPARAMETER TUNING #
      #########################
      
      # PHASE I: tune num.trees
      hyper_grid <- expand.grid(
        num.trees = seq(num_p * 10,1500,by=50)
      )
      
      # execute full cartesian grid search
      for(i in seq_len(nrow(hyper_grid))) {
        # fit model for ith hyperparameter combination
        fit <- ranger(
          formula         = oldGrowth ~ ., 
          data            = train_final_s, 
          num.trees       = hyper_grid$num.trees[i],
          class.weights   = weights_vec,
          verbose         = F,
          seed            = main_seed,
          respect.unordered.factors = T, 
          num.threads = 10,
          probability = T
        )
        # export OOB error 
        hyper_grid$brier[i] <- fit$prediction.error
      }
      
      arranged_results <- hyper_grid %>%
        arrange(brier) %>%
        mutate(perc_gain = (default_brier - brier) / default_brier * 100) %>%
        head(10)
      
      num.trees_best <- arranged_results$num.trees[1]
      benchmark <- arranged_results$brier[1]
      
      
      # PHASE II: tune mtry
      hyper_grid <- expand.grid(
        mtry = seq(floor(sqrt(num_p)),num_p-1,by=1)
      )
      
      # execute full cartesian grid search
      for(i in seq_len(nrow(hyper_grid))) {
        # fit model for ith hyperparameter combination
        fit <- ranger(
          formula         = oldGrowth ~ ., 
          data            = train_final_s, 
          class.weights   = weights_vec,
          mtry            = hyper_grid$mtry[i],
          num.trees       = num.trees_best,
          verbose         = F,
          seed            = main_seed,
          respect.unordered.factors = T, 
          num.threads = 10,
          probability = T
        )
        # export OOB error 
        hyper_grid$brier[i] <- fit$prediction.error
      }
      
      arranged_results <- hyper_grid %>%
        arrange(brier) %>%
        mutate(perc_gain = (benchmark - brier) / benchmark * 100) %>%
        head(10)
      
      mtry_best <- arranged_results$mtry[1]
      benchmark <- arranged_results$brier[1]
      
      
      # PHASE III: tune min.node.size
      hyper_grid <- expand.grid(
        min.node.size = seq(1,20,by=1)
      )
      # execute full cartesian grid search
      for(i in seq_len(nrow(hyper_grid))) {
        # fit model for ith hyperparameter combination
        fit <- ranger(
          formula         = oldGrowth ~ ., 
          data            = train_final_s, 
          class.weights   = weights_vec,
          min.node.size   = hyper_grid$min.node.size[i],
          mtry            = mtry_best, 
          num.trees       = num.trees_best,
          verbose         = F,
          seed            = main_seed,
          respect.unordered.factors = T, 
          num.threads = 10,
          probability = T
        )
        # export OOB error 
        hyper_grid$brier[i] <- fit$prediction.error
      }
      
      arranged_results <- hyper_grid %>%
        arrange(brier) %>%
        mutate(perc_gain = (benchmark - brier) / benchmark * 100) %>%
        head(10)
      
      min_node_size_best <- arranged_results$min.node.size[1]
      benchmark <- arranged_results$brier[1]
      
      
      # PHASE IV: tune replace, sample.fraction and splitrule
      hyper_grid <- expand.grid(
        replace = c(TRUE, FALSE),                               
        sample.fraction = c(0.6, 0.8, 1),
        #splitrule = c("gini", "extratrees","hellinger")
        splitrule = c("gini", "extratrees")
        
      )
      
      # execute full cartesian grid search
      for(i in 1:nrow(hyper_grid)) {
        # fit model for ith hyperparameter combination
        fit <- ranger(
          formula         = oldGrowth ~ ., 
          data            = train_final_s, 
          class.weights   = weights_vec,
          num.trees       = num.trees_best,
          mtry            = mtry_best,
          min.node.size   = min_node_size_best,
          replace         = hyper_grid$replace[i],
          sample.fraction = hyper_grid$sample.fraction[i],
          splitrule       = hyper_grid$splitrule[i],
          verbose         = F,
          seed            = main_seed,
          respect.unordered.factors = T, 
          num.threads = 10,
          probability = T
        )
        # export OOB error 
        hyper_grid$brier[i] <- fit$prediction.error
      }
      
      arranged_results <- hyper_grid %>%
        arrange(brier) %>%
        mutate(perc_gain = (benchmark - brier) / benchmark * 100) %>%
        head(10)
      
      replace_best <- arranged_results$replace[1]
      sample_fraction_best <- arranged_results$sample.fraction[1]
      splitrule_best <- as.character(arranged_results$splitrule[1])
      benchmark <- arranged_results$brier[1]
      
      
      # PHASE V: re-tune num.trees
      hyper_grid <- expand.grid(
        num.trees = seq(num_p * 10,1500,by=50)
      )
      
      # execute full cartesian grid search
      for(i in seq_len(nrow(hyper_grid))) {
        # fit model for ith hyperparameter combination
        fit <- ranger(
          formula         = oldGrowth ~ ., 
          data            = train_final_s, 
          class.weights   = weights_vec,
          num.trees       = hyper_grid$num.trees[i],
          mtry            = mtry_best,
          min.node.size   = min_node_size_best,
          replace         = replace_best,
          sample.fraction = sample_fraction_best,
          splitrule       = splitrule_best,
          verbose         = F,
          seed            = main_seed,
          respect.unordered.factors = T, 
          num.threads = 10,
          probability = T
        )
        # export OOB error 
        hyper_grid$brier[i] <- fit$prediction.error
      }
      
      arranged_results <- hyper_grid %>%
        arrange(brier) %>%
        mutate(perc_gain = (benchmark - brier) / benchmark * 100) %>%
        head(10)
      
      num.trees_best2 <- arranged_results$num.trees[1]
      benchmark <- arranged_results$brier[1]
      
      
      
      
      # SMOTE PART:
      # Following lines are executed if p_vector value is other than 1 or w, which means that
      # minority oversampling is computed to balance training data
      if (p != 1 && p != "w") {
        message("P != 1 -> executing SMOTE")
        balanced_data_s <- smotenc(train_final_s, "oldGrowth", k = 5, over_ratio = p / (1 - p))
        rownames(balanced_data_s) <- NULL
        # DEBUG
        #table(balanced_data_s$oldGrowth)
      } else {
        message ("P = 1 -> continuing without SMOTE")
        balanced_data_s <- train_final_s  
        rownames(balanced_data_s) <- NULL
        rownames(train_final_s)    <- NULL
      }
      
      
      
      # Do the default model fitting and classification
      rf_default <- ranger(oldGrowth ~ ., 
                           data=balanced_data_s,
                           class.weights = weights_vec,
                           importance = "permutation", 
                           num.trees = num_p * 10, 
                           respect.unordered.factors = T, 
                           num.threads = 10,
                           probability = T,
                           seed= main_seed)
      
      
      train_time_rf <- system.time({
        # Do the final model fitting and classification
        rf <- ranger(oldGrowth ~ .,
                     data=balanced_data_s, 
                     class.weights = weights_vec,
                     importance = "permutation", 
                     num.trees       = num.trees_best2,
                     mtry            = mtry_best,
                     min.node.size   = min_node_size_best,
                     replace         = replace_best,
                     sample.fraction = sample_fraction_best,
                     splitrule       = splitrule_best,
                     respect.unordered.factors = T, 
                     num.threads = 10,
                     probability = T,
                     seed = main_seed)
      })
      
      time_vec_rf <- append(time_vec_rf,train_time_rf[["elapsed"]])
      
      
      
      
      # Add RF default predictions here 
      pred_rf_default <- predict(rf_default, data = test)$predictions[, 2]
      pred_rf_default_tr <- rf_default$predictions[1:nrow(train_final_s), 2]
      
      f1_vec_thr_rf1 <- c()
      
      for (thre in thr_vec) {
        train_final_s$ogf_rf_default <- as.numeric(pred_rf_default_tr>thre)        
        cf <- confusionMatrix(data=as.factor(train_final_s$ogf_rf_default), reference = as.factor(train_final_s$oldGrowth),
                              positive = "1")
        #cf
        f1_score_default <- cf$byClass[7]
        #f1_score_default
        f1_vec_thr_rf1 <- append(f1_vec_thr_rf1, f1_score_default)
        #print(thre)
      }
      # retrieve the highest F1-score and its threshold
      max_value <- max(f1_vec_thr_rf1, na.rm = T)
      #max_value  
      max_f1_thrIND <- which(f1_vec_thr_rf1 == max_value)[1]
      
      test$ogf_rf_default <- as.numeric(pred_rf_default>thr_vec[max_f1_thrIND])
      
      # # COMMENT FOLLOWING LINES OUT IF YOU WANT TO CHECK F1-SCORE FOR THE CURRENT FOLD
      # cf <- confusionMatrix(data=as.factor(test$ogf_rf_default), reference = as.factor(test$oldGrowth),
      #                       positive = "1")
      # cf
      # f1_score_default_rf <- cf$byClass[7]
      # f1_score_default_rf
      
      colnames(test)[ncol(test)] <- paste("ogf_rf_default",h,"_",p, sep = "")  
      train_final_s <- train_final_s[, -ncol(train_final_s)]
      
      
      
      
      # Add RF tuned predictions here
      pred_rf <- predict(rf, data = test)$predictions[, 2]
      pred_rf_tr <- rf$predictions[1:nrow(train_final_s), 2]
      
      pr_auc_rf <- yardstick::pr_auc_vec(
        truth = factor(test$oldGrowth, levels = c(0, 1)),
        estimate = pred_rf,
        event_level = "second")
      
      f1_vec_thr_rf2 <- c()
      
      for (thre in thr_vec) {
        train_final_s$ogf_rf <- as.numeric(pred_rf_tr>thre)        
        cf <- confusionMatrix(data=as.factor(train_final_s$ogf_rf), reference = as.factor(train_final_s$oldGrowth),
                              positive = "1")
        #cf
        f1_score_default <- cf$byClass[7]
        #f1_score_default
        f1_vec_thr_rf2 <- append(f1_vec_thr_rf2, f1_score_default)
        #print(thre)
      }
      # retrieve the highest F1-score and its threshold
      max_value <- max(f1_vec_thr_rf2, na.rm = T)
      #max_value  
      max_f1_thrIND <- which(f1_vec_thr_rf2 == max_value)[1]
      
      test$ogf_rf <- as.numeric(pred_rf>thr_vec[max_f1_thrIND])
      cf <- confusionMatrix(data=as.factor(test$ogf_rf), reference = as.factor(test$oldGrowth),
                            positive = "1")
      #cf
      f1_score_rf <- cf$byClass[7]
      #f1_score
      colnames(test)[ncol(test)] <- paste("ogf_rf",h,"_",p, sep = "") 
      train_final_s <- train_final_s[, -ncol(train_final_s)]
      
      
      rf_param_df <- rbind(rf_param_df, data.frame(num_trees_init = num.trees_best, num_trees_fin = num.trees_best2, 
                                                   mtry = mtry_best, min_node_size = min_node_size_best,
                                                   replace = replace_best, sample_fraction = sample_fraction_best,
                                                   splitrule = splitrule_best))
      
      
      ##############################
      ### 2: XGBoost predictions ###
      ##############################
      
      
      # Align test columns to match your raw training data structural columns
      separate_test <- test
      missing_cols <- setdiff(colnames(separate_test), colnames(train_final_s))
      for (col in missing_cols) { separate_test[, col] <- 0 }
      extra_cols <- setdiff(colnames(separate_test), colnames(train_final_s))
      separate_test <- separate_test[, !(colnames(separate_test) %in% extra_cols), drop = FALSE]
      separate_test <- separate_test[, colnames(train_final_s), drop = FALSE]
      
      # Record the exact row counts for indexing
      n_raw   <- nrow(train_final_s)
      n_smote <- nrow(balanced_data_s)
      n_test  <- nrow(separate_test)
      
      # Combine all 3 datasets temporarily into ONE framework for identical columns
      # Order is vital here: Raw first, then SMOTE, then Test
      full_framework <- rbind(train_final_s, balanced_data_s, separate_test)
      
      if (NO_CATEG != TRUE) {
        # Generate the one-hot encoding formula and matrix
        formula <- as.formula(paste("~", paste(cat_vars, collapse = " + "), "-1"))
        X_cat   <- model.matrix(formula, data = full_framework)
        num_vars <- colnames(VSURF_sel_predictors)
        X_num   <- full_framework[, num_vars, drop = FALSE]
        X_final <- cbind(X_num, X_cat)
      } else {
        num_vars <- colnames(VSURF_sel_predictors)
        X_num   <- full_framework[, num_vars, drop = FALSE]
        X_final <- X_num
      }
      
      # Extract matrices cleanly using explicit, isolated row indexing boundaries
      # A. Raw un-SMOTEd matrices (For your tuning loops & OOF threshold searches)
      X_train_raw <- X_final[1:n_raw, ]
      y_train_raw <- as.numeric(as.character(train_final_s$oldGrowth))
      dtrain_raw  <- xgb.DMatrix(as.matrix(X_train_raw), label = y_train_raw)
      
      # B. Balanced SMOTEd matrices (For final default and tuned model training fits)
      X_train_smote <- X_final[(n_raw + 1):(n_raw + n_smote), ]
      y_train_smote <- as.numeric(as.character(balanced_data_s$oldGrowth))
      dtrain_smote  <- xgb.DMatrix(as.matrix(X_train_smote), label = y_train_smote)
      
      # C. Test matrices (For final evaluation predictions)
      X_test <- X_final[(n_raw + n_smote + 1):nrow(X_final), ]
      y_test <- as.numeric(as.character(separate_test$oldGrowth))
      dtest  <- xgb.DMatrix(as.matrix(X_test), label = y_test)
      
      
      
      # Initialize XGBoost parameters
      params_base <- list(
        objective = "binary:logistic",
        eval_metric = "auc",
        eta = 0.3,
        nthread = 10
      )
      
      if (p == "w") {
        params_base$scale_pos_weight <- w_pos
      }
      
      message("Starting XGBoost hyperparameter tuning phase I")
      
      # TUNE PHASE 1: Find optimal number of iterations (nrounds)
      cv_model <- xgb.cv(
        params = params_base,
        data = dtrain_raw,
        nrounds = 2000,
        nfold = 5,
        stratified = TRUE,
        early_stopping_rounds = 20,
        maximize = T,
        verbose = 0
      )
      
      # OBS: DEPENDING ON THE VERSION OF XGBOOST PACKAGE, THE FOLLOWING CODE FOR BEST ITERATION
      # NUMBER CAN BE DIFFERENT FROM 'cv_model$early_stop$best_iteration'. IT CAN BE E.G., 
      # 'cv_model$best_iteration'
      
      message(paste0("Optimal number of iterations (initial): ", cv_model$early_stop$best_iteration))
      nround_init <- cv_model$early_stop$best_iteration
      
      
      message("Starting XGBoost hyperparameter tuning phase II")
      
      # TUNE PHASE 2: Tune max_depth and min_child_weight
      param_grid <- expand.grid("max_depth" = c(3,4,5,6,7,8),"min_child_weight" = c(1,5,10))
      tuning_results <- data.frame()
      
      for (i in 1:nrow(param_grid)) {
        
        params <- params_base
        params$max_depth <- param_grid$max_depth[i]
        params$min_child_weight <- param_grid$min_child_weight[i]
        
        message(paste("Testing combination",i,"/",nrow(param_grid), 
                      ": max_depth =", params$max_depth,
                      ", min_child_weight =", params$min_child_weight, sep = " "))
        
        cv <- xgb.cv(
          params = params,
          data = dtrain_raw,
          nrounds = nround_init,
          nfold = 5,
          early_stopping_rounds = 20,
          maximize = TRUE,
          verbose = 0
        )
        
        tuning_results <- rbind(tuning_results, data.frame(
          max_depth = params$max_depth,
          min_child_weight = params$min_child_weight,
          best_iter = cv$early_stop$best_iteration,
          best_auc = cv$evaluation_log$test_auc_mean[cv$early_stop$best_iteration]
        ))
      }
      
      #print(tuning_results[order(-tuning_results$best_auc), ])
      
      best_row <- tuning_results[which.max(tuning_results$best_auc), ]
      best_auc <- best_row$best_auc
      best_max_dept <- best_row$max_depth
      best_min_child_weight <- best_row$min_child_weight
      
      
      
      message("Starting XGBoost hyperparameter tuning phase III")
      
      # TUNE PHASE 3: Tune subsample and colsample_bytree
      param_grid <- expand.grid("max_depth" = c(best_max_dept),"min_child_weight" = c(best_min_child_weight),
                                "subsample" = c(seq(0.7,1,length.out = 4)),
                                "colsample_bytree" = c(seq(0.7,1,length.out = 4)))
      tuning_results <- data.frame()
      
      for (i in 1:nrow(param_grid)) {
        
        params <- params_base
        params$max_depth <- param_grid$max_depth[i]
        params$min_child_weight <- param_grid$min_child_weight[i]
        params$subsample <- param_grid$subsample[i]
        params$colsample_bytree <- param_grid$colsample_bytree[i]
        
        message(paste("Testing combination",i,"/",nrow(param_grid), 
                      ": subsample =", params$subsample,
                      ", colsample_bytree =", params$colsample_bytree, sep = " "))
        
        cv <- xgb.cv(
          params = params,
          data = dtrain_raw,
          nrounds = nround_init,
          nfold = 5,
          early_stopping_rounds = 20,
          maximize = TRUE,
          verbose = 0
        )
        
        tuning_results <- rbind(tuning_results, data.frame(
          subsample = params$subsample,
          colsample_bytree = params$colsample_bytree,
          best_iter = cv$early_stop$best_iteration,
          best_auc = cv$evaluation_log$test_auc_mean[cv$early_stop$best_iteration]
        ))
      }
      
      #print(tuning_results[order(-tuning_results$best_auc), ])
      
      best_row <- tuning_results[which.max(tuning_results$best_auc), ]
      best_auc <- best_row$best_auc
      best_subsample <- best_row$subsample
      best_colsample_bytree <- best_row$colsample_bytree
      
      
      
      message("Starting XGBoost hyperparameter tuning phase IV")
      
      # TUNE PHASE 4: Tune lambda
      param_grid <- expand.grid("max_depth" = c(best_max_dept),"min_child_weight" = c(best_min_child_weight),
                                "subsample" = c(best_subsample),"colsample_bytree" = c(best_colsample_bytree),
                                "lambda" = c(0, 0.01, 1, 5))
      tuning_results <- data.frame()
      
      for (i in 1:nrow(param_grid)) {
        
        params <- params_base
        params$max_depth <- param_grid$max_depth[i]
        params$min_child_weight <- param_grid$min_child_weight[i]
        params$subsample <- param_grid$subsample[i]
        params$colsample_bytree <- param_grid$colsample_bytree[i]
        params$lambda <- param_grid$lambda[i]
        
        message(paste("Testing combination",i,"/",nrow(param_grid), 
                      ": lambda =", params$lambda, sep = " "))
        
        cv <- xgb.cv(
          params = params,
          data = dtrain_raw,
          nrounds = nround_init,
          nfold = 5,
          early_stopping_rounds = 20,
          maximize = TRUE,
          verbose = 0
        )
        
        tuning_results <- rbind(tuning_results, data.frame(
          lambda = params$lambda,
          best_iter = cv$early_stop$best_iteration,
          best_auc = cv$evaluation_log$test_auc_mean[cv$early_stop$best_iteration]
        ))
      }
      
      #print(tuning_results[order(-tuning_results$best_auc), ])
      
      best_row <- tuning_results[which.max(tuning_results$best_auc), ]
      best_error_afterL2 <- best_row$best_auc
      
      if (best_error_afterL2 < best_auc) {
        best_lambda <- 0
      } else {
        best_lambda <- best_row$lambda 
      }
      
      
      
      message("Starting XGBoost hyperparameter tuning phase V (final)")
      
      # Initialize XGBoost parameters
      params_tune <- list(
        objective = "binary:logistic",
        eval_metric = "auc",
        eta = 0.1,
        max_depth = best_max_dept,
        min_child_weight = best_min_child_weight,
        colsample_bytree = best_colsample_bytree,
        subsample = best_subsample,
        lambda = best_lambda,
        nthread = 10
      )
      
      
      # TUNE PHASE 5: Re-tune number of iterations (nrounds)
      cv_model_fin <- xgb.cv(
        params = params_tune,
        data = dtrain_raw,
        nrounds = 2000,
        nfold = 5,
        stratified = TRUE,
        early_stopping_rounds = 20,
        maximize = T,
        verbose = 0
      )
      message(paste0("Optimal number of iterations (final): ", cv_model_fin$early_stop$best_iteration))
      nround_fin <- cv_model_fin$early_stop$best_iteration
      
      
      
      # Train default XGBoost model      
      xgb_model_default <- xgb.train(
        params = params_base,
        data = dtrain_smote,
        nrounds = nround_init,
        verbose = 0
      )
      pred_default_xgb <- predict(xgb_model_default, dtest)
      pred_default_xgb_tr <- predict(xgb_model_default, dtrain_raw)
      
      
      f1_vec_thr_xgb <- c()
      
      for (thre in thr_vec) {
        train_final_s$ogf_xgb_default <- as.numeric(pred_default_xgb_tr>thre) # Predicted response mean       
        cf <- confusionMatrix(data=as.factor(train_final_s$ogf_xgb_default), reference = as.factor(train_final_s$oldGrowth),
                              positive = "1")
        #cf
        f1_score_default <- cf$byClass[7]
        #f1_score_default
        f1_vec_thr_xgb <- append(f1_vec_thr_xgb, f1_score_default)
        #print(thre)
      }
      # retrieve the highest F1-score and its threshold
      max_value <- max(f1_vec_thr_xgb, na.rm = T)
      #max_value  
      max_f1_thrIND <- which(f1_vec_thr_xgb == max_value)[1]
      
      test$ogf_xgb_default <- as.numeric(pred_default_xgb>thr_vec[max_f1_thrIND]) # Predicted response mean
      # cf <- confusionMatrix(data=as.factor(test$ogf_xgb_default), reference = as.factor(test$oldGrowth),
      #                       positive = "1")
      # #cf
      # f1_score_xgb_default <- cf$byClass[7]
      # f1_score_xgb_default
      colnames(test)[ncol(test)] <- paste("ogf_xgb_default",h,"_",p, sep = "")  
      train_final_s <- train_final_s[, -ncol(train_final_s)]
      
      
      # Train tuned XGBoost model
      train_time_xgboost <- system.time({
        xgb_model_fin <- xgb.train(
          params = params_tune,
          data = dtrain_smote,
          nrounds = nround_fin,
          verbose = 0)
      })
      pred_xgb <- predict(xgb_model_fin, dtest)
      pred_xgb_tr <- predict(xgb_model_fin, dtrain_raw)
      
      pr_auc_xgb <- yardstick::pr_auc_vec(
        truth = factor(y_test, levels = c(0, 1)),
        estimate = pred_xgb,
        #estimate = pred_default_xgb,
        event_level = "second"
      )
      
      f1_vec_thr_xgb <- c()
      
      for (thre in thr_vec) {
        train_final_s$ogf_xgb <- as.numeric(pred_xgb_tr>thre) # Predicted response mean        
        cf <- confusionMatrix(data=as.factor(train_final_s$ogf_xgb), reference = as.factor(train_final_s$oldGrowth),
                              positive = "1")
        #cf
        f1_score_default <- cf$byClass[7]
        #f1_score_default
        f1_vec_thr_xgb <- append(f1_vec_thr_xgb, f1_score_default)
        #print(thre)
      }
      # retrieve the highest F1-score and its threshold
      max_value <- max(f1_vec_thr_xgb, na.rm = T)
      #max_value  
      max_f1_thrIND <- which(f1_vec_thr_xgb == max_value)[1]
      test$ogf_xgb <- as.numeric(pred_xgb>thr_vec[max_f1_thrIND]) 
      cf <- confusionMatrix(data=as.factor(test$ogf_xgb), reference = as.factor(test$oldGrowth),
                            positive = "1")
      #cf
      f1_score_xgb <- cf$byClass[7]
      # f1_score_xgb
      colnames(test)[ncol(test)] <- paste("ogf_xgb",h,"_",p, sep = "")
      train_final_s <- train_final_s[, -ncol(train_final_s)]
      
      time_vec_xgboost <- append(time_vec_xgboost,train_time_xgboost[["elapsed"]])
      
      
      # CREATE SIMILAR PARAMETER DF AS WITH RF
      xgb_param_df <- rbind(xgb_param_df, data.frame(nrounds_init = nround_init, nrounds_fin = nround_fin,
                                                     max_depth = best_max_dept, min_child_weight = best_min_child_weight,
                                                     subsample = best_subsample, colsample_bytree = best_colsample_bytree,
                                                     lambda = best_lambda))
      
      
      #############################
      ### 3: GLMER predictions ####
      #############################
      
      selected1 <- colnames(train_final)
      
      if (NO_CATEG != TRUE) {
        varnames <- selected1[!selected1 %in% c(cat_vars, "oldGrowth")]
        categVar <- selected1[selected1 %in% cat_vars]
        newCategVar <- paste("(1|",categVar,")",sep = "")
        # create formula
        final_form <- as.formula(paste("oldGrowth ~ ", paste(c(varnames,newCategVar), collapse= "+")))
      } else {
        varnames <- selected1[!selected1 %in% c(cat_vars, "oldGrowth")]
        # create formula
        final_form <- as.formula(paste("oldGrowth ~ ", paste(c(varnames), collapse= "+")))
      }
      
      balanced_data_s$w_col <- weights_vec_glmm
      
      train_time_glmmTMB <- system.time({
        # fit glmmTMB model
        glmmTMBr_mod <- glmmTMB(final_form, data = balanced_data_s, weights = w_col,
                                family = binomial(link = "logit"),
                                control = glmmTMBControl(parallel = 10,
                                                         optCtrl = list(iter.max=1e3,eval.max=1e3)))
      })
      
      
      f1_vec_thr <- c()
      
      train_final_s$w_col <- 1
      
      for (thr in thr_vec) {
        
        train_final_s$ogf_glmmTMB <- as.numeric(predict(glmmTMBr_mod,train_final_s,allow.new.levels = T, type = "response")>thr)
        cf <- confusionMatrix(data=as.factor(train_final_s$ogf_glmmTMB), reference = as.factor(train_final_s$oldGrowth),
                              positive = "1")
        #cf
        f1_score_glmmTMB <- cf$byClass[7]
        #f1_score_glmmTMB
        f1_vec_thr <- append(f1_vec_thr, f1_score_glmmTMB)
        #print(thr)
        
      }
      
      # retrieve the highest f1 and its threshold
      max_value <- max(f1_vec_thr, na.rm = T)
      max_f1_thrIND <- which(f1_vec_thr == max_value)[1]
      
      test$w_col <- 1
      
      # predicted probabilities for the test data
      pred_glmmTMB <- predict(
        glmmTMBr_mod,
        test,
        allow.new.levels = T,
        type = "response"
      )
      pr_auc_glmmTMB <- yardstick::pr_auc_vec(
        truth = factor(test$oldGrowth, levels = c(0, 1)),
        estimate = pred_glmmTMB,
        event_level = "second"
      )
      
      
      
      test$ogf_glmmTMB <- as.numeric(predict(glmmTMBr_mod,test,allow.new.levels = T, 
                                             type = "response")>thr_vec[max_f1_thrIND])
      cf <- confusionMatrix(data=as.factor(test$ogf_glmmTMB), reference = as.factor(test$oldGrowth),
                            positive = "1")
      #cf
      f1_score_glmmTMB <- cf$byClass[7]
      # f1_score_glmmTMB
      colnames(test)[ncol(test)] <- paste("ogf_glmmTMB",h,"_",p, sep = "")
      train_final_s <- train_final_s[, -ncol(train_final_s)]
      train_final_s <- subset(train_final_s, select = -w_col)
      test <- subset(test, select = -w_col)
      
      time_vec_glmmTMB <- append(time_vec_glmmTMB,train_time_glmmTMB[["elapsed"]])
      
      
      
      # Combine age predictions of RF, XGBoost and glmmTMB
      test_w_pred <- test[c("plotID","oldGrowth",tail(names(test), 5))]
      result_df <- rbind(result_df, test_w_pred)
      
      # Variable importance part
      # AUC metric
      auc_metric <- function(truth, estimate) {
        as.numeric(pROC::auc(
          response = truth,
          predictor = estimate,
          direction = "<",
          quiet = TRUE      
        ))
      }
      
      # 1. RF (ranger)
      pred_rf <- function(object, newdata) {
        predict(object, data = newdata)$predictions[,2]
      }
      test_sel <- test[colnames(test) %in% colnames(train_final_s)]
      # RF-vi
      imp_rf_result <- vi_permute(
        rf,
        feature_names = colnames(test_sel[,-1]),
        train = test_sel[,-1],
        target = test_sel$oldGrowth,
        metric = auc_metric,
        pred_wrapper = pred_rf,
        nsim = 5,
        smaller_is_better = F
      )
      imp_rf_result <- imp_rf_result[,-3]
      colnames(imp_rf_result)[1] <- "metric"
      colnames(imp_rf_result)[2] <- paste("varImp_RF_",h,"_",p,"_",f, sep = "")
      importance_df_rf <- merge(importance_df_rf, imp_rf_result, by.x = "metric", all.x = T)
      
      
      # 2. XGBoost
      pred_xgb <- function(object, newdata) {
        predict(object, newdata = as.matrix(newdata))
      }
      # XGBoost-vi
      imp_xgb_result <- vi_permute(
        xgb_model_fin,
        feature_names = colnames(X_test),
        train = X_test,
        target = y_test,
        metric = auc_metric,
        pred_wrapper = pred_xgb,
        nsim = 5,
        smaller_is_better = F
      )
      
      # Create a column for the categorical variable (prefix of feature name)
      imp_xgb_result$Group <- sapply(imp_xgb_result$Variable, function(x) {
        # Extract the prefix (before the first digit or number in the name)
        feature_parts <- strsplit(x, "\\d")[[1]]
        feature_parts[1]  # Get the first part (categorical variable name)
      })
      
      imp_xgb_result_cat <- imp_xgb_result[imp_xgb_result$Group %in% cat_vars,]
      imp_xgb_result_fix <-  imp_xgb_result[!imp_xgb_result$Group %in% cat_vars,]
      
      # Aggregate importance by categorical variable (Group)
      # aggregated_importance <- aggregate(Importance ~ Group, data = imp_xgb_result_cat, FUN = mean)
      # colnames(aggregated_importance)[1] <- "metric"
      colnames(imp_xgb_result_fix)[1] <- "metric"
      #imp_xgb_result <- rbind(imp_xgb_result_fix[,c(1,2)], aggregated_importance)
      imp_xgb_result <- imp_xgb_result_fix[,c(1,2)]
      colnames(imp_xgb_result)[2] <- paste("varImp_",h,"_",p,"_",f, sep = "")
      importance_df_xgboost <- merge(importance_df_xgboost, imp_xgb_result, by.x = "metric", all.x = T)
      
      
      # 3. glmmTMB
      
      # save random effect means
      # lme_ranEff <- as.data.frame(ranef(glmmTMBr_mod))
      # lme_ranEff <- lme_ranEff[c(2,4,5)]
      # names(lme_ranEff)[3] <- "mean"
      # 
      # lme_randeffGRP[[posi]] <- lme_ranEff
      
      # save random effect variances and sd
      # block_var <- c(sqrt(VarCorr(glmmTMBr_mod)$cond$block))
      # year_scan_var <- c(sqrt(VarCorr(glmmTMBr_mod)$cond$year_scan))
      # lidar_type_var <- c(sqrt(VarCorr(glmmTMBr_mod)$cond$lidar_type))
      # forest_zone_var <- c(sqrt(VarCorr(glmmTMBr_mod)$cond$forest_zone_SUBname))
      # site_type_var <- c(sqrt(VarCorr(glmmTMBr_mod)$cond$MVMI_site_type))
      # soil_type_var <- c(sqrt(VarCorr(glmmTMBr_mod)$cond$MVMI_soil_type))
      # main_type_var <- c(sqrt(VarCorr(glmmTMBr_mod)$cond$MVMI_main_type))
      # p_drainage_var <- c(sqrt(VarCorr(glmmTMBr_mod)$cond$p_drainage))  
      # 
      # varName_vec <- c("block","year_scan","lidar_type","forest_zone_SUBname","MVMI_site_type",
      #                  "MVMI_soil_type","MVMI_main_type","p_drainage")
      # variance_vec <- c(block_var, year_scan_var, lidar_type_var, forest_zone_var, site_type_var,
      #                   soil_type_var, main_type_var, p_drainage_var)
      # 
      # lme_randeff_stdev <- as.data.frame(cbind(varName_vec, round(variance_vec, digits = 3)))
      # names(lme_randeff_stdev)[1] <- "name"
      # names(lme_randeff_stdev)[2] <- "sd"
      # lme_randeff_stdev$variance <- round(as.numeric(lme_randeff_stdev$sd)^2, digits = 3)
      # names(lme_randeff_stdev)[3] <- "variance"
      # lme_randeff_stdev$sd <- as.numeric(lme_randeff_stdev$sd)
      # 
      # lme_randeffVAR[[posi]] <- lme_randeff_stdev
      
      # extract coefficients
      imp.coefs <- data.frame(coef(summary(glmmTMBr_mod))$cond)
      
      # Save variable estimates, stdevs, z-values and p-values
      lme_imp[[posi]] <- imp.coefs
      
      pred_glmm <- function(object, newdata) {
        predict(object, newdata = newdata, allow.new.levels = T,type = "response")
      }
      
      test_sel$w_col <- 1
      
      # glmmTMB
      imp_glmm_result <- vi_permute(
        glmmTMBr_mod,
        feature_names = colnames(test_sel[,-1]),
        train = test_sel[,-1],
        target = test_sel$oldGrowth,
        metric = auc_metric,
        pred_wrapper = pred_glmm,
        nsim = 5,
        smaller_is_better = F
      )
      
      imp_glmm_result <- subset(imp_glmm_result, Variable != "w_col")
      
      colnames(imp_glmm_result)[2] <- paste("varImp_",h,"_",p,"_",f, sep = "")
      colnames(imp_glmm_result)[1] <- "metric"
      importance_df_glmm <- merge(importance_df_glmm, imp_glmm_result[,c(1,2)], by.x = "metric", all.x = T)
      
      
      # Put F1-scores and PR-AUCs of this fold into specific data frame
      new_row <- list(fold = f, f1_rf = f1_score_rf, f1_xgb = f1_score_xgb, f1_logreg = f1_score_glmmTMB, 
                      prAuc_rf = pr_auc_rf, prAuc_xgb = pr_auc_xgb, prAuc_logreg = pr_auc_glmmTMB)
      f1_pr_df <- rbind(f1_pr_df, new_row)
      
      
      # Print phase of execution
      print(paste("Rep: ", h,"; p: ",p,"; Fold:",f),sep = "")
      
    }
    
    
    
    if (h == 1) {
      result_df_fin <- result_df
    } else {
      result_df <- result_df[-c(2)]
      result_df_fin <- merge(result_df_fin,result_df, by = "plotID")
    }
    
    # Compute mean variable importance for RF
    rownames(importance_df_rf) <- importance_df_rf[,1]
    importance_df_rf <- importance_df_rf[,-1]
    importance_df_rf <- importance_df_rf[rowSums(is.na(importance_df_rf)) < ncol(importance_df_rf), ]
    impDF_rf <- as.data.frame(rowMeans(importance_df_rf, na.rm = T))
    impDF_rf$metric <- rownames(impDF_rf)  
    names(impDF_rf)[1] <- paste("Mean_RF_", h,"_",p,sep = "")
    
    
    if (h == 1) {
      importance_df_fin_rf <- impDF_rf
    } else {
      importance_df_fin_rf <- merge(importance_df_fin_rf,impDF_rf, by = "metric", all.x = T)
    }
    
    
    # Compute mean variable importance for XGBoost
    rownames(importance_df_xgboost) <- importance_df_xgboost[,1]
    importance_df_xgboost <- importance_df_xgboost[,-1]
    importance_df_xgboost <- importance_df_xgboost[rowSums(is.na(importance_df_xgboost)) < ncol(importance_df_xgboost), ]
    impDF_xgboost <- as.data.frame(rowMeans(importance_df_xgboost, na.rm = T))
    impDF_xgboost$metric <- rownames(impDF_xgboost)  
    names(impDF_xgboost)[1] <- paste("Mean_XGB_", h,"_",p,sep = "")
    
    
    if (h == 1) {
      importance_df_fin_xgboost <- impDF_xgboost
    } else {
      importance_df_fin_xgboost <- merge(importance_df_fin_xgboost,impDF_xgboost, by = "metric", all.x = T)
    }
    
    # Compute mean variable importance for glmmTMB
    rownames(importance_df_glmm) <- importance_df_glmm[,1]
    importance_df_glmm <- importance_df_glmm[,-1]
    importance_df_glmm <- importance_df_glmm[rowSums(is.na(importance_df_glmm)) < ncol(importance_df_glmm), ]
    impDF_glmm <- as.data.frame(rowMeans(importance_df_glmm, na.rm = T))
    impDF_glmm$metric <- rownames(impDF_glmm)  
    names(impDF_glmm)[1] <- paste("Mean_glmer_", h,"_",p,sep = "")
    
    
    if (h == 1) {
      importance_df_fin_glmm <- impDF_glmm
    } else {
      importance_df_fin_glmm <- merge(importance_df_fin_glmm,impDF_glmm, by = "metric", all.x = T)
    }
    
    
  } # p_vector bracket
  
  
} # repetition bracket 



print(paste("Average time of RF model fitting: ", round(mean(time_vec_rf), digits = 2), sep = " "))
cat("\n")
print(paste("Average time of XGBoost model fitting: ", round(mean(time_vec_xgboost), digits = 2), sep = " "))
cat("\n")
print(paste("Average time of glmmTLMB model fitting: ", round(mean(time_vec_glmmTMB), digits = 2), sep = " "))
cat("\n")
print(paste("Mean number of fixed predictors in models was:",round(mean(num_var_vec)), sep = " "))

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken


write.table(importance_df_fin_rf, ".../output/rf_w_imp_all.txt", col.names = T, sep = " ", row.names = F)
write.table(importance_df_fin_xgboost, ".../output/xgb_w_imp_all.txt", col.names = T, sep = " ", row.names = F)
write.table(importance_df_fin_glmm, ".../output/glmm_w_imp_all.txt", col.names = T, sep = " ", row.names = F)
write.table(result_df_fin, ".../output/ogf_w_result_all.txt", col.names = T, sep = " ", row.names = F)

write.table(rf_param_df,".../output/rf_w_param_df.txt", col.names = T, sep = " ", row.names = F)
write.table(xgb_param_df,".../output/xgb_w_param_df.txt",col.names = T, sep = " ", row.names = F)

write.table(f1_pr_df,".../output/f1_prAUC_w_df.txt",col.names = T, sep = " ", row.names = F)

saveRDS(lme_imp,".../output/ogf_w_glmmTMB_outputs.rds")

