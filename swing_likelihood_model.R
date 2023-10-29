library(tidyverse)
library(tidymodels)
library(finetune) # optimize grid tuning
library(BradleyTerry2) # dependency required to run tune_race_win_loss from finetune
library(baseballr)
library(xgboost)
library(Metrics) # to get AUC
library(pdp)
theme_set(theme_bw())

# Retrieve pitch data
games_to_get <- mlb_schedule(2022) %>%
  filter(series_description == 'Regular Season' & 
           status_detailed_state != 'Cancelled') %>%
  select(game_pk) %>%
  unique()

game_list = vector("list", length = length(games_to_get$game_pk))
for (i in 1:length(games_to_get$game_pk)) {
  game_list[[i]] <- mlb_pbp(games_to_get$game_pk[i]) %>%
    filter(isPitch) %>%
    mutate_at(c('atBatIndex'), as.numeric) %>% 
    select(game_pk,
           inning = about.inning,
           halfInning = about.halfInning,
           atBatIndex,
           pitchNumber,
           currentAwayScore = result.awayScore,
           currentHomeScore = result.homeScore,
           batter = matchup.batter.fullName,
           batter_id = matchup.batter.id,
           batSide = matchup.batSide.code,
           pitcher = matchup.pitcher.fullName,
           pitcher_id = matchup.pitcher.id,
           pitchHand = matchup.pitchHand.description,
           matchup.splits.menOnBase,
           details.description,
           balls = count.balls.start,
           strikes = count.strikes.start,
           outs = count.outs.start,
           pitchData.startSpeed,
           pitchData.endSpeed,
           pitchData.zone,
           pitchData.extension,
           pitchData.coordinates.aX,
           pitchData.coordinates.aY,
           pitchData.coordinates.aX,
           pitchData.coordinates.pX,
           pitchData.coordinates.pZ,
           pitchData.coordinates.pfxX,
           pitchData.coordinates.pfxZ,
           pitchData.coordinates.vX0,
           pitchData.coordinates.vY0,
           pitchData.coordinates.vZ0,
           pitchData.coordinates.x0,
           pitchData.coordinates.y0,
           pitchData.coordinates.z0,
           pitchData.breaks.breakAngle,
           pitchData.breaks.breakLength,
           pitchData.breaks.breakY,
           spinRate = pitchData.breaks.spinRate,
           spinDirection = pitchData.breaks.spinDirection) %>%
    unique()
}

pitches_2022 <- dplyr::bind_rows(game_list) 

#write_csv(pitches_2022, "pitches_2022.csv")
#pitches_2022 <- read_csv("pitches_2022.csv")

# Spot-check a single game
pitches_2022 %>%
  filter(game_pk == 663178) %>%
  arrange(atBatIndex, pitchNumber) %>%
  View()

# Data prep: score and men on base are populated with data from after the PA
# Lag these variables to get the values before that PA
lag_pa <- pitches_2022 %>%
  select(game_pk, inning, halfInning, atBatIndex, 
         currentAwayScore, currentHomeScore, 
         matchup.splits.menOnBase) %>%
  unique() %>%
  group_by(game_pk) %>%
  mutate(pre_pitch_away_score = if_else(!is.na(lag(currentAwayScore, order_by = atBatIndex)),
                                        lag(currentAwayScore, order_by = atBatIndex),
                                        currentAwayScore),
         pre_pitch_home_score = if_else(!is.na(lag(currentHomeScore, order_by = atBatIndex)),
                                        lag(currentHomeScore, order_by = atBatIndex),
                                        currentHomeScore)) %>%
  group_by(game_pk, inning, halfInning) %>%
  mutate(pre_pitch_men_on = lag(matchup.splits.menOnBase, order_by = atBatIndex, default = "Empty")) %>%
  ungroup()

# Data prep: get the count before the pitch is thrown
# Join pre-PA data from the previous step
pitches_for_model <- pitches_2022 %>%
  filter(!is.na(pitchData.coordinates.pX) & !is.na(pitchData.coordinates.pZ)) %>%
  group_by(batter_id) %>%
  filter(n() >= 500) %>%
  group_by(game_pk, atBatIndex) %>%
  mutate(pre_pitch_strikes = lag(strikes, default = 0, order_by = pitchNumber),
         pre_pitch_balls = lag(balls, default = 0, order_by = pitchNumber)) %>%
  ungroup() %>%
  left_join(lag_pa, by = c("game_pk", "inning", "halfInning", "atBatIndex",
                           "currentAwayScore", "currentHomeScore", "matchup.splits.menOnBase"))

pitches_for_model %>%
  group_by(pre_pitch_strikes) %>%
  summarize(incorrect_strike = mean(details.description == "Called Strike" &
                                      pitchData.zone > 10, na.rm = TRUE),
            incorrect_ball = mean(details.description == "Ball" &
                                    pitchData.zone < 10, na.rm = TRUE),
            n = n())

swings <- c("Foul",
            "Foul Tip",
            "In play, no out",
            "In play, out(s)",
            "In play, run(s)",
            "Swinging Strike",
            "Swinging Strike (Blocked)")

# Build the modeling data
# Add flags for whether the prior pitch was correctly called
# Add a target variable as a factor
pitches_with_flag <- pitches_for_model %>%
  group_by(game_pk, atBatIndex) %>%
  mutate(incorrect_strike = if_else(details.description == "Called Strike" &
                                      pitchData.zone > 10, 1, 0),
         correct_strike = if_else(details.description == "Called Strike" &
                                      pitchData.zone < 10, 1, 0),
         incorrect_ball = if_else(details.description == "Ball" &
                                    pitchData.zone < 10, 1, 0),
         correct_ball = if_else(str_detect(details.description, "Ball") &
                                    pitchData.zone > 10, 1, 0),
         whiff = if_else(str_detect(details.description, "Swinging Strike") |
                           details.description == "Foul Tip", 1, 0),
         foul = if_else(details.description == "Foul", 1, 0),
         previous_pitch_incorrect_strike = lag(incorrect_strike, order_by = pitchNumber, default = 0),
         previous_pitch_correct_strike = lag(correct_strike, order_by = pitchNumber, default = 0),
         previous_pitch_incorrect_ball = lag(incorrect_ball, order_by = pitchNumber, default = 0),
         previous_pitch_correct_ball = lag(correct_ball, order_by = pitchNumber, default = 0),
         previous_pitch_whiff = lag(whiff, order_by = pitchNumber, default = 0),
         previous_pitch_foul = lag(foul, order_by = pitchNumber, default = 0)) %>%
  arrange(game_pk, atBatIndex, pitchNumber) %>%
  # for future research - a flag like this would identify if any prior pitch in the PA
  # was incorrectly called
  mutate(previous_pa_incorrect_strike = cummax(previous_pitch_incorrect_strike),
         previous_pa_incorrect_ball = cummax(previous_pitch_incorrect_ball)) %>%
  ungroup() %>%
  filter(pre_pitch_strikes < 3 & pre_pitch_balls < 4 & 
           !str_detect(details.description, "Bunt") &
           !details.description == "Pitchout" &
            !details.description == "Hit By Pitch") %>%
  mutate(swing = as.factor(details.description %in% swings)) # for xgboost, the dependent variable must be a factor
  
# Save the data to be able to easily read it in 
# saveRDS(pitches_with_flag, "pitches_with_flag.rds")
pitches_with_flag <- readRDS("pitches_with_flag.rds")
pitches_with_flag <- pitches_with_flag %>%
  mutate(previousPitchCategory = case_when(
    previous_pitch_correct_strike == 1 ~ "Correct Strike",
    previous_pitch_incorrect_strike == 1 ~ "Incorrect Strike",
    previous_pitch_correct_ball == 1 ~ "Correct Ball",
    previous_pitch_incorrect_ball == 1 ~ "Incorrect Ball",
    previous_pitch_whiff == 1 ~ "Whiff",
    previous_pitch_foul == 1 ~ "Foul",
    .default = "None"
    ))

# Is there a difference following a correctly vs incorrectly called pitch?
pitches_with_flag %>%
  filter(str_detect(previousPitchCategory, "Strike")) %>%
  group_by(previousPitchCategory, pre_pitch_strikes) %>%
  summarize(swing_rate = mean(details.description %in% swings, na.rm = TRUE),
            n = n()) %>%
  arrange(pre_pitch_strikes)
  
pitches_with_flag %>%
  filter(str_detect(previousPitchCategory, "Ball")) %>%
  group_by(previousPitchCategory, pre_pitch_balls) %>%
  summarize(swing_rate = mean(details.description %in% swings, na.rm = TRUE),
            zone_rate = mean(pitchData.zone < 10),
            n = n()) %>%
  arrange(pre_pitch_balls)

# Begin modeling
# Set seed for reproducibility
# Create training and test data
set.seed(1229)
pitches_sample <- pitches_with_flag[sample(nrow(pitches_with_flag),
                                           round(nrow(pitches_with_flag)/10)),]
saveRDS(pitches_sample, "pitches_sample.rds")

set.seed(218)
xgb_split <- initial_split(pitches_sample, prop = .7)

train_data = training(xgb_split)
test_data = testing(xgb_split)

doParallel::registerDoParallel(cores = 3)

mset <- metric_set(roc_auc, yardstick::accuracy)
grid_control <- control_grid(save_pred = TRUE)

rec1 <- recipe(swing ~ inning + halfInning + outs + 
                 batter + batSide + pitchHand +
                 pitchData.startSpeed + pitchData.endSpeed + pitchData.extension +
                 pitchData.coordinates.aX + pitchData.coordinates.aY +
                 pitchData.coordinates.pX + pitchData.coordinates.pZ +
                 pitchData.coordinates.pfxX + pitchData.coordinates.pfxZ +
                 pitchData.coordinates.vX0 + pitchData.coordinates.vY0 + pitchData.coordinates.vZ0 +
                 pitchData.coordinates.x0 + pitchData.coordinates.y0 + pitchData.coordinates.z0 +
                 pitchData.breaks.breakAngle + pitchData.breaks.breakLength +
                 spinRate + spinDirection +
                 pre_pitch_strikes + pre_pitch_balls + pre_pitch_away_score + pre_pitch_home_score +
                 pre_pitch_men_on + previousPitchCategory,
               data = train_data) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) # this step will be used for variable importance

train_fold <- train_data %>%
  rsample::vfold_cv(v = 5)

# First step - tune the number of trees with the default learning rate
wf1 <- workflow() %>%
  add_recipe(rec1) %>%
  add_model(boost_tree(trees = tune()) %>% 
              set_engine("xgboost") %>%
              set_mode("classification"))

tune1 <- wf1 %>%
  tune_grid(train_fold,
            grid = crossing(trees = c(50, 100, 200)),
            metrics = mset,
            control = grid_control)
saveRDS(tune1, "tune_n_trees.rds")
rm(tune1)

autoplot(tune1)
select_best(tune1, metric = "roc_auc")

# Next, tune the tree depth and minimum root size
wf2 <- workflow() %>%
  add_recipe(rec1) %>%
  add_model(boost_tree(trees = 100,
                       tree_depth = tune(),
                       min_n = tune()) %>% 
              set_engine("xgboost") %>%
              set_mode("classification"))

tune2 <- wf2 %>%
  tune_grid(train_fold,
            grid = crossing(tree_depth = seq(4, 8),
                            min_n = c(5, 15, 50)),
            metrics = mset,
            control = grid_control)
saveRDS(tune2, "tune_depth_minn.rds")

autoplot(tune2)
select_best(tune2, metric = "roc_auc")
rm(tune2)

# Tune the proportion of variables used in each step and the sample size
wf3 <- workflow() %>%
  add_recipe(rec1) %>%
  add_model(boost_tree(trees = 100,
                       tree_depth = 6,
                       min_n = 5,
                       mtry = tune(),
                       sample_size = tune()) %>% 
              set_engine("xgboost", counts = FALSE) %>%
              set_mode("classification"))

tune3 <- wf3 %>%
  tune_race_win_loss(train_fold,
                     grid = crossing(mtry = seq(.4, .8, .1), #seq(.6, 1, .1),
                                     sample_size = c(.9)), #seq(.8, 1, .1)),
                     metrics = mset)
saveRDS(tune3, "tune_vars_sample.rds")

autoplot(tune3)
select_best(tune3, metric = "roc_auc")
select_best(tune3, metric = "accuracy")
# In the first run, the combo of mtry = .6 and sample_size = .9 did best in AUC.
# Although mtry of .8 performed best in the second iteration, the difference was very slight
# and there was no performance benefit from dropping mtry, so I retained the result from
# the first run.

# Drop learning rate - tune learning rate in combination with threes
wf4 <- workflow() %>%
  add_recipe(rec1) %>%
  add_model(boost_tree(learn_rate = tune(),
                       trees = tune(),
                       tree_depth = 6,
                       min_n = 5,
                       mtry = .6,
                       sample_size = .9) %>% 
              set_engine("xgboost", counts = FALSE) %>%
              set_mode("classification"))

tune4 <- wf4 %>%
  tune_race_win_loss(train_fold,
            grid = crossing(learn_rate = c(.05, .1), #c(.3, .1, .05, .01),
                            trees = seq(400, 1000, 200)), #seq(100, 600, 100)),
            metrics = mset)
saveRDS(tune4, "tune_vars_sample.rds")

autoplot(tune4)
select_best(tune4, metric = "roc_auc")

# In the first run, the best combination for AUC was 600 trees, learning rate .005 but
# both AUC and accuracy appeared to still be increasing as the number of trees grew.
# At 1000 trees and learning rate .05, the AUC and accuracy still appeared to be
# increasing but at a slower rate.

# Optional final tuning step to refine parameters
# tune_final <- wf4 %>%
#   finetune::tune_sim_anneal(train_fold, 
#                   iter = 10, 
#                   metrics = mset,
#                   initial = select_best(tune4, metric = "roc_auc"))

# Finalize model
wf_best <- wf4 %>%
  finalize_workflow(select_best(tune4, metric = "roc_auc"))

fit_best <- wf_best %>%
  fit(train_data)
saveRDS(fit_best, "final_model.rds")

# What are the most important variables in the model?
xg_importances <- xgboost::xgb.importance(model = fit_best$fit$fit$fit)

xg_importances %>%
  filter(row_number(-Gain) <= 20) %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>%
  ggplot(aes(Gain, Feature)) +
  geom_point()

# How well does the model do on test data?
score_test_data <- predict(fit_best, test_data)
score_test_prob <- predict(fit_best, test_data, type = "prob")

table(test_data$swing, score_test_data$.pred_class)

sum(diag(table(test_data$swing, score_test_data$.pred_class)))/
  sum(table(test_data$swing == TRUE, score_test_data$.pred_class))
# 78% accuracy

auc(as.numeric(test_data$swing) - 1, score_test_prob$.pred_TRUE)
# .866

feature_names <- fit_best$fit$fit$fit$feature_names

# Calculate partial dependencies
train_prepped <- rec1 %>%
  prep() %>%
  juice() %>%
  select(-swing) 

pdp::partial(fit_best$fit$fit$fit,
             train = train_prepped %>%
               select(match(fit_best$fit$fit$fit$feature_names,
                            train_prepped %>% names())),
             type = "classification",
             pred.var = "previousPitchCategory_Incorrect.Strike",
             prob = T)
# Essentially no impact

# Summarize predicted data for pitches following incorrect calls
test_data %>%
  add_column(pred = score_test_prob$.pred_TRUE) %>%
  group_by(previousPitchCategory) %>%
  summarize(median_predicted = median(pred), 
            predicted_swing_rate = mean(pred >= .5),
            actual_swing_rate = mean(details.description %in% swings), 
            pitches_in_zone = mean(pitchData.zone < 10), 
            mean_strikes = mean(pre_pitch_strikes),
            mean_velo = mean(pitchData.startSpeed, na.rm = TRUE),
            n = n()) %>%
  View()
# Interestingly, hitters swing less often than they are predicted to after "None",
# which is usually the first pitch of PA but sometimes the pitch right after a pitchout
# or bunt attempt

# Visualize pitch locations - is there a difference in the locations
# following correct and incorrect calls?
# Use test data to reduce sample size for ease of plotting
zone <- data.frame(x = c(-.83, .83, .83, -.83, -.83),
                   y = c(3.5, 3.5, 1.5, 1.5, 3.5))

test_data %>%
  mutate(grp = case_when(previous_pitch_incorrect_ball == 1 ~ "Incorrect Ball",
                         previous_pitch_correct_ball == 1 ~ "Correct Ball")) %>%
  filter(!is.na(grp)) %>%
  mutate(x = round(pitchData.coordinates.pX * 2) / 2,
         z = round(pitchData.coordinates.pZ * 2) / 2) %>%
  count(grp, x, z) %>%
  group_by(grp) %>%
  mutate(pct = n/sum(n)) %>%
  ggplot() +
  geom_tile(aes(x, z, fill = pct)) + 
  geom_path(data = zone, aes(x, y), color = "red", linewidth = 2) +
  facet_wrap(~grp) +
  coord_equal() +
  labs(x = "", y = "", fill = "% of pitches") +
  scale_fill_viridis_c(labels = scales::percent)
  