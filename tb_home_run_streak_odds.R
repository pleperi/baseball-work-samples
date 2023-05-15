# The Tampa Bay Rays hit home runs in 22 consecutive games to start the 2023
# season. I'll use an exponential distribution to calculate the odds of that 
# streak with a simulation-based approach.
# 
# An exponential distribution is used to model wait times, with a rate parameter 
# that signals the average wait time. Through the first 41 games of the season, 
# spanning 1,575 plate appearances, the Rays had 80 home runs, so I will set the 
# rate parameter to 80/1575. I will draw samples from that distribution and use 
# the actual number of PAs in each of the Rays' first 22 games to determine
# whether at least one home run occured within a game. By calculating the
# HR streak in each of my samples, I can calculate the likelihood of a streak
# of 22 (or more) games given the Rays' home run rate for the season so far.

set.seed(2023)
pa_seq <- c(0, cumsum(c(32, 45, 35, 37, 40, 38, 40, 40, 44, 31, 35, 41, 36, 36,
                          38, 45, 37, 46, 42, 38, 38, 32)))

# Function to simulate a single game
sim_consecutive_hr_games <- function(pa_count = pa_seq) {
  hr_pa <- cumsum(round(rexp(100, rate=80/1575))+1)
  while(hr_pa[100] < pa_count[23]) {
    hr_pa <- cumsum(round(rexp(100, rate=80/1575))+1)
  }

  consecutive_games <- 0
  i <- 2
  while (i <= length(pa_count)) {
    if (sum(hr_pa > pa_count[i-1] & hr_pa <= pa_count[i]) == 0) {
      return(consecutive_games)
    } else {
      consecutive_games <- consecutive_games + 1
      i <- i + 1
    } 
  }
  consecutive_games
}

# Function to loop through a specified number of trials
simulate_games <- function(trials=10000) {
  results <- rep(0, trials)
  for (i in 1:trials) {
    results[i] <- sim_consecutive_hr_games()
  }
  results
}

hr_streaks <- simulate_games()
mean(hr_streaks == 22)
# About 3%

# The "exact" answer (assuming the same home run rate) can be calculated
# using a binomial distribution. For each game, I will calculate the odds
# of no home runs being hit and subtract that number from one to get the 
# odds of at least one home run. Taking the cumulative product gives
# the odds of a HR streak of at least that many games.
data.frame(pa = c(32, 45, 35, 37, 40, 38, 40, 40, 44, 31, 35, 41, 36, 36,
                  38, 45, 37, 46, 42, 38, 38, 32)) %>%
  mutate(odds_no_hr = (1-(80/1575))**pa,
         odds_hr = 1 - odds_no_hr,
         odds_streak = cumprod(odds_hr))
# The exact answer is about 3.8%, with the difference likely due to rounding
# error from forcing the exponential distribution to take whole-number values.