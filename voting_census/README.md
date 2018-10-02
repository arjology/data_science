## Random Forrests, Gradient Boosting, and the 2016 U.S. elections

The 2016 U.S. Presidential elections were nothing short of spectacular failures of prediction models. I was curious if there was an explanation for which segments of the population might have had profound effects. 

This is the start of a longer exploratory project, but in the meantime I have been playing with random forrests and gradient boosting techniques.

The data is all freely available but just takes some tweeking.
The [jupyter notebook](https://github.com/arjology/data_science/blob/master/US%20voting%20and%20census.ipynb) has all the steps for the downloading and processing of the various data sources. 

There is of course some correlation between the population density, as well as gender-specific populations, and the spread (i.e. (votes_dem - votes_rep)/tot_votes)
![Pops vs Spread](https://github.com/arjology/data_science/blob/master/figures/US_voting_spread_vols_vs_pop_density.png)

Let's take a look first at the results, colered in each state at the county level:

![Percent Democrat and Republican](https://github.com/arjology/data_science/blob/master/figures/US_voting_pct_gop_dem.png) 

There is a clear difference between the central states and the coasts (and hence the coastal elites).

...

Ultimately, a simple random forrest had enough predictive power given the various demographic breakdowns of each county. But we can still do better.
![Random forrest predictions](https://github.com/arjology/data_science/blob/master/figures/US_voting_RF_binary_classification.png)
