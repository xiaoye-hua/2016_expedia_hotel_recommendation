# Features

1. Seasonality
    1. [x] month, dayofweek, 
    2. [ ] whether_holiday
    2. [ ] stay days: dayofweek, whether_holiday
2. Context
    1. [x] day_distance
    2. [x] length of stay
    3. [x] is_domestic
3. Data issues
    1. [ ] posa_continent is highly correlated with site_name
    2. [ ] orig_destination_distance has 3582 (35.8%) missing values
    3. [ ] Reduce categories
        1. [ ] site_name
        2. [ ] user_location_country
    4. [ ] too much categoriy
        1. [ ] hotel_cluster ????? !!!!
        2. [ ] user_location_country
        2. [ ] user_location_city
        3. [ ] user_id
        3. [x] srch_destination_id -> use the latent variable 
        4. [ ] hotel_country
        5. [ ] hotel_market
    5. [ ] meaning??
        1. [ ] cnt in training data??