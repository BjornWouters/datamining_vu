import pandas as pd


def normalize_values(df, test=False):
    if not test:
        averaged_df = df.groupby(['prop_id']).mean().fillna(0)
    else:
        averaged_df = df.fillna(0)

    for name, series in averaged_df.iteritems():
        if name in ['click_bool', 'booking_bool', 'promotion_flag',
                    'prop_brand_bool', 'srch_id', 'prop_id']:
            continue
        averaged_df[name] = ((averaged_df[name] -
                              averaged_df[name].mean()) /
                             averaged_df[name].std(ddof=0))

    # Don't use srch_ids where there is no booking realized
    if not test:
        averaged_df = averaged_df[averaged_df.booking_bool != 0]

    return averaged_df.fillna(0)


def import_data(filename):
    fields = [
        'prop_id', 'prop_starrating', 'prop_review_score',
        'prop_brand_bool', 'prop_location_score1', 'prop_location_score2',
        'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag',
        'srch_room_count', 'srch_children_count', 'srch_adults_count',
        'visitor_hist_adr_usd', 'visitor_hist_starrating', 'click_bool',
        'booking_bool'
    ]
    df = pd.read_csv(filename, usecols=fields)
    fields1 = ['srch_id', 'prop_id', 'click_bool', 'booking_bool',
               'visitor_hist_starrating', 'visitor_hist_adr_usd']
    df1 = pd.read_csv(filename, usecols=fields1).fillna(0)
    return df, df1


def import_test_data(filename):
    fields = [
        'srch_id', 'prop_id', 'prop_starrating', 'prop_review_score',
        'prop_brand_bool', 'prop_location_score1', 'prop_location_score2',
        'prop_log_historical_price', 'price_usd', 'promotion_flag',
        'srch_room_count', 'srch_children_count', 'srch_adults_count',
        'visitor_hist_adr_usd', 'visitor_hist_starrating'
    ]
    df = pd.read_csv(filename, usecols=fields)
    return df


def feature_selection(df):
    # Relative price in comparison with the star rating
    df['rating_price'] = df['prop_starrating']/df['price_usd']

    # Price differencevisitor_hist_starrating
    df['price_difference'] = df['price_usd']-df['visitor_hist_adr_usd']

    # Rating differences
    df['rate_differences'] = df['prop_review_score']-df['visitor_hist_starrating']

    return df


def main():
    df, df1 = import_data('../data/training_set_VU_DM_2014.csv')
    test_set = import_test_data('../data/test_set_VU_DM_2014.csv')

    df = feature_selection(df)
    df = normalize_values(df)

    test_set = feature_selection(test_set)
    test_set = normalize_values(test_set, test=True)

    test_set.to_csv('../results/test_set.csv', index=False)
    df.to_csv('../results/prepared_train.csv')
    df1.to_csv('../results/predict_dataset.csv', index=False)


if __name__ == '__main__':
    main()
