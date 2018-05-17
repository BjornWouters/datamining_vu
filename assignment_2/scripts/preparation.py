import pandas as pd


def normalize_values(df):
    averaged_df = df.groupby(['prop_id']).mean().fillna(0)
    for name, series in averaged_df.iteritems():
        if name in ['click_bool', 'booking_bool']:
            continue
        averaged_df[name] = ((averaged_df[name] -
                              averaged_df[name].mean()) /
                             averaged_df[name].std(ddof=0))
    return averaged_df


def import_data(filename):
    fields = [
        'prop_id', 'prop_starrating', 'prop_review_score',
        'prop_brand_bool', 'prop_location_score1', 'prop_location_score2',
        'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag',
        'click_bool', 'booking_bool'
    ]
    df = pd.read_csv(filename, usecols=fields)
    return df


def main():
    df = import_data('../data/small_train.csv')
    df = normalize_values(df)
    df.to_csv('../results/prepared_small_train.csv')


if __name__ == '__main__':
    main()
