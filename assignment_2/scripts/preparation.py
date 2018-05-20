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
    fields1 = ['srch_id'] + fields
    df1 = pd.read_csv(filename, usecols=fields1).fillna(0)
    return df, df1


def main():
    df, df1 = import_data('training_set_VU_DM_2014.csv')
    df1.to_csv('predict_dataset.csv')
    df = normalize_values(df)
    df.to_csv('prepared_train.csv')


if __name__ == '__main__':
    main()