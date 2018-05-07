from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import seaborn as sns


def import_data(filename):
    df = pd.read_csv(filename)
    return df


def get_booking_numbers(df):
    unique_search_ids = df.srch_id.unique().size
    booked_ids = len(df[df.booking_bool == 1])
    # Check for double bookings
    unique_booked_ids = len(df[df.booking_bool == 1].srch_id.unique())
    print('Unique search: {}\nBooked ids: {}\nUnique booked ids: {}'.format(
        unique_search_ids, booked_ids, unique_booked_ids))

    unique_hotels = df.prop_id.unique().size
    print('Unique hotels: {}'.format(unique_hotels))


def plot_time_dependence(df):
    # Only booked instances
    df = df


def plot_showed_properties(df):
    sns.countplot(df.prop_id.value_counts(), orient='h')
    print('Highest: {}'.format(df.prop_id.value_counts().max()))
    plt.show()


def plot_bookings_per_hotel(df):
    # Count the booking per hotel
    # Add the non-booked hotels

    # Include the non-booked as well
    # non_booked = df[df.booking_bool == 0]
    # non_booked = non_booked.set_index('prop_id')
    # booked = df[df.booking_bool == 1].prop_id.value_counts().append(
    #     non_booked.booking_bool)
    booked = df[df.booking_bool == 1].prop_id.value_counts()
    print('Highest: {}'.format(booked.max()))
    sns.countplot(booked, orient='h')
    plt.show()


def plot_correlation_show_buy(df):
    showed_hotels = df.prop_id.value_counts()
    showed_hotels.name = 'showed_hotels'
    booked_hotels = df[df.booking_bool == 1].prop_id.value_counts()
    booked_hotels.name = 'booked_hotels'
    compare_df = pd.concat([showed_hotels, booked_hotels], axis=1)
    compare_df = compare_df.fillna(0)

    # Calculate the point density
    xy = np.vstack([compare_df.showed_hotels, compare_df.booked_hotels])
    z = gaussian_kde(xy)(xy)
    m, b = np.polyfit(compare_df.showed_hotels, compare_df.booked_hotels, 1)

    fig, ax = plt.subplots()
    ax.scatter(compare_df.showed_hotels, compare_df.booked_hotels, c=z, s=100, edgecolor='')
    plt.plot(compare_df.showed_hotels, m * compare_df.showed_hotels + b, '-')
    plt.ylabel('Number of bookings')
    plt.xlabel('Number of listings')
    plt.show()


def calc_different_hotel_listings(df_train, df_test):
    listings_train = df_train.prop_id.unique()
    listings_test = df_test.prop_id.unique()
    print('Number of unique listings: {}'.format(len(
        np.setdiff1d(listings_test, listings_train))))


def plot_booked_time_distribution(df):
    booked = df[df.booking_bool == 1]
    booked.date_time = df.date_time.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    booked['hour'] = booked.date_time.dt.hour
    booked['day'] = booked.date_time.dt.day
    booked['month'] = booked.date_time.dt.month

    hourly = booked.groupby(['prop_id'])['hour'].median().astype(int)
    daily = booked.groupby(['prop_id'])['day'].median().astype(int)
    monthly = booked.groupby(['prop_id'])['month'].median().astype(int)

    fig, axs = plt.subplots(nrows=3)
    sns.countplot(hourly, orient='h', ax=axs[0])
    sns.countplot(daily, orient='h', ax=axs[1])
    sns.countplot(monthly, orient='h', ax=axs[2])
    plt.suptitle('Differences in bookings per timestamp', y=1)
    plt.show()


def plot_listing_properties(df):
    booked = df[df.booking_bool == 1]
    non_booked = df[df.booking_bool == 0]

    # Starrating
    # relative_booked = booked.prop_starrating.value_counts()/booked.prop_starrating.value_counts().sum()
    # relative_non_booked = non_booked.prop_starrating.value_counts()/non_booked.prop_starrating.value_counts().sum()

    # Review score
    # relative_booked = booked.prop_review_score.value_counts()/booked.prop_review_score.value_counts().sum()
    # relative_non_booked = non_booked.prop_review_score.value_counts()/non_booked.prop_review_score.value_counts().sum()

    # Major hotel chain
    # relative_booked = booked.prop_brand_bool.value_counts()/booked.prop_brand_bool.value_counts().sum()
    # relative_non_booked = non_booked.prop_brand_bool.value_counts()/non_booked.prop_brand_bool.value_counts().sum()

    # Location score 1
    # relative_booked = booked['prop_location_score2'].value_counts(bins=7, normalize=True).sort_index().reindex(range(0, 7))
    # relative_non_booked = non_booked['prop_location_score2'].value_counts(bins=7, normalize=True).sort_index().reindex(range(0, 7))

    # Location score 2
    # relative_booked = booked['prop_location_score2'].value_counts(bins=10, normalize=True).sort_index()
    # relative_non_booked = non_booked['prop_location_score2'].value_counts(bins=10, normalize=True).sort_index()

    # Log historical price
    # relative_booked = booked['prop_log_historical_price'].value_counts(bins=10, normalize=True).sort_index().reindex(range(0, 7))
    # relative_non_booked = non_booked['prop_log_historical_price'].value_counts(bins=10, normalize=True).sort_index().reindex(range(0, 7))

    # Log historical price
    # relative_booked = booked['price_usd'].value_counts(bins=10, normalize=True).sort_index().reindex(range(0, 1000, 100))
    # relative_non_booked = non_booked['price_usd'].value_counts(bins=10, normalize=True).sort_index().reindex(range(0, 1000, 100))

    # Promotion flag
    relative_booked = booked.promotion_flag.value_counts()/booked.promotion_flag.value_counts().sum()
    relative_non_booked = non_booked.promotion_flag.value_counts()/non_booked.promotion_flag.value_counts().sum()

    additional_plot = relative_booked/relative_non_booked
    additional_plot.plot(kind='bar')
    plt.xlabel('Promotion flag (yes=1, no=0)')
    plt.ylabel('additional factor')
    plt.suptitle('Promotion flag vs additional bookings', y=1)
    plt.show()


def main():
    df_train = import_data('../data/small_train.csv')
    # df_test = import_data('../data/small_train.csv')
    # get_booking_numbers(df_train)
    # plot_time_dependence(df_train)
    # plot_showed_properties(df_train)
    # plot_bookings_per_hotel(df_train)
    # plot_correlation_show_buy(df_train)
    # calc_different_hotel_listings(df_train, df_test)
    # plot_booked_time_distribution(df_train)
    plot_listing_properties(df_train)


if __name__ == '__main__':
    main()