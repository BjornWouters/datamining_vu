import pandas as pd


def calculate_score(listings):
    sum_score = 0
    for i, row in enumerate(listings.iterrows(), start=1):
        index, listing = row
        score = 0
        if listing.booking_bool == 1:
            score = 5
        elif listing.click_bool == 1:
            score = 1

        if score == 0:
            discount_score = 0
        else:
            discount_score = score/i

        sum_score += discount_score

    return sum_score


def max_scores(df):
    score_list = list()
    id_list = list()
    srch_ids = df.srch_id.unique()
    for id in srch_ids:
        listings = df[df.srch_id == id]
        sorted_listings = listings.sort_values(by=['booking_bool', 'click_bool'], ascending=False)
        score = calculate_score(sorted_listings)

        id_list.append(id)
        score_list.append(score)

    max_df = pd.DataFrame(data={'listing_id': id_list, 'score': score_list})
    max_df.to_csv('../data/max_scores.csv', index=False)


def main():
    train_file = '../data/small_train.csv'
    df = pd.read_csv(train_file)
    max_scores(df)


if __name__ == '__main__':
    main()
