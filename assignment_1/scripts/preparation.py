from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np


def plot_mood(df, patients):
    # Create new df
    df_new = pd.DataFrame()
    # With the range of dates (2014-02-26 - 2014-06-08 = 103 days)
    date_range = pd.date_range('2014-02-26', periods=103, freq='D')
    df_new['date'] = date_range
    df.date = df.index

    # Only filter on the mood
    df = df.loc[df.variable == 'mood']
    for patient in patients[1:]:
        patient_df = df.loc[df.id == patient]
        df_deduplicated = patient_df[~patient_df.index.duplicated(keep='first')]
        mean_mood = df_deduplicated.value.resample('D').mean()
        # To DF
        mean_mood = pd.DataFrame(mean_mood)
        mean_mood['date'] = mean_mood.index
        # Set mood to patient name
        mean_mood[patient] = mean_mood['value']
        del mean_mood['value']

        # Single patient purposes
        # del mean_mood['date']
        # mean_mood.plot()
        # plt.show()
        # import sys; sys.exit()

        # Combine new df with each other
        df_new = pd.merge_ordered(df_new, mean_mood, on='date')

    del df_new['date']
    df_new.plot(legend=False)
    plt.show()


def prepare_data(patient, df):
    patient1_data = df.loc[df.id == patient]

    # Get all the values in a specific timestamp
    # Remove duplicate timestamps as this shouldn't be possible
    patient1_data = patient1_data[~patient1_data.index.duplicated(keep='first')]
    patient1_data.date = patient1_data.index

    # Create new dataframe where the preprocessed data is going in to
    df_new = pd.DataFrame()
    # With the range of dates (2014-02-26 - 2014-06-08 = 103 days)
    date_range = pd.date_range('2014-02-26', periods=103, freq='D')
    df_new['date'] = date_range

    # Get all the different variables
    variables = patient1_data.variable.unique()
    for variable in variables:
        # Call and sms are counted so first sum them up
        if not variable == 'mood':
            mean_var = patient1_data.loc[patient1_data.variable == variable].value.resample('D').mean()
            # Normalize with the Z-score
            mean_var = (mean_var - mean_var.mean()) / mean_var.std(ddof=0)
        else:
            # Substract 1 day for normal classifier purposes
            patient1_data.index = patient1_data.index - np.timedelta64(1, 'D')
            mean_var = patient1_data.loc[patient1_data.variable == variable].value.resample('D').mean()
            # Reset normal index
            patient1_data.index = patient1_data.index + np.timedelta64(1, 'D')
        # To DF
        mean_var = pd.DataFrame(mean_var)
        mean_var['date'] = mean_var.index
        # Set mood to patient name
        mean_var[variable] = mean_var['value']
        del mean_var['value']

        # Combine new df with each other
        df_new = pd.merge_ordered(df_new, mean_var, on='date')
       
        
    #Drop rows that contain less than 5 observations
    df_new = df_new.dropna(thresh=5)
    #Drop rows where there is no mood value and replace all NaN for 0
    df_new = df_new.dropna(subset=['mood']).fillna(0)
    return df_new[df_new.mood.notnull()]


def main():
    dataset = '../data/dataset_mood_smartphone.csv'
    df = pd.read_csv(dataset)
    # Convert all timestamp strings to timestamped objects
    df['time'] = df['time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    df = df.set_index('time')

    patients = df.id.unique()
    for patient in patients:
        df_prepared = prepare_data(patient, df)
        df_prepared.to_csv('../results/normalized_normal/{}.csv'.format(patient), index_label=False, index=False)
    # plot_mood(df, patients)
        print(df_prepared)
if __name__ == '__main__':
    main()
