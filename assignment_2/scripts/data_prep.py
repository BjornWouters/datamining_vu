import numpy as np
from scipy.stats import mstats


def import_data(filename): #Import and read file
    file = open(filename, 'r')
    file = file.read()
    return file


def check_booking(file):
    # Prepare fle for reading
    file = file.split("\n")
    search = {}
    booking_file = {}
    calculation = [[], [], [], [], [], [], [], [], [], [], []]
    for line in file:
        line = line.split(',')
        if line[0] == 'srch_id': # Skip first line
            continue
        # Delete the non-booked searches
        id = int(line[0]) # Search ID
        book = int(line[53]) # Booked or not (1/0)
        if id in search.keys():
            search[id] += book
        else:
            search[id] = book
            booking_file[id] = []
        short_line, calculation = missing_values(line, calculation)
        booking_file[id].append(short_line)
    for key in search.keys():
        if search[key] != 1:
            del booking_file[key]
    return booking_file, calculation


def missing_values(line, calculation): # Change missing values to 0
    short_line = []
    for i in range(6, 17):
        if line[i] == 'NULL' or line[i] == '':
            short_line.append(0)
        else:
            short_line.append(float(line[i]))
            calculation[i-6].append(float(line[i])) # Add values to matrix for use in z-scores
    return short_line, calculation


def property_data(booking_dict, calculation): # Sort data per property
    prop_dict = {}
    for key in booking_dict.keys():
        for i in range(len(booking_dict[key])):
            hotel = booking_dict[key][i]
            if hotel[1] not in prop_dict.keys():
                prop_dict[hotel[1]] = []
            prop_dict[hotel[1]].append(hotel)
    for key in prop_dict:
        prop_dict[key] = average_hotel(prop_dict[key], calculation)
    return prop_dict


def average_hotel(hotel, calculation): # Calculate the average per hotel
    av_hotel = [[], [], [], [], [], [], [], [], [], [], []]
    searches = len(hotel)
    for search in range(searches):
        for i in range(11):
            # corr_hotel = hotel[search][i]     # Maak actief voor zonder z-score
            corr_hotel = z_score_corr(hotel[search][i], calculation, i)     # Maak inactief voor zonder z-score
            if [corr_hotel] != av_hotel[i]:
                av_hotel[i].append(corr_hotel)
    for j in range(11):
        av_hotel[j] = str(np.average(av_hotel[j]))
    return av_hotel


def z_score_corr(value, calculation, i): # Normalise the data by deviding by z-score
    mean = np.mean(calculation[i])
    sd = np.std(calculation[i])
    corr_value = value / ((value - mean)/sd)
    return corr_value


def write_file(dict): # Write all new data to file
    file = open('prep_small_train.txt', 'w')
    file.write("prop_id" + "\t" + "prop_country_id" + "\t" + "prop_starrating" + "\t" + "prop_review_score" + "\t" + "prop_brand_bool" + "\t" + "prop_location_score1" + "\t" + "prop_location_score2" + "\t" + "prop_log_historical_price" + "\t" + "position" + "\t" + "price_usd,promotion_flag" + "\n")
    for key in sorted(dict.keys()):
        line = '\t'.join([str(key)] + dict[key])
        file.write(line + '\n')


def main():
    file = import_data('small_train.csv')   # Import data
    booking_dict, calc_list = check_booking(file)   # Delete non-booked searches
    prop_dict = property_data(booking_dict, calc_list)  # Sort by hotel and normalise with z-score
    write_file(prop_dict)   # Write result to file

if __name__ == '__main__':
    main()
