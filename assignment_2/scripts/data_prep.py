import numpy as np


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


def missing_values(line, calculation):  # Change missing values to 0
    short_line = []
    for i in [7, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        if line[i] == 'NULL' or line[i] == '':
            short_line.append(0)
        else:
            short_line.append(float(line[i]))
            calculation[i-6].append(float(line[i]))  # Add values to matrix for use in z-scores
    return short_line, calculation


def property_data(booking_dict, calculation):  # Sort data per property
    prop_dict = {}
    for key in booking_dict.keys():
        for i in range(len(booking_dict[key])):
            hotel = booking_dict[key][i]
            if hotel[0] not in prop_dict.keys():
                prop_dict[hotel[0]] = []
            prop_dict[hotel[0]].append(hotel)
    for key in prop_dict:
        prop_dict[key] = average_hotel(prop_dict[key], calculation)
    return prop_dict


def average_hotel(hotel, calculation): # Calculate the average per hotel
    av_hotel = [[], [], [], [], [], [], [], [], [], [], []]
    searches = len(hotel)
    need_norm = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Choose features for z-score normalisation
    # 0: prop_id; 1: prop_country_id; 2: prop_starrating; 3: prop_review_score;
    # 4: prop_brand_bool; 5: prop_location_score1; 6:prop_location_score2; 7: prop_log_historical_price;
    # 8: position; 9: price_usd; 10: promotion_flag
    for search in range(searches):
        for i in range(11):
            if i in need_norm:
                corr_hotel = z_score_corr(hotel[search][i], calculation, i)  # Only features in need_norm get normalised
            else:
                corr_hotel = hotel[search][i]
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
    file = open('../results/prep_small_train.csv', 'w')
    file.write("prop_id" + "," + "prop_country_id" + "," + "prop_starrating" + "," + "prop_review_score" + "," + "prop_brand_bool" + "," + "prop_location_score1" + "," + "prop_location_score2" + "," + "prop_log_historical_price" + "," + "position" + "," + "price_usd" + "," + "promotion_flag" + "\n")
    for key in sorted(dict.keys()):
        line = ','.join(dict[key])
        file.write(line + '\n')


def main():
    file = import_data('../data/small_train.csv')   # Import data
    booking_dict, calc_list = check_booking(file)   # Delete non-booked searches
    prop_dict = property_data(booking_dict, calc_list)  # Sort by hotel and normalise with z-score
    write_file(prop_dict)   # Write result to file


if __name__ == '__main__':
    main()
