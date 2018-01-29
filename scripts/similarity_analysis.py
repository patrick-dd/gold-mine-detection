"""

Performs similarity matrix on image patches

Returns a csv file with the approximate lat, lon and the ten closest images

STATUS: hacky -- uses json created locally because postgres database not set up

"""

import os
import json
import numpy as np
from scipy import spatial

print(__doc__)


def get_lat_lon_model(some_image, img_row, img_col):
    """
    HACKY SOLUTION

    We've lost access to the database with image geometries

    But we do have a file with lat lons for different image sizes (224 rather
    than 300)

    So we create a linear regression model to get approximate centroids

    """
    max_col_pixels = 2016
    max_row_pixels = 1568
    # LATITUDE
    max_lat = some_image['lats'][-1]
    min_lat = some_image['lats'][0]
    img_row = min_lat + (max_lat-min_lat) * (img_row/max_row_pixels)
    # LONGITUDE
    max_lon = some_image['lons'][-1]
    min_lon = some_image['lons'][0]
    img_col = min_lon + (max_lon-min_lon) * (img_col/max_col_pixels)
    return img_row, img_col


def get_similarities(row, data):
    """
    calculates the cosine similarity between row and each row in data
    inputs
        row :: array-like (1-D)
        data :: array-like (2-D)
    returns
        output :: array-like (1-D)
    """
    output = []
    for r in data:
        output.append(spatial.distance.cosine(row, r))
    return output


def create_ranking_matrix(similarity_matrix):
    """
    creates a matrix where the values are the indices corresponding to
    rankings in the similarity_matrix

        E.g. where the first element of column i is the index of the
        highest value in column i in the similarity_matrix
        The third row of columni is the third highest value, etc.

    input
        similarity_matrix :: n x n numpy array
    returns
        ranking_matrix :: n x n numpy array
    """

    ranking_matrix = []
    for i in range(similarity_matrix.shape[0]):
        ranking_i = [i[0] for i in sorted(enumerate(similarity_matrix[i]),
                                          key=lambda x:x[1],
                                          reverse=True)]
        ranking_matrix.append(ranking_i)
    # return ranking matrix without the first column -- best match is itself
    out_matrix = [row[1:] for row in ranking_matrix]
    return out_matrix


def get_missing(array, first=True):
    """
    returns the ids of similar looking images that we haven't collected yet
    array is
    | ID | row | col | filename | lat | lon | X_1 | ... | X_n |

    where
        ID is the image id
        row is the row pixel index of top left corner of subimage
        col is the col pixel index of top left corner of subimage
        filename is the filename of the larger image
        lat is approximate subimage latitude
        lon is approximate subimage longitude
        X_1 ... X_n are IDs of n most similar subimages

    first is a boolean flag for the first iteration of this

    returns a list of the similar subimage ID that are not in the ID column

    """
    if first:
        existing = set([array[0][0]])
        potential = set(array[0][5:])
    else:
        existing = [int(x[0]) for x in array]
        existing = set(existing)
        tmp = [x[:5] for x in array]
        potential = []
        for row in tmp:
            for x in row:
                potential.append(int(x))
        potential = set(potential)
    missing = potential.difference(existing)
    return list(missing)


if __name__ == "__main__":

    data_dir = '../../data/images/'
    meta_data = []
    hog_features = []
    vgg_features = []
    number_similarities = 10

    features = np.load(open(data_dir + 'vectors.npy', 'rb'))
    meta_data = []
    raw_features = []
    for row in features:
        meta_data.append(row[:3])
        raw_features.append(row[3:])
    del features

    # get metadata
    # This is the hacky bit
    coordinate_data = json.load(
            open('../../data/images/coordinate_data.json', 'r')
            )
    os.remove(data_dir + 'image_rankings.csv')
    header = ['row', 'col',  'filename', 'lat', 'lon'
              ] + ['r_{}'.format(x) for x in range(number_similarities)]
    header = ",".join([str(x) for x in header]) + '\n'
    with open(data_dir + 'image_rankings.csv', 'a') as outfile:
        outfile.write(header)

    data = []

    i = 929361
    print('Image ID: ', meta_data[i][2])
    filename = 'image_{}.jpg'.format(int(meta_data[i][2]))
    img_row = meta_data[i][0]
    img_col = meta_data[i][1]
    this_image = coordinate_data[filename]
    # WHAT'S THE FILENAME
    lat, lon = get_lat_lon_model(this_image, img_row, img_col)
    print('LatLon {}, {}'.format(lat, lon))
    tmp_md = list(meta_data[i]) + [lat, lon]
    # the first three rows are metadata
    # the last row is a nan
    tmp_features = raw_features[i]
    similarities = get_similarities(
            tmp_features,
            raw_features)
    rankings = [s[0] for s in sorted(
        enumerate(similarities),
        key=lambda x:x[1],
        reverse=True)]
    # only store closest rankings
    rankings = rankings[:number_similarities]
    tmp = list(tmp_md) + list(rankings)
    data.append(tmp)
    tmp = ",".join([str(x) for x in tmp]) + '\n'
    with open(data_dir + 'image_rankings.csv', 'a') as outfile:
        outfile.write(tmp)

    missing = get_missing(data, first=True)
    missing = [int(i) for i in missing]
    for i in missing:
        print('Image ID: ', meta_data[i][2])
        print('Subimage ID: ', i)
        filename = 'image_{}.jpg'.format(int(meta_data[i][2]))
        img_row = meta_data[i][0]
        img_col = meta_data[i][1]
        this_image = coordinate_data[filename]
        # WHAT'S THE FILENAME
        lat, lon = get_lat_lon_model(this_image, img_row, img_col)
        print('LatLon {}, {}'.format(lat, lon))
        # lat = longlat['lats'][img_row]
        # lon = longlat['lons'][img_col]
        tmp_md = list(meta_data[i]) + [lat, lon]
        # the first three rows are metadata
        # the last row is a nan
        tmp_features = raw_features[i]
        similarities = get_similarities(
                tmp_features,
                raw_features)
        rankings = [s[0] for s in sorted(
            enumerate(similarities),
            key=lambda x:x[1],
            reverse=True)]
        # only store closest rankings
        rankings = rankings[:number_similarities]
        tmp = list(tmp_md) + list(rankings)
        data.append(tmp)
        tmp = ",".join([str(x) for x in tmp]) + '\n'
        with open(data_dir + 'image_rankings.csv', 'a') as outfile:
            outfile.write(tmp)

    missing = get_missing(data, first=False)
    missing = [int(i) for i in missing]
    for i in missing:
        print('Image ID: ', meta_data[i][2])
        print('Subimage ID: ', i)
        filename = 'image_{}.jpg'.format(int(meta_data[i][2]))
        img_row = meta_data[i][0]
        img_col = meta_data[i][1]
        this_image = coordinate_data[filename]
        # WHAT'S THE FILENAME
        lat, lon = get_lat_lon_model(this_image, img_row, img_col)
        tmp_md = list(meta_data[i]) + [lat, lon]
        # the first three rows are metadata
        # the last row is a nan
        tmp_features = raw_features[i]
        similarities = get_similarities(
                tmp_features,
                raw_features)
        rankings = [s[0] for s in sorted(
            enumerate(similarities),
            key=lambda x:x[1],
            reverse=True)]
        # only store closest rankings
        rankings = rankings[:number_similarities]
        tmp = list(tmp_md) + list(rankings)
        data.append(tmp)
        tmp = ",".join([str(x) for x in tmp]) + '\n'
        with open(data_dir + 'image_rankings.csv', 'a') as outfile:
            outfile.write(tmp)

    missing = get_missing(data, first=False)
    missing = [int(i) for i in missing]
    for i in missing:
        print('Image ID: ', meta_data[i][2])
        print('Subimage ID: ', i)
        filename = 'image_{}.jpg'.format(int(meta_data[i][2]))
        img_row = meta_data[i][0]
        img_col = meta_data[i][1]
        this_image = coordinate_data[filename]
        # WHAT'S THE FILENAME
        lat, lon = get_lat_lon_model(this_image, img_row, img_col)
        print('LatLon {}, {}'.format(lat, lon))
        tmp_md = list(meta_data[i]) + [lat[0], lon[0]]
        # the first three rows are metadata
        # the last row is a nan
        tmp_features = raw_features[i]
        similarities = get_similarities(
                tmp_features,
                raw_features)
        rankings = [s[0] for s in sorted(
            enumerate(similarities),
            key=lambda x:x[1],
            reverse=True)]
        # only store closest rankings
        rankings = rankings[:number_similarities]
        tmp = list(tmp_md) + list(rankings)
        data.append(tmp)
        tmp = ",".join([str(x) for x in tmp]) + '\n'
        with open(data_dir + 'image_rankings.csv', 'a') as outfile:
            outfile.write(tmp)


# EOF
