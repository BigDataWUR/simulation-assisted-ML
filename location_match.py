from haversine import haversine_vector, Unit


def get_n_closest_locations(n):
    """Returns the closest 'n' locations to each of the eight available locations, ordered based on increasing distance.

    Parameters
    ----------
    n: the amount of locations which we want

    Returns
    -------
    Returns a dict where each key is the name of one of the eight locations and values lists of 'n' locations.
    dict[str, list[str]]
    """
    
    # Clim1 - Waiotu
    # Clim2 - Ruakura
    # Clim3 - Wairoa
    # Clim4 - Marton
    # Clim5 - Mahana
    # Clim6 - Kokatahi
    # Clim7 - Lincoln
    # Clim8 - Wyndham

    locations = ['Clim1',
                 'Clim2',
                 'Clim3',
                 'Clim4',
                 'Clim5',
                 'Clim6',
                 'Clim7',
                 'Clim8']

    latitudes = [-35.525,
                 -37.775,
                 -39.035,
                 -40.075,
                 -41.275,
                 -42.825,
                 -43.625,
                 -46.325]

    longitudes = [174.225,
                  175.325,
                  177.359,
                  175.375,
                  173.075,
                  171.025,
                  172.475,
                  168.825]

    tuples = list(map(lambda val1, val2: (val1, val2), latitudes, longitudes))

    haversines = haversine_vector(tuples, tuples, Unit.KILOMETERS, comb=True)

    # distances_dict={}
    closest_dict = {}

    for i in range(len(locations)):
        d = dict(zip(locations, haversines[i]))
        #distances_dict[locations[i]] = dict(zip(locations, haversines[i]))
        closest_dict[locations[i]] = [k for k, v in sorted(
            d.items(), key=lambda item: item[1])][1:n+1]

    return closest_dict


def climate_match(model_type, location_type, location):
    """Returns the closest locations to 'location' based on climate matching (the locations are already matched here)

    Parameters
    ----------
    model_type: str, one of 'known' 'unknown'
    location_type: str, one of 'local' 'regional' 'global'
    location: str, one of 'Clim1' 'Clim2' 'Clim3' 'Clim4' 'Clim5' 'Clim6' 'Clim7' 'Clim8' 

    Returns
    -------
    Returns a list of locations.
    """
    
    match_map = {
        'known':
        {'local': {'Clim1': ['Clim1'],
                   'Clim2': ['Clim2'],
                   'Clim3': ['Clim3'],
                   'Clim4': ['Clim4'],
                   'Clim5': ['Clim5'],
                   'Clim6': ['Clim6'],
                   'Clim7': ['Clim7'],
                   'Clim8': ['Clim8']},
         'regional': {'Clim1': ['Clim2', 'Clim3', 'Clim1'],
                      'Clim2': ['Clim4', 'Clim3', 'Clim2'],
                      'Clim3': ['Clim2', 'Clim1', 'Clim3'],
                      'Clim4': ['Clim2', 'Clim5', 'Clim4'],
                      'Clim5': ['Clim4', 'Clim2', 'Clim5'],
                      'Clim6': ['Clim1', 'Clim3', 'Clim6'],
                      'Clim7': ['Clim5', 'Clim4', 'Clim7'],
                      'Clim8': ['Clim4', 'Clim5', 'Clim8']},
         'global': {'Clim1': ['Clim1', 'Clim2', 'Clim3', 'Clim4', 'Clim5', 'Clim6', 'Clim7', 'Clim8'],
                    'Clim2': ['Clim1', 'Clim2', 'Clim3', 'Clim4', 'Clim5', 'Clim6', 'Clim7', 'Clim8'],
                    'Clim3': ['Clim1', 'Clim2', 'Clim3', 'Clim4', 'Clim5', 'Clim6', 'Clim7', 'Clim8'],
                    'Clim4': ['Clim1', 'Clim2', 'Clim3', 'Clim4', 'Clim5', 'Clim6', 'Clim7', 'Clim8'],
                    'Clim5': ['Clim1', 'Clim2', 'Clim3', 'Clim4', 'Clim5', 'Clim6', 'Clim7', 'Clim8'],
                    'Clim6': ['Clim1', 'Clim2', 'Clim3', 'Clim4', 'Clim5', 'Clim6', 'Clim7', 'Clim8'],
                    'Clim7': ['Clim1', 'Clim2', 'Clim3', 'Clim4', 'Clim5', 'Clim6', 'Clim7', 'Clim8'],
                    'Clim8': ['Clim1', 'Clim2', 'Clim3', 'Clim4', 'Clim5', 'Clim6', 'Clim7', 'Clim8']}},
        'unknown':
        {'local': {'Clim1': ['Clim3'],
                   'Clim2': ['Clim4'],
                   'Clim3': ['Clim2'],
                   'Clim4': ['Clim5'],
                   'Clim5': ['Clim4'],
                   'Clim6': ['Clim1'],
                   'Clim7': ['Clim5'],
                   'Clim8': ['Clim4']},
         'regional': {'Clim1': ['Clim2', 'Clim3', 'Clim4'],
                      'Clim2': ['Clim4', 'Clim3', 'Clim5'],
                      'Clim3': ['Clim2', 'Clim1', 'Clim4'],
                      'Clim4': ['Clim2', 'Clim5', 'Clim8'],
                      'Clim5': ['Clim4', 'Clim2', 'Clim8'],
                      'Clim6': ['Clim1', 'Clim3', 'Clim4'],
                      'Clim7': ['Clim5', 'Clim4', 'Clim8'],
                      'Clim8': ['Clim4', 'Clim5', 'Clim2']},
         'global': {'Clim1': ['Clim2', 'Clim3', 'Clim4', 'Clim5', 'Clim6', 'Clim7', 'Clim8'],
                    'Clim2': ['Clim1', 'Clim3', 'Clim4', 'Clim5', 'Clim6', 'Clim7', 'Clim8'],
                    'Clim3': ['Clim1', 'Clim2', 'Clim4', 'Clim5', 'Clim6', 'Clim7', 'Clim8'],
                    'Clim4': ['Clim1', 'Clim2', 'Clim3', 'Clim5', 'Clim6', 'Clim7', 'Clim8'],
                    'Clim5': ['Clim1', 'Clim2', 'Clim3', 'Clim4', 'Clim6', 'Clim7', 'Clim8'],
                    'Clim6': ['Clim1', 'Clim2', 'Clim3', 'Clim4', 'Clim5', 'Clim7', 'Clim8'],
                    'Clim7': ['Clim1', 'Clim2', 'Clim3', 'Clim4', 'Clim5', 'Clim6', 'Clim8'],
                    'Clim8': ['Clim1', 'Clim2', 'Clim3', 'Clim4', 'Clim5', 'Clim6', 'Clim7']}}}

    return match_map[location_type][model_type][location]
