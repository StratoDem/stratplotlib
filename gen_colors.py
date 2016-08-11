import numpy as np
import pandas


def gen_group_colors(groups):
    """
    Maps a set of groups to a color unique to each group.
    Supports up to 128 groups.

    Parameters
    ----------
    groups: Iterable
        list of groups

    Returns
    -------
    dict:
        Dictionary with the structure {group_n: color_n}
    """
    color_pool = [
        '#DF3E3E', '#258686', '#9ED03A', '#BE356F', '#355DBE', '#886F4C', '#0086ED', '#D0AC94',
        '#7ED379', '#1CE6FF', '#FF34FF', '#FF4A46', '#008941', '#006FA6', '#A30059', '#922329',
        '#FFDBE5', '#7A4900', '#0000A6', '#63FFAC', '#B79762', '#004D43', '#8FB0FF', '#997D87',
        '#5A0007', '#809693', '#FEFFE6', '#1B4400', '#4FC601', '#3B5DFF', '#4A3B53', '#FF2F80',
        '#61615A', '#BA0900', '#6B7900', '#00C2A0', '#FFAA92', '#FF90C9', '#B903AA', '#D16100',
        '#DDEFFF', '#000035', '#7B4F4B', '#A1C299', '#300018', '#0AA6D8', '#013349', '#00846F',
        '#372101', '#FFB500', '#C2FFED', '#A079BF', '#CC0744', '#C0B9B2', '#C2FF99', '#001E09',
        '#00489C', '#6F0062', '#0CBD66', '#EEC3FF', '#456D75', '#B77B68', '#7A87A1', '#788D66',
        '#885578', '#FAD09F', '#FF8A9A', '#D157A0', '#BEC459', '#456648', '#FFFF00', '#012C58',
        '#34362D', '#B4A8BD', '#00A6AA', '#452C2C', '#636375', '#A3C8C9', '#FF913F', '#938A81',
        '#575329', '#00FECF', '#B05B6F', '#8CD0FF', '#3B9700', '#04F757', '#C8A1A1', '#1E6E00',
        '#7900D7', '#A77500', '#6367A9', '#A05837', '#6B002C', '#772600', '#D790FF', '#9B9700',
        '#549E79', '#FFF69F', '#201625', '#72418F', '#BC23FF', '#99ADC0', '#3A2465', '#000000',
        '#5B4534', '#FDE8DC', '#404E55', '#0089A3', '#CB7E98', '#A4E804', '#324E72', '#6A3A4C',
        '#83AB58', '#001C1E', '#D1F7CE', '#004B28', '#C8D0F6', '#A3A489', '#806C66', '#222800',
        '#BF5650', '#E83000', '#66796D', '#DA007C', '#FF1A59', '#8ADBB4', '#1E0200', '#5B4E51',
        '#C895C5', '#320033', '#FF6832', '#66E1D3', '#CFCDAC']

    groups = pandas.Series(groups)
    unique_groups = np.sort(groups.unique())

    assert len(unique_groups) <= len(color_pool), \
        'Too many  groups specified. Maximum of 128 groups supported'
    dict_group_to_color = {unique_groups[i]: color_pool[i] for i in range(len(unique_groups))}
    return dict_group_to_color


def gen_color_column(ds):
    """
    Takes a pandas.DataFrame with a group column and maps each group to a color.

    Parameters
    ----------
    ds: pandas.Series

    Returns
    -------
    pandas.Series
        A pandas Series with colors corresponding to each group
    """

    assert isinstance(ds, pandas.Series)
    group_to_color_dict = gen_group_colors(ds)
    assert isinstance(group_to_color_dict, dict)

    return ds.map(lambda x: group_to_color_dict[x])

