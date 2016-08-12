import json

import shapely.geometry
from shapely.geometry import MultiPolygon
import shapely.wkt

import pandas
import geopandas

import folium
from folium import PolyLine
from folium.plugins import MarkerCluster

from sklearn.preprocessing import MinMaxScaler

from gen_colors import gen_color_column


class StratoMap:
    """StratoMap object built on top of folium and geopandas"""

    def __init__(self, location=None,
                 tiles='http://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
                 attr='&copy; <a href="https://carto.com/attributions">CARTO</a>'
                      ' | &copy; <a href="https://stratodem.com">StratoDem Analytics</a>',
                 width='100%', height='100%', zoom_start=1, min_zoom=1, max_zoom=18):
        """
        The StratoMap object that initializes a folium map with extended settings

        Parameters
        ----------
        location: tuple or list, default None
            lat-lng to start the map at
        tiles: str, default CARTO positron
            What tiles to use, can be a folium-supported str or an api-style url
        attr: str, default CARTO and StratoDem attribution
            Attribution to append to the OSM default
        width: pixel int or percentage str, default '100%'
        height: pixel int or percentage str, default '100%'
        zoom_start: int, default 1
            Starting level of the zoom
        min_zoom: int, default 1
            Minimum zoom level allowed
        max_zoom: int, default 1
            Maximum zoom level allowed
        """
        if location is not None:
            if not (isinstance(location, tuple) or isinstance(location, list)):
                raise TypeError('location must be a tuple or list of length 2 [lat, lng]')
            if not len(location) == 2:
                raise ValueError('location must be a list of length 2 [lat, lng]')
            if not all((isinstance(val, float) or isinstance(val, int)) for val in location):
                raise ValueError('location must be a list of floats or ints of length 2 [lat, lng]')
        if not isinstance(tiles, str):
            raise TypeError('tiles must be a str')
        if not isinstance(attr, str):
            raise TypeError('attr must be a str')
        if not isinstance(width, str) or isinstance(width, int):
            raise TypeError('width must be a str (e.g. \'90%\' of page) or int (e.g. 500 pixels)')
        if not isinstance(height, str) or isinstance(height, int):
            raise TypeError('height must be a str (e.g. \'90%\' of page) or int (e.g. 500 pixels)')
        if not isinstance(zoom_start, int):
            raise TypeError('zoom_start must be an int > 0')
        if not zoom_start > 0:
            raise ValueError('zoom_start must be an int > 0')
        if not isinstance(min_zoom, int):
            raise TypeError('min_zoom must be an int > 0')
        if not min_zoom > 0:
            raise ValueError('min_zoom must be an int > 0')
        if not isinstance(max_zoom, int):
            raise TypeError('max_zoom must be an int > 0')
        if not max_zoom > 0:
            raise ValueError('max_zoom must be an int > 0')

        self.map = folium.Map(location=location, tiles=tiles, attr=attr, width=width, height=height,
                              zoom_start=zoom_start, min_zoom=min_zoom, max_zoom=max_zoom)

    def save(self, output_path='_map.html'):
        """
        Saves the underlying mapping object as an html file into the output path specified

        Parameters
        ----------
        output_path: str, default '_map.html'
            output file path

        Returns
        -------
        """
        self.map.save(output_path)

    def show(self):
        """
        Displays the map

        Returns
        -------
        StratoMap
        """
        return self.map

    def choropleth(self, df, value_col, geometry_col='GEOMETRY', popup_col=None,
                   threshold_scale=None, palette='YlOrRd', fill_opacity=0.85, zoom_start=7,
                   line_opacity=0.5, line_color='white'):
        """
        Add a choropleth from a pandas.DataFrame to the StratoMap object
        This assumes that every row is unique

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame for which to create the choropleth, has a value_col and geometry_col
        value_col: str
            Name of the column in df that contains the desired choropleth values
        geometry_col: str, default 'GEOMETRY'
            Name of the column in df that contains the GeoDataFrame's geometries
        popup_col: str
            Name of the column in df that contains the popup text.
            If None, marker functionality is not enabled
        threshold_scale: list, default None
            List of cutoffs to use for the coloring scale
        palette: str, default 'YlOrRd'
            Color-scheme for the choropleth
            OPTIONS:
            BuGn, BuPu, GnBu,
            OrRd, PuBuGn, PuRd,
            RdPu, YlGn, YlGnBu,
            YlOrBr, YlOrBr, YlOrRd
        fill_opacity: float, default 0.85
            Opacity of the choropleth layer, also known as 'alpha'
        zoom_start: int, default 7
            Initial zoom for the choropleth map when displayed (overrides init)
        line_opacity: float, default 0.5
            Opacity of the lines in the choropleth layer, also known as 'alpha'
        line_color: str, default 'white'
            Color of the choropleth line, given in CSS name or hex code

        Returns
        -------
        folium.Map
        """
        if not isinstance(df, pandas.DataFrame):
            raise TypeError('Provided df is not a pandas.DataFrame')
        if geometry_col not in df.columns:
            raise ValueError('geometry_col must be in df.columns')
        if value_col not in df.columns:
            raise ValueError('value_col must be in df.columns')
        if threshold_scale is not None:
            if not isinstance(threshold_scale, list):
                raise TypeError('threshold_scale must be a list of floats')
            else:
                if not all((isinstance(val, int) or isinstance(val, float))
                           for val in threshold_scale):
                    raise TypeError('threshold_scale must be a list of floats')
        if not isinstance(palette, str) or palette not in ('BuGn', 'BuPu', 'GnBu', 'OrRd',
                                                           'PuBuGn', 'PuRd', 'RdPu', 'YlGn',
                                                           'YlGnBu', 'YlOrBr', 'YlOrBr', 'YlOrRd'):
            raise TypeError('palette must be a str,\noptions are: BuGn, BuPu, GnBu, OrRd, PuBuGn, '
                            'PuRd, RdPu, YlGn, YlGnBu, YlOrBr, YlOrBr, YlOrRd')
        if not isinstance(fill_opacity, float) and not (0 <= fill_opacity <= 1):
            raise TypeError('opacity must be a float between 0 and 1')
        if not isinstance(line_opacity, float) and not (0 <= line_opacity <= 1):
            raise TypeError('line_opacity must be a float between 0 and 1')
        if not isinstance(zoom_start, int):
            raise TypeError('zoom_start must be an int > 0')
        if not zoom_start > 0:
            raise ValueError('zoom_start must be an int > 0')
        if not isinstance(line_color, str):
            raise TypeError('line_color must be a CSS supported color name or code')

        df = df.copy()
        assert isinstance(df, pandas.DataFrame)

        # STANDARDIZE GEOMETRY
        df['geometry'] = self._get_geometry(df[geometry_col])
        df = self._clean_geometry(df)
        df = geopandas.GeoDataFrame(df)

        df_centroid = self._get_df_centroid(df)

        df_plot = df[['geometry', value_col]].copy()
        df_plot['geo_col'] = range(1, 1 + len(df_plot))

        self.map.choropleth(geo_str=json.loads(df_plot.to_json()),
                            data=df_plot[['geo_col', value_col]], columns=['geo_col', value_col],
                            key_on='feature.properties.geo_col',
                            fill_color=palette, fill_opacity=fill_opacity, line_opacity=line_opacity,
                            line_color=line_color, legend_name=value_col,
                            threshold_scale=threshold_scale)

        if popup_col:
            self.map.add_children(MarkerCluster(locations=df_centroid['yx'],
                                                popups=df[popup_col].astype(str)))

        bounds = self._get_bounds(df_centroid['x'], df_centroid['y'])
        self.map.fit_bounds(bounds)

        return self.show()

    def groupmap(self, df, group_col=None, popup_col=None, geometry_col='GEOMETRY',
                 zoom_start=7, fill_opacity=0.65, line_color='white', line_width=1.5):
        """
        Plots and colors shapely.geometry objects based on a group ID.
        Group ID can be any hashable identifier

        Parameters
        ----------
        df: pandas.DataFrame
            Dataframe containing geometry and group ID data
        group_col: str
            Name of the column in df containing the group ID data
        popup_col: str
            Name of the column in df containing the popup text for markers
            If None, no markers will be shown
        geometry_col: str
            The name of the column in df containing the shapely.geometry objects
        zoom_start: int
            The zoom level in which the map is initially displayed
        fill_opacity: float
            Opacity of the color layer, also referred to as alpha
        line_color: str
            Color of the geometry boundary lines
        line_width: float
            Width of the geometry boundary lines

        Returns
        -------
        folium.Map
        """
        if not isinstance(df, pandas.DataFrame):
            raise TypeError('Provided df is not a pandas.DataFrame')
        if group_col is not None and group_col not in df.columns:
            raise ValueError('group_col must be in df.columns')
        if popup_col and popup_col not in df.columns:
            raise ValueError('popup_col must be in df.columns')
        if geometry_col not in df.columns:
            raise ValueError('geometry_col must be in df.columns')
        if not isinstance(zoom_start, int):
            raise TypeError('zoom_start must be an int > 0')
        if not zoom_start > 0:
            raise ValueError('zoom_start must be an int > 0')
        if not (isinstance(fill_opacity, int) or isinstance(fill_opacity, float)):
            raise TypeError('fill_opacity must be a float or an int [0, 1]')
        if not 0 <= fill_opacity <= 1:
            raise ValueError('fill_opacity must be between 0 and 1 (inclusive)')
        if not (isinstance(line_width, int) or isinstance(line_width, float)):
            raise TypeError('line_width must be a float or an int (non-negative)')
        if not line_width >= 0:
            raise ValueError('line_width must be non-negative')

        df = df.copy()
        assert isinstance(df, pandas.DataFrame)

        df['geometry'] = self._get_geometry(df[geometry_col])
        df = self._clean_geometry(df)
        df = geopandas.GeoDataFrame(df)

        df_centroid = self._get_df_centroid(df)

        if group_col:
            df['COLOR'] = gen_color_column(df[group_col])
        else:
            df['COLOR'] = 'blue'

        df['style'] = df['COLOR'].map(
            lambda color: {'fillColor': color,
                           'weight': line_width,
                           'fillOpacity': fill_opacity,
                           'color': line_color,
                           'alpha': 1})

        df = geopandas.GeoDataFrame(df)
        df_json = df[['style', 'geometry']]
        df_json = df_json.to_json()

        folium.GeoJson(df_json).add_to(self.map)

        if popup_col:
            self.map.add_children(
                    MarkerCluster(locations=df_centroid['yx'], popups=df[popup_col].astype(str)))

        bounds = self._get_bounds(df_centroid['x'], df_centroid['y'])
        self.map.fit_bounds(bounds)

        return self.show()

    def marker(self, x, y, radius=4., number_of_sides=5, fill_color='red', color='red', rotation=0, popup=None):
        """
        Plots a single or multiple polygon marker(s) based on x,y coordinates

        Parameters
        ----------
        x: float
                x-coordinate of the polygon
        y: float
            y-coordinate of the polygon
        radius: float: default 4.
                Radius of the polygon
        number_of_sides: int, default int
            Number of sides for the polygon
        fill_color: str, default 'red'
            The color for the interior of the polygon
        color: str, default 'red'
            The color for the ring of the polygon
        rotation: float or int, default 0
        popup: str, default None
            The content to display in the popup window

        Returns
        -------

        """
        x = pandas.Series(x)
        y = pandas.Series(y)
        assert len(x) == len(y)

        df_marker = pandas.DataFrame()
        df_marker['x'] = x
        df_marker['y'] = y

        names = ['COLOR', 'FILL_COLOR', 'RADIUS', 'SIDES', 'POPUP', 'ROTATION']
        for i, elem in enumerate([color, fill_color, radius, number_of_sides, popup, rotation]):
            if elem is None:
                continue
            s = pandas.Series(elem)
            if len(s) > 1:
                assert len(s) == len(df_marker.shape[0])
                df_marker[names[i]] = s
            elif len(s) == 1:
                df_marker[names[i]] = elem
            else:
                raise ValueError

        df_marker.apply(self._add_markers, axis=1)

        return self.show()

    def line(self, source, target, color='blue', line_opacity=0.8, weight=10, popup=None):
        """
        Draws corresponding lines given a list of source and target coordinates

        Parameters
        ----------
        source: tuple or list, or list of tuples or list of lists
            The source coordinates (x,y) . Can also be a list of coordinates
            e.g. [ (x1,y1), (x2,y2) ]
            or [ [x1,y1], [x2,y2] ]
        target: tuple or list, or list of tuples or list of lists
            The target coordinates (x,y) . Can also be a list of coordinates
            e.g. [ (x1,y1), (x2,y2) ]
            or [ [x1,y1], [x2,y2] ]
        color: str or list
            valid color string of the line color. default: 'blue'
        line_opacity: float or list
            opacity of the drawn line(s). Also known as alpha. default: 0.8
        weight: float or list
            Changes the thickness of the drawn line(s). default: 10
        popup: str or list
            Sets the popup text when the line is clicked. default: None

        Returns
        -------

        """
        assert len(source) > 0
        assert len(target) > 0
        assert len(source) == len(target)

        df_paths = pandas.DataFrame()
        df_paths['source'] = pandas.Series(source)
        df_paths['destination'] = pandas.Series(target)

        df_paths['COORDINATES'] = list(zip(df_paths['source'], df_paths['destination']))
        names = ['COLOR', 'OPACITY', 'WEIGHT', 'POPUP']

        for i, x in enumerate([color, line_opacity, weight, popup]):
            if x is None:
                continue
            s = pandas.Series(x)
            if len(s) > 1:
                assert len(s) == len(df_paths['COORDINATES'])
                df_paths[names[i]] = s
            elif len(s) == 1:
                df_paths[names[i]] = x
            else:
                raise ValueError

        df_paths.apply(self._add_paths, axis=1)

        return self.show()

    def path(self, coords, color='blue', line_opacity=0.8, weight=10, popup=None):
        """
        Draws paths for a set of lines in a DataFrame

        Parameters
        ----------
        coords
        color
        line_opacity
        weight
        popup

        Returns
        -------

        """
        assert len(coords) > 0

        df_paths = pandas.DataFrame()
        coordinates = pandas.Series(coords)
        df_paths['COORDINATES'] = coordinates
        names = ['COLOR', 'OPACITY', 'WEIGHT', 'POPUP']

        for i, x in enumerate([color, line_opacity, weight, popup]):
            if x is None:
                continue
            s = pandas.Series(x)
            if len(s) > 1:
                assert len(s) == len(coords)
                df_paths[names[i]] = s
            elif len(s) == 1:
                df_paths[names[i]] = x
            else:
                raise ValueError

        df_paths.apply(self._add_paths, axis=1)

        return self.show()

    def scatter(self, df, x_col, y_col, group_col=None, value_col=None, popup_col=None,
                color='blue', scale=1., fill_opacity=0.8, value_pow=1, zoom_start=7):
        """
        Adds a scatter plot to the map-object given a dataframe with a x and y column.
        Can accomodate different groups by changing colors and different values by changing radius.

        Parameters
        ----------
        df: pandas.DataFrame
        x_col: str
            Name of column in df containing x coordinates
        y_col: str
            Name of column in df containing y coordinates
        group_col: str, default None
            Name of column in df containing a hashable group identifier (determine point color)
        value_col: str, default None
            Name of column in df containing the values (determine point radius)
        popup_col: str, default None
            Name of the column in df containing the popup text
            If None, the marker functionality is not enabled
        color: str, default 'blue'
            Valid color for the scatter points
        scale: float, default 1
            Linearly scales the size of the scattered dots
        fill_opacity: float, default 0.8
            Sets the opacity of the scattered dots
        value_pow: float, default 1
            Raises all scaled values to the provided power.
            Can be used to make scatter point size differences more visible.
        zoom_start: int, default 7
            Zoom level for the initial display of the map
        Returns
        -------
        folium.Map
        """
        if not isinstance(df, pandas.DataFrame):
            raise TypeError('Provided df is not a pandas.DataFrame')
        if x_col not in df.columns:
            raise ValueError('x_col must be in df.columns')
        if y_col not in df.columns:
            raise ValueError('y_col must be in df.columns')
        if group_col and group_col not in df.columns:
            raise ValueError('group_col must be in df.columns')
        if popup_col and popup_col not in df.columns:
            raise ValueError('popup_col must be in df.columns')
        if not (isinstance(value_pow, float) or isinstance(value_pow, int)):
            raise TypeError('value_pow must be a float')
        if not (isinstance(scale, float) or isinstance(scale, int)):
            raise TypeError('scale must be a non-negative float')
        if not scale >= 0:
            raise ValueError('scale must be a non-negative float')
        if not isinstance(zoom_start, int):
            raise TypeError('zoom_start must be an int > 0')
        if not zoom_start > 0:
            raise ValueError('zoom_start must be an int > 0')
        if not (isinstance(fill_opacity, int) or isinstance(fill_opacity, float)):
            raise TypeError('fill_opacity must be a float or an int [0, 1]')
        if not 0 <= fill_opacity <= 1:
            raise ValueError('fill_opacity must be between 0 and 1 (inclusive)')

        df = df.copy()
        assert isinstance(df, pandas.DataFrame)

        if group_col:
            df['COLOR'] = gen_color_column(df[group_col])

        # noinspection PyTypeChecker
        mms = MinMaxScaler(feature_range=(2000*scale, 8000*scale), copy=True)

        if value_col:
            df['VALUES_SCALED'] = mms.fit_transform(df[value_col].reshape(-1, 1))
            if value_pow is not 1:
                df['VALUES_SCALED'] **= value_pow
                df['VALUES_SCALED'] = mms.fit_transform(df[value_col].reshape(-1, 1))
        f = folium.map.FeatureGroup()

        def add_markers(df_marker):
            """Add dots to the map in place"""
            self._add_dots(df_marker, x_col, y_col, group_col, value_col, color,
                           scale, fill_opacity, f)

        df.apply(add_markers, axis=1)

        self.map.add_child(f)

        if popup_col:
            df['yx'] = list(zip(df[y_col], df[x_col]))
            self.map.add_children(MarkerCluster(
                locations=df['yx'], popups=df[popup_col].astype(str)))

        bounds = self._get_bounds(df[x_col], df[y_col])
        self.map.fit_bounds(bounds)

        return self.show()

    def set_tiles(self, tiles):
        """

        Parameters
        ----------
        tiles: str

        Returns
        -------
        folium.Map
        """
        if not isinstance(tiles, str):
            raise TypeError('tiles must be a str')

        self.map.add_tile_layer(tiles=tiles)

        return self.show()

    def reset(self):
        """Wipe the whole map and initialize a new folium.Map"""
        self.map = folium.Map()

    def _add_markers(self, row):
        """
        Adds a single polygon marker

        Parameters
        ----------
        row: pandas.Series
            Row of a dataframe containing 'x', 'y', 'FILL_COLOR', 'COLOR', 'RADIUS', 'SIDES', 'ROTATION', ('POPUP') columns
            x: float
                x-coordinate of the polygon
            y: float
                y-coordinate of the polygon
            radius: float: default 4.
                Radius of the polygon
            number_of_sides: int, default int
                Number of sides for the polygon
            fill_color: str, default 'red'
                The color for the interior of the polygon
            color: str, default 'red'
                The color for the ring of the polygon
            rotation: float or int, default 0
            popup: str, default None
                The content to display in the popup window

        Returns
        -------
        folium.Map
        """
        x = row['x']
        y = row['y']
        fill_color = row['FILL_COLOR']
        color = row['COLOR']
        radius = row['RADIUS']
        num_sides = row['SIDES']
        rotation = row['ROTATION']
        popup = row['POPUP'] if 'POPUP' in row.index else None

        if not (isinstance(x, float) or isinstance(x, int)):
            raise TypeError('x must be a float')
        if not (isinstance(y, float) or isinstance(y, int)):
            raise TypeError('y must be a float')
        if not (isinstance(radius, float) or isinstance(radius, int)):
            raise TypeError('y must be a float')
        if not radius > 0:
            raise ValueError('radius must be greater than 0')
        if not isinstance(num_sides, int):
            raise TypeError('number_of_sides must be an int')
        if not num_sides >= 3:
            raise ValueError('number_of_sides must be greater than or equal to 3')
        if not isinstance(color, str):
            raise TypeError('fill_color must be a color str')
        if not isinstance(color, str):
            raise TypeError('color must be a color str')
        if not isinstance(fill_color, str):
            raise TypeError('color must be a color str')
        if not (isinstance(rotation, float) or isinstance(rotation, int)):
            raise TypeError('rotation must be a float')
        if popup is not None and not isinstance(popup, str):
            raise TypeError('popup must be a str')

        self.map.add_children(folium.RegularPolygonMarker(
            [y, x],
            radius=radius,
            number_of_sides=num_sides,
            fill_color=fill_color,
            color=color,
            rotation=rotation,
            popup=popup))

    def _add_paths(self, row):
        """
        Helper function for paths(). Adds a single line overlay to the map object
        Parameters
        ----------
        row: pandas.Series
        Dataframe row with 'COORDINATES', 'OPACITY', 'WEIGHT', 'COLOR', 'POPUP' columns
            coord_list: list
                list of the coordinates describing the line. Must be more than two sets of points
                e.g. [(y1, x1), (y2, x2), ..., (yn, xn)]
            color: str
                Color for the line
            line_opacity: float
                percent opacity for the drawn line
            weight: int
                sets the thickness of the line
            popup: str
                Popup string of the line
        Returns
        -------

        """
        coord_list = row['COORDINATES']
        opacity = row['OPACITY']
        weight = row['WEIGHT']
        color = row['COLOR']
        popup = row['POPUP'] if 'POPUP' in row.index else None

        if not len(coord_list) >= 2:
            raise ValueError('coord_list should be at least 2 tuples')
        if not all(len(pt) == 2 for pt in coord_list):
            raise ValueError('coord_list must be a list of (lat, lng) tuples')
        if not isinstance(color, str):
            raise TypeError('color must be a str')
        if not isinstance(opacity, float):
            raise TypeError('opacity must be a float')
        if not (isinstance(opacity, float) or isinstance(opacity, int)):
            raise TypeError('opacity must be a number')
        if popup is not None and not isinstance(popup, str):
            raise TypeError('popup must be a str')

        self.map.add_children(
            PolyLine(coord_list, color=color, opacity=opacity, weight=weight, popup=popup))

    def _clean_geometry(self, df):
        """
        Handles Multipolygon geometry objects

        Parameters
        ----------
        df: pandas.DataFrame

        geometry_col: str
            Name of column in df that contains shapely.geometry objects

        Returns
        -------
        df_cleaned: pandas.DataFrame
            Dataframe without Multipolygons
        """

        df_temp = df.copy()
        assert isinstance(df_temp, pandas.DataFrame)
        df_cleaned = df_temp.groupby(
            list(df_temp.dtypes[(df_temp.dtypes == 'int64') |
                                (df_temp.dtypes == 'float64')].index)) \
            .apply(self._handle_multi).reset_index(drop=True)

        return df_cleaned

    def _extract_coordinates(self, ds):
        """
        Takes a pandas.Series with shapely.geometry.polygon.Polygon objects
        and returns a pandas.DataFrame with an x, y and a (x, y) column containing representative
        points for the provided geometries

        Parameters
        ----------
        ds: pandas.Series

        Returns
        -------
        pandas.DataFrame
            DataFrame containing an x, y and xy column
        """
        assert isinstance(ds, pandas.Series)

        df_coord = pandas.DataFrame({
            'x': ds.map(self._get_x).astype(float),
            'y': ds.map(self._get_y).astype(float)
        })
        df_coord['xy'] = list(zip(df_coord['x'], df_coord['y']))

        return df_coord

    @staticmethod
    def _get_geometry(ds):
        """
        Attempts to convert a pandas.Series of geometry objects into to shapely.geometry objects.

        Parameters
        ----------
        ds: pandas.Series

        Returns
        -------
        pandas.Series
        """
        assert isinstance(ds, pandas.Series)

        ds_geometries = ds.copy()
        assert isinstance(ds_geometries, pandas.Series)

        if all(isinstance(geometry, shapely.geometry.Polygon) for geometry in ds_geometries.values):
            return ds_geometries

        try:
            ds_geometries = ds_geometries.map(lambda x: shapely.wkt.loads(x))
            return ds_geometries
        except ValueError:
            raise ValueError('Failed to convert column objects into shapely.geometry objects')

    @staticmethod
    def _handle_multi(df):
        """
        Helper function for _clean_geometry(). Handles Multipolygons

        Parameters
        ----------
        df: pandas.DataFrame

        Returns
        -------
        """
        assert isinstance(df, pandas.DataFrame)
        geometry_col = 'geometry'

        df_row = df.copy()
        assert isinstance(df_row, pandas.DataFrame)

        row_list = []
        x = df_row[geometry_col].iloc[0]

        if isinstance(x, MultiPolygon):
            for y in x.geoms:
                df_row_temp = df_row
                assert isinstance(df_row_temp, pandas.DataFrame)

                df_row_temp[geometry_col] = y
                row_list.append(df_row_temp)

            df_multi_row = pandas.concat(row_list, ignore_index=True)
            assert isinstance(df_multi_row, pandas.DataFrame)
            df_to_return = df_multi_row
            assert isinstance(df_to_return, pandas.DataFrame)
            assert all(isinstance(geometry, shapely.geometry.Polygon)
                       for geometry in df_to_return[geometry_col].values)
            return df_to_return

        elif isinstance(x, shapely.geometry.polygon.Polygon):
            return df_row

        else:
            return pandas.DataFrame()

    @staticmethod
    def _get_x(row_elem):
        if isinstance(row_elem, shapely.geometry.polygon.Polygon):
            return row_elem.representative_point().y
        else:
            raise ValueError('Element in geometry column is %s. Expected Polygon'
                             % row_elem.__class__)

    @staticmethod
    def _get_y(row_elem):
        if isinstance(row_elem, shapely.geometry.polygon.Polygon):
            return row_elem.representative_point().x
        else:
            raise ValueError('Element in geometry column is %s. Expected Polygon'
                             % row_elem.__class__)

    @staticmethod
    def _get_df_centroid(df):
        """
        Returns a pandas.DataFrame with x, y, and yx columns

        Parameters
        ----------
        df: pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        assert isinstance(df, pandas.DataFrame)

        ds_centroid = df.centroid

        df_centroid = pandas.DataFrame({
            'x': ds_centroid.map(lambda pt: pt.x),
            'y': ds_centroid.map(lambda pt: pt.y)
        })

        df_centroid['yx'] = df_centroid.apply(lambda row: (row['y'], row['x']), axis=1)

        return df_centroid

    @staticmethod
    def _get_bounds(x_pts, y_pts):
        """
        Returns a leaflet-style list of [[miny, minx], [maxy, maxx]] bounds

        Parameters
        ----------
        x_pts: pandas.Series
        y_pts: pandas.Series

        Returns
        -------
        list
        """
        assert isinstance(x_pts, pandas.Series)
        assert isinstance(y_pts, pandas.Series)

        return [[float(y_pts.min()), float(x_pts.min())], [float(y_pts.max()), float(x_pts.max())]]

    @staticmethod
    def _find_centroid(x_pts, y_pts):
        """
        Finds the centroid given x and y coordinate arrays

        Parameters
        ----------
        x_pts
        y_pts

        Returns
        -------
        (x,y) coordinate corresponding to the center of the points
        """
        length = x_pts.shape[0]
        assert length == y_pts.shape[0], 'Lengths of the input x and y arrays are not equal'
        assert length > 0, 'Lengths of the input x and y arrays must be nonzero'

        sum_x = x_pts.sum()
        sum_y = y_pts.sum()
        return sum_x / length, sum_y / length

    @staticmethod
    def _add_dots(df, x_col, y_col, group_col, value_col, color, scale, opacity, f):
        """
        Adds the markers (coordinate, color, size) to the featureGroup for all the coordinates given.
        This function is used to .apply() it to a dataframe for the plotting of the scatterplot.

        Parameters
        ----------
        df: pandas.DataFrame
        x_col: str
        y_col: str
        group_col: str
        value_col: str
        color: str
        scale: float
        opacity: float
        f: folium.map.FeatureGroup

        Returns
        -------
        """
        coordinates = [(df[y_col]), (df[x_col])]
        fill_color = df['COLOR'] if group_col else color
        radius = 4000 * scale if value_col is None else df['VALUES_SCALED']

        f.add_child(
            folium.features.CircleMarker(
                coordinates,
                radius=radius,
                color=None,
                fill_color=fill_color,
                fill_opacity=opacity)
        )

        f.add_child(
            folium.features.CircleMarker(
                coordinates,
                radius=radius / 100,
                color=None,
                fill_color='black',
                fill_opacity=0.6)
        )

