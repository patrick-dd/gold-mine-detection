"""

Takes images from Bing maps, stores them in an S3 bucket

Based off Matt's script in github.com/ArnholdInstitute/ATLAS-data

"""

from multiprocessing import JoinableQueue
import psycopg2
import os
import math
import geopy
import geopy.distance
import json
import argparse
import requests
import boto3
from shapely.geometry import box, shape, mapping
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


def gsd(lat, zoom):
    """
    Computes the Ground Sample Distance (GSD).  More details can be found
    here: https://msdn.microsoft.com/en-us/library/bb259689.aspx

    Args:
        lat : float - latitude of the GSD of interest
        zoom : int - zoom level (WMTS)
    """
    return (math.cos(lat * math.pi / 180) * 2 * math.pi * 6378137
            ) / (256 * 2**zoom)


class TileServerClient:
    def __init__(self, queue, height, width, conn, country,
                 logo_height=0, max_downloads=float('inf')):
        self.queue = queue
        self.img_height = height
        self.img_width = width
        self.max_downloads = max_downloads
        self.cursor = conn.cursor()
        self.conn = conn
        self.country = country
        self.logo_height = logo_height

    def get_map(self, geom, datapoint, dir='images/pureearth'):
        minx, miny, maxx, maxy = geom.bounds
        # These are for the digital globe static maps API
        # url = 'https://api.mapbox.com/v4/digitalglobe.nal0g75k/%f,%f,18'\
        # + '/1280x1280.png?access_token=' + self.api_key

        # We step vertically by 640 pixels so that we can chop off the google
        # logo at the bottom
        # url = 'https://maps.googleapis.com/maps/api/staticmap?maptype='\
        #      + 'satellite&center=%s,%s&zoom=18&size=600x640&key=%%s'

        # BING maps
        url = 'https://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/%f,%f' \
              + '/18?mapSize=2000,1500&mapLayer=mapLayer&format=jpeg&key=%%s'

        put_count = 0

        i = 0
        x = minx
        y = miny
        self.count = 0
        while x < maxx:
            y = miny
            while y < maxy:
                if put_count > self.max_downloads:
                    print('Hit max downloads')
                    return
                self.count += 1
                next_x = geopy.distance.VincentyDistance(
                        meters=self.img_width * gsd(y, 18)).destination(
                                point=(y, x), bearing=90).longitude

                short_y = geopy.distance.VincentyDistance(
                        meters=(self.img_height - self.logo_height) *
                        gsd(y, 18)).destination(
                                point=(y, x), bearing=0).latitude
                long_y = geopy.distance.VincentyDistance(
                        meters=self.img_height * gsd(y, 18)).destination(
                                point=(y, x), bearing=0).latitude
                current_box = box(x, y, next_x, long_y)
                (lon,), (lat,) = current_box.centroid.xy
                if current_box.intersects(geom):
                    current_url = url % (lat, lon)
                    filename = os.path.join(dir, 'image_%d.jpg' % datapoint)
                    self.cursor.execute("""
                        SELECT filename FROM buildings.images
                        WHERE project=%s AND filename=%s
                    """, (self.country, filename))

                    if len(self.cursor.fetchall()) == 0:
                        put_count += 1
                        queue.put((current_url, filename,
                                   (x, y, next_x, long_y)))
                    i += 1
                y = short_y
            x = geopy.distance.VincentyDistance(
                    meters=self.img_width * gsd(y, 18)).destination(
                            point=(y, x), bearing=90).longitude


def download(queue, db_args, country, api_keys):
    conn = psycopg2.connect(**db_args)
    s3 = boto3.resource('s3')
    cur = conn.cursor()
    key_idx = len(api_keys) - 1
    while len(api_keys) > 0:
        url, filename, bounds = queue.get(block=True)
        print('Downloading %s' % filename)

        response = requests.get(url % api_keys[key_idx], stream=True)

        if response.status_code == 200:
            with open(filename, 'wb') as handle:
                for block in response.iter_content(1024):
                    handle.write(block)

            s3.meta.client.upload_file(filename, 'dg-images', filename)
            os.remove(filename)

            cur.execute("""
                INSERT INTO buildings.images (project, filename, shifted)
                VALUES (%s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))
                ON CONFLICT DO NOTHING
            """, (country, filename, json.dumps(mapping(box(*bounds)))))
            conn.commit()
            return conn
        elif response.status_code == 403:
            print('Received 403')
            api_keys.pop()
            key_idx -= 1
            return conn
        else:
            print(response.status_code)
            return conn


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--country', default='point_0123',
                        help='Country/project to download')
    parser.add_argument('--keys', default='keys.json',
                        help='Location of Bing API keys')
    parser.add_argument('--outdir', default='.',
                        help='Save directory')
    parser.add_argument('--osm', default=False,
                        help='Using OSM generated geojson')
    args = parser.parse_args()

    db_args = {
        'dbname': 'aigh',
        'host': os.environ.get('DB_HOST', 'localhost'),
        'user': os.environ.get('DB_USER', ''),
        'password': os.environ.get('DB_PASSWORD', '')
    }

    conn = psycopg2.connect(**db_args)
    cur = conn.cursor()
    cur.execute("""
        CREATE SCHEMA IF NOT EXISTS buildings;
        CREATE TABLE IF NOT EXISTS buildings.images(
            project text,
            filename text,
            geom geometry('POLYGON', 4326),
            UNIQUE(project, filename)
        )
    """)

    queue = JoinableQueue(maxsize=20)

    fc = json.load(open('../../data/geojson/%s.json' % args.country, 'r'))
    keys = json.load(open('%s' % args.keys, 'r'))['bing_static_maps']

    # creating folders for all the pure earth images
    if not os.path.exists('images/%s' % args.country):
        os.makedirs('images/%s' % args.country)

    # next test :: when we use a feature collection rather than a feature
    for feature in fc['features'][20:]:
        if args.osm:
            feature['id'] = feature['properties']['osm_id']
        print('Downloading for datapoint', feature['id'])
        client = TileServerClient(queue, 1500, 2000, conn, args.country,
                                  logo_height=25, max_downloads=40000)
        geom = shape(feature['geometry'])
        # using the feature id as a filename extension
        client.get_map(geom,
                       int(feature['id']),
                       dir='images/{}'.format(args.country))

        download(queue, db_args, args.country, keys)

# EOF
