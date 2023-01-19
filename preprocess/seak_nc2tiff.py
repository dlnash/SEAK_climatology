#!/usr/bin/python3

import argparse
from argparse import RawTextHelpFormatter
import os
import sys
from osgeo import gdal, ogr, osr
import netCDF4 as nc
import numpy as np
import configparser


  ### how to execute from command line
  ### python seak_nc2tiff.py <file.nc> <output_base>
  ### need to update local directory to geo_southeast.nc

def readWRF(ncFile, variable):

  ### Set up coordinate transformation
  ### Geographic to Lambert Conformal Conic
  inProj = osr.SpatialReference()
  inProj.ImportFromProj4('+proj=longlat +datum=WGS84 +no_defs')
  outProj = osr.SpatialReference()
  outProj.ImportFromProj4('+proj=lcc +lat_1=58 +lat_2=58 +lat_0=58 ' \
  '+lon_0=-138.5 +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs')
  transform = osr.CoordinateTransformation(inProj, outProj)

  dataset = nc.Dataset(ncFile, 'r')
  data = dataset.variables[variable][:,:]
  (rows, cols) = data.shape
  dataset.close()
  dataset = nc.Dataset('./geo_southeast.nc', 'r')
  lats = dataset.variables['XLAT_M'][0,:,:]
  print(np.shape(lats))
  lons = dataset.variables['XLONG_M'][0,:,:]
  dataset.close()
  x = np.zeros_like(data)
  print(np.shape(x))
  y = np.zeros_like(data)
  for ii in range(rows):
    for kk in range(cols):
      point = ogr.Geometry(ogr.wkbPoint)
      point.AddPoint(float(lons[ii,kk]), float(lats[ii,kk]))
      point.Transform(transform)
      x[ii,kk] = point.GetX()
      y[ii,kk] = point.GetY()
  originX = round(np.min(x), -1)
  originY = round(np.max(y), -1)
  posting = 4000
  geoTrans = [originX, posting, 0, originY, 0, -posting]

  return (np.flip(data, axis=0), geoTrans, outProj.ExportToWkt())


def data2geotiff(data, geoTrans, proj, outFile):

  (rows, cols) = data.shape
  gdalDriver = gdal.GetDriverByName('GTiff')
  outRaster = gdalDriver.Create(outFile, cols, rows, 1, gdal.GDT_Float32,
    ['COMPRESS=DEFLATE'])
  outRaster.SetGeoTransform(geoTrans)
  outRaster.SetProjection(proj)
  outBand = outRaster.GetRasterBand(1)
  outBand.SetNoDataValue(np.nan)
  outBand.WriteArray(data)
  outRaster = None


def wrf2geotiff(ncFile, outBase):

  ### Accumulated total grid scale precipitation
  outFile = outBase+'_PCPT.tif' ### Check variable name
  print('Converting accumulated total grid scale precipitation to GeoTIFF (%s)'\
    % outFile)
  (data, geoTrans, proj) = readWRF(ncFile, 'PCPT') ### Check variable name
  data2geotiff(data, geoTrans, proj, outFile)


if __name__ == '__main__':

  parser = argparse.ArgumentParser(prog='wrf2geotiff',
    description='converts WRF variables to GeoTIFFs',
    formatter_class=RawTextHelpFormatter)
  parser.add_argument('wrf', metavar='<WRF file>',
    help='name of the WRF netCDF file')
  parser.add_argument('outBase', metavar='<output basename>',
    help='basename of the output GeoTIFFs')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  args = parser.parse_args()

  if not os.path.exists(args.wrf):
    print('WRF netCDF file (%s) does not exist!' % args.wrf)
    sys.exit(1)

  wrf2geotiff(args.wrf, args.outBase)
