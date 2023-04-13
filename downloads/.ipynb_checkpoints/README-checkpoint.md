1. Download ERA5 data using cds-api and ERA5_config.yml

```
# in download directory
conda activate cds
bash download_ERA5.sh
```

2. Download southeast Alaska WRF data from Lader et al., 2020

You will need awscli installed on your machine.

```
# in download directory
bash download_SEAK-WRF.sh
```

3. Download the AR detection result

Navigate to [https://dataverse.ucla.edu/dataverse/ar](https://dataverse.ucla.edu/dataverse/ar) and select 'globalARcatalog_ERA-Interim_1979-2019_v3.0.nc' to download. You will need to provide information regarding the use of this dataset. 

4. Download the Global Multi-resolution Terrain Elevation Data 2010 7.5 arc-second data

Navigate and login to Earth Explorer [https://earthexplorer.usgs.gov/](https://earthexplorer.usgs.gov/). Select the "Data Sets" tab and type in "GMTED2010" in the "Data Set Search" box. From the drop-down box, select GMTED2010. Next, select the "Additional Criteria" tab and click the + icon by Entity ID. Type "GMTED2010N50W150" into the box. Select the Results >> button and wait for the results to load. The correct tile should show up in the results and can be confirmed by clicking the footprint icon. To download, select the icon with the green down arrow and then select the Download button under the "7.5 ARC SEC (711.85 MiB)" download option.

5. Download the Anadromous Waters Catalog for Southeast Alaska (Rivers and Streams)

Navigate to the (Alaska Department of Fish and Game)[https://www.adfg.alaska.gov/sf/SARR/AWC/index.cfm?ADFG=maps.dataFiles] 2022 Regulatory AWC GIS data page. Select "Shapefile" for AWC for the Southwestern Region. 

6. Download the Alaska Randolph Glacier Index v6.0

Navigate to the (State of Alaska Geoportal)[https://gis.data.alaska.gov/datasets/81ab2195b29f4ef3ab4cafe565a7c2e0_0/explore?location=59.824584%2C-151.498483%2C5.51] and click "Download" to retrieve the Alaska Randolf Glacier Inventory v6.0 shapefile.

