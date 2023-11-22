*This README.md file was generated on 20231122 by Deanna Nash*

General Information
--------------------

### `SEAK_combined_*.csv`:

*Information on data for "Atmospheric Rivers in Southeast Alaska: Meteorological Conditions Associated with Extreme Precipitation"*

### Author Information:

- Principal Investigator: Jon Rutz (jrutz@ucsd.edu)
- Co-investigator: Deanna Nash (dnash@ucsd.edu)

### Date of data collection/creation:
20220930 to 20221122

### Geographic location of data collection:
Southeast Alaska (54&deg;N to 60&deg;N, 130&deg;W to 141&deg;W)

### Funders and sponsors of data collection:
This research was supported by the National Science Foundation (NSF) Coastlines and People Program (award 2052972). Gunalchéesh to the Tlingit people for their stewardship of Lingít Aaní since time immemorial and today. This work used the COMET supercomputer, which was made available by the Atmospheric River Program Phase 2 and 3 supported by the California Department of Water Resources (awards 4600013361 and 4600014294 respectively) and the Forecast Informed Reservoir Operations Program supported by the U.S. Army Corps of Engineers Engineer Research and Development Center (award USACE W912HZ-15-2-0019).


Sharing/Access Information
--------------------------

### License & restrictions on data reuse:
Creative Commons Attribution 4.0 International (CC BY 4.0)

### Recommended citation for the data:
<!-- suggested: Creator (PublicationYear). Title. Publisher. Identifier -->

### Related publications:
> Nash, Deanna, Rutz, J.J., and Jacobs, A. (2023). “Atmospheric Rivers in Southeast Alaska: Meteorological Conditions Associated with Extreme Precipitation”. In: <em>JGR Atmospheres</em> (in review) [preprint available here](https://essopenarchive.org/users/634498/articles/652354-atmospheric-rivers-in-southeast-alaska-meteorological-conditions-associated-with-extreme-precipitation?commit=2c9b271885b99d17baae69cc25279bd3868e6d4a)

### Links to other publicly accessible locations of the data:
[https://github.com/dlnash/SEAK_climatology](https://github.com/dlnash/SEAK_climatology)

### Related data sets: 
- The AR data were provided by Bin Guan via [https://dataverse.ucla.edu/dataverse/ar](https://dataverse.ucla.edu/dataverse/ar). Development of the AR detection algorithm and databases was supported by NASA. 
- ERA5 data on single levels [https://doi.org/10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47) and pressure levels [https://doi.org/10.24381/cds.bd0915c6](https://doi.org/10.24381/cds.bd0915c6) were downloaded from the Copernicus Climate Change Service (C3S) Climate Data Store. The results contain modified Copernicus Climate Change Service information 2020. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains. 
- The High Resolution Downscaled Climate Data for Southeast Alaska was accessed on 17 November, 2022 from [https://registry.opendata.aws/wrf-se-alaska-snap](https://registry.opendata.aws/wrf-se-alaska-snap). 

Data & File Overview
--------------------

### File list:

The files (listed below) contain daily meteorological information between January 1, 1980 and December 31, 2019 for the grid cell closest to each of the six communities for this analysis. 
```
.
├── data
│   ├── SEAK_combined_Kasaan.csv
│   ├── SEAK_combined_Craig.csv
│   ├── SEAK_combined_Yakutat.csv
│   ├── SEAK_combined_Klukwan.csv
│   ├── SEAK_combined_Skagway.csv
└── └── SEAK_combined_Hoonah.csv
```

File/Format-Specific Information
--------------------------------

### `SEAK_climatolog/data/SEAK_combined_*.csv`
- Number of variables: 13
- Number of cases/rows: 14610
- Variable name, defining any abbreviations, units of measure, other notes:
    - time, N/A, yyyy-mm-dd, N/A
    - IVT, Integrated water Vapor Transport, kg m-1 s-1, daily maximum IVT magnitude based on hourly ERA5
    - lat, latitude, degrees, latitude of the grid cell closest to the community
    - lon, longitude, degrees, longitude of the grid cell closest to the community
    - uIVT, IVT in the zonal direction, kg m-1 s-1, uIVT at the time of daily maximum IVT based on ERA5
    - vIVT, IVT in the meridional direction, kg m-1 s-1, vIVT at the time of daily maximum IVT based on ERA5
    - IWV, Integrated Water Vapor, mm, daily maximum IWV based on ERA5
    - ivtdir, IVT direction, degrees, IVT direction at the time of the daily maximum IVT based on ERA5
    - AR, Amospheric River, binary, 1 - AR detected by tARgetv3 in Southeast Alaska that day, 0 - no AR detected
    - impact, notable impact identified by National Weather Service Juneau, binary, 1 - yes impact, 0 - no impact, **this requires more analysis**
    - prec, precipitation, mm day-1, daily sum of precipitation based on SEAK-WRF (Lader et al., 2020)
    - UVdir, wind direction, degrees, 1000 hPa wind direction (based on SEAK-WRF) at the time of daily maximum IVT (based on ERA5)
    - extreme, extreme precipitation, binary, 1 - extreme precipitation, 0 - not extreme precipitation, "extreme" precipitation is defined as precipitation exceeding the 95th percentile threshold for that community based on all precipitation values greater than 2.5 mm (0.1 inch)


Methodological Information
--------------------------

See [`../downloads/README.md`](https://github.com/dlnash/SEAK_climatology/tree/main/downloads) for more information on the data used and [`../preprocess/README.md`](https://github.com/dlnash/SEAK_climatology/tree/main/preprocess) for information on the preprocessing steps.