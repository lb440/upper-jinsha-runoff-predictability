# Data access and file structure

The observed daily runoff records used in the study were obtained from the *Hydrological Yearbook of the People's Republic of China*. They are subject to third-party data-use restrictions and are not redistributed in this repository.

Users who lawfully obtain the runoff records can create a local CSV matching `data_dictionary.csv` and set `RUNOFF_DATA_PATH` to it. ERA5-Land forcing data are available from the Copernicus Climate Data Store at https://doi.org/10.24381/cds.e2161bac.

`example_model_input_synthetic.csv` is entirely synthetic. It only demonstrates the required column names and date format, and cannot reproduce reported results.
