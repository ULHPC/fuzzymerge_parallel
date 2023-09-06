# fuzzymerge_parallel

Merge two pandas dataframes by using a function to calculate the edit distance (Levenshtein Distance) using multiprocessing or Dask to do it in parallel.




## Features

- Performs fuzzy merging of dataframes based on string columns
- Utilize distance functions (e.g., Levenshtein) for intelligent matching
- Benefit from parallel computing techniques for enhanced performance
- Easily integrate into your existing data processing pipelines

## Installation

To download and install the FuzzyMergeParallel Python package from GitHub, you can follow these improved instructions:

To install FuzzyMergeParallel via pip from its GitHub repository, follow these steps:

1. Download the Package: Begin by downloading the package from GitHub. You can use git to clone the repository to your local machine:
    ```bash
    git clone https://github.com/ULHPC/fuzzymerge_parallel.git
    ```
    
2. Navigate to the Package Directory: Open a terminal or command prompt and change your current directory to the downloaded package folder:
    ```bash
    cd FuzzyMergeParallel
    ```

3. Install the Package: Finally, use pip to install the package in "editable" mode (with the -e flag) to allow for development and updates:
    ```bash
    pip install -e .
    ```

This command will install the package along with its dependencies. You can now import and use FuzzyMergeParallel in your Python projects.    

## Dependencies

To use this package, you will need to have the following dependencies installed:

- [Click](https://pypi.org/project/Click/) >= 7.0
- [dask[distributed]](https://pypi.org/project/dask/) >= 2023.5.0
- [Levenshtein](https://pypi.org/project/python-Levenshtein/) >= 0.21.0
- [nltk](https://pypi.org/project/nltk/) >= 3.8.1
- [numpy](https://pypi.org/project/numpy/) >= 1.23.5
- [pandas](https://pypi.org/project/pandas/) >= 1.5.3
- [tqdm](https://pypi.org/project/tqdm/) >= 4.65.0
- [psutil](https://pypi.org/project/psutil/) == 5.9.5
- [pytest](https://pypi.org/project/pytest/) >= 7.4.1

## Description

The FuzzyMergeParallel class is exposed and it is highly configurable. The following parameters and other attributes can be set up before doing the merge operation:



| Parameter        | Description                                                      |
|------------------|------------------------------------------------------------------|
| left             | The left input data to be merged.                                |
| right            | The right input data to be merged.                               |
| left_on          | Column(s) in the left DataFrame to use as merge keys.            |
| right_on         | Column(s) in the right DataFrame to use as merge keys.           |

Example create a FuzzyMergeParallel class:

```python
fuzzy_merger = FuzzyMergeParallel(left_df, right_df, left_on='left_column_name', right_on='right_column_name')
```


| Attribute        | Description                                                      |
|------------------|------------------------------------------------------------------|
| uselower         | Whether to convert strings to lowercase before comparison. Default is True.        |
| threshold        | The threshold value for fuzzy matching similarity. Default is 0.9.                |
| how              | The type of merge to be performed. Default is 'outer'.                               |
| on               | Column(s) to merge on if not specified in left_on or right_on. Default is False.    |
| left_index       | Whether to use the left DataFrame's index as merge key(s). Default is False.       |
| right_index      | Whether to use the right DataFrame's index as merge key(s). Default is False.      |
| parallel         | Whether to perform the merge operation in parallel. Default is False.              |
| n_threads        | The number of threads to use for parallel execution. Default is 0 (all available threads)             |
| hide_progress    | Whether to display a progress bar during the merge operation. Default is False.    |
| num_batches      | The number of batches to split the ratio computation. Default is automatic.              |
| ratio_function   | The distance ratio function.                Defaults to `Levenshtein.ratio()`.                      |
| dask_client      | A dask client object.                                            |

Example set extra attributes by stating the name of the attribute and its value with `set_parameter()`:

```python
fuzzy_merger.set_parameter('how', 'inner')
fuzzy_merger.set_parameter('threshold', 0.75)
```

## Usage

```python
fuzzy_merger = FuzzyMergeParallel(left_df, right_df, left_on='left_column_name', right_on='right_column_name')
# Set parameters
fuzzy_merger.set_parameter('how', 'inner')
# Run the merge sequentially
result = fuzzy_merger.merge()

# Set parameters for multiprocessing
fuzzy_merger.set_parameter('parallel', True)
fuzzy_merger.set_parameter('n_threads', 64)
# Run the merge multiprocessing
result = fuzzy_merger.merge()

# Set parameters for dask
## Create a dask client
from dask.distributed import Client
client = Client(...)  # Connect to distributed cluster and override default
fuzzy_merger.set_parameter('parallel', True)
fuzzy_merger.set_parameter('dask_client', client)
# Run the merge in dask
result = fuzzy_merger.merge()
```





## Contributing

Contributions are welcome! If you encounter any issues, have suggestions, or want to contribute improvements, please submit a pull request or open an issue on the GitHub repository.


## Authors

- Oscar J. Castro Lopez (oscar.castro@uni.lu)
  - Parallel Computing & Optimisation Group (PCOG) - **University of Luxembourg**

## License

This project is licensed under the MIT License.