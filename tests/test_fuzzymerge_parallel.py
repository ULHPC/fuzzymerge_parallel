#!/usr/bin/env python

"""Tests for `fuzzymerge_parallel` package."""

import pytest
from click.testing import CliRunner
from nltk.corpus import movie_reviews
import nltk
import pandas as pd
from fuzzymerge_parallel.FuzzyMergeParallel import FuzzyMergeParallel
from fuzzymerge_parallel import cli



@pytest.fixture(scope="module")
def dataset(rows=5000):
    """Load nltk dataset

    Args:
        rows (int, optional): Row numbers. Defaults to 5000.

    Yields:
        pd.Dataframe: two dataframes.
    """
    try:
        # Check if the movie_reviews corpus is already downloaded
        nltk.data.find('corpora/movie_reviews.zip')
    except LookupError:
        # Download the movie_reviews corpus
        nltk.download('movie_reviews')
    # Get the list of words from the movie_reviews corpus
    words = movie_reviews.words()

    # Split the list of words into two halves
    half_size = len(words) // 2
    first_half = words[:half_size]
    second_half = words[half_size:]
    # Create DataFrames from the word lists
    df1 = pd.DataFrame({'words_left': first_half})
    df2 = pd.DataFrame({'words_right': second_half})
    df1 = df1[0:rows]
    df2 = df2[0:rows]
    yield df1, df2


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'fuzzymerge_parallel.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def test_sequential(dataset):
    """Test sequential execution of FuzzyMergeParallel.
    """
    left_df, right_df = dataset
    fm_seq = FuzzyMergeParallel(
        left_df, right_df, left_on='words_left', right_on='words_right')
    # Set parameters
    fm_seq.set_parameter('how', 'inner')
    fm_seq.set_parameter('parallel', False)
    # Run the merge sequentially
    result = fm_seq.merge()
    assert result.shape == (600, 2)


def test_multiprocessing(dataset):
    """Test multiprocessing execution of FuzzyMergeParallel.
    """
    left_df, right_df = dataset
    fm_multi = FuzzyMergeParallel(
        left_df, right_df, left_on='words_left', right_on='words_right')
    # Set parameters
    fm_multi.set_parameter('how', 'inner')
    # Set parameters for multiprocessing
    fm_multi.set_parameter('parallel', True)
    fm_multi.set_parameter('n_threads', 0)  # All available cores
    # Run the merge multiprocessing
    result = fm_multi.merge()
    assert result.shape == (600, 2)


def get_local_client():
    """Create a dask client with local cluster.
    """
    from dask.distributed import Client, LocalCluster
    # Create a local Dask cluster
    cluster = LocalCluster()
    # Create a Dask client to connect to the cluster
    client = Client(cluster)
    # Get the total number of cores available in the cluster
    total_cores = sum(client.ncores().values())
    print("total cores ", total_cores)
    return client


def test_dask(dataset):
    """Test dask execution of FuzzyMergeParallel.
    """
    left_df, right_df = dataset
    client = get_local_client()
    fm_dask = FuzzyMergeParallel(
        left_df, right_df, left_on='words_left', right_on='words_right')
    # Set parameters
    fm_dask.set_parameter('how', 'inner')
    # Set parameters for multiprocessing
    fm_dask.set_parameter('parallel', True)
    fm_dask.set_parameter('dask_client', client)
    # Run the merge in dask
    result = fm_dask.merge()
    client.close()
    assert result.shape == (600, 2)
