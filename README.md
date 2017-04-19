## Synopsis

This is the Python partitioner project. The partitioner module performs advanced NLP tasks essentially equivalent to tokenization (e.g., splitting texts into words), with generalizations into multiword expressions (MWE) segmentation. A definition for those unfamiliar with MWEs: 

“A group of tokens in a sentence that cohere more strongly than ordinary syntactic combinations.”

Thus, partitioner may be used to split texts "phrases" of one or more words.

## Code Example

Usage for the base setup is quite simple. The following will utilize all of the English data sets, requiring significant memory to load the training data:

>>> from partitioner.tools import partitioner
>>> pa = partitioner()
>>> pa.partition("How could something like this simply pop up out of the blue?")
['How', ' ', 'could', ' ', 'something', ' ', 'like', ' ', 'this', ' ', 'simply', ' ', 'pop up', ' ', 'out of the blue', '?']

The large data set with the memory overhead comes from English Wikipedia. While bulky, this data set provides a huge number of named entities. To load from a specific source, use:

>>> pa = partitioner(language="en", source="wiktionary")

or one of the other data sets. To load all sets from a specific language (assuming data has been added beyond the starter data, which comes from Wikipedia), use:

>>> pa = partitioner(language="es", source="")

## Motivation

The original goal of the partitioner project was to create a fast, efficient, and general algorithm that segments texts into the smallest-possible meaningful units, which we refer to as phrases. This essentially coincides with the NLP task for comprehensive MWE segmentation segmentation. Reference for this modules function may be found in the following article:

https://arxiv.org/pdf/1608.02025.pdf

## Installation

Using pip from the command line:

>>> pip install partitioner

Alternatively, if using git from a command line first clone the repository:

>>> git clone https://github.com/jakerylandwilliams/partitioner.git

then navigate the repository's main directory and run:

>>> sudo python setup.py install

## Contributors

Jake Ryland Williams
Andy Reagan

## License

Apache