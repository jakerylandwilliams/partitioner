## Synopsis

This is the Python partitioner project. The partitioner module performs advanced NLP tasks essentially equivalent to tokenization (e.g., splitting texts into words), with generalizations into multiword expressions (MWE) segmentation. A definition for those unfamiliar with MWEs: 

“A group of tokens in a sentence that cohere more strongly than ordinary syntactic combinations.”

Thus, partitioner may be used to split texts "phrases" of one or more words.

## Code Example

To load the module, run:

\>\>\> from partitioner.tools import partitioner

Since the module comes with no data, running informed partitions will require acquiring the training data, which may be acquired by engaging the `.download()` method:

\>\>\> pa = partitioner()

\>\>\> pa.download()

Note that the above will require responding to a prompt.

Additionally, since high-perfornace versions of the partitioner utilize the nltk package's `PerceptronTagger()` function, consider running:

\>\>\> import nltk

\>\>\> nltk.download()

and download all nltk data.

Once the training data has been downloaded, the following will load all English data sets. This requires significant memory resources, but results in a high-performance model:

\>\>\> pa = partitioner(language = "en", doPOS = True, doLFD = True, maxgap = 8, q = {"type": 0.74, "POS": 0.71})

\>\>\> pa.partition("How could something like this simply pop up out of the blue?")

['How', ' ', 'could', ' ', 'something', ' ', 'like', ' ', 'this', ' ', 'simply', ' ', 'pop up', ' ', 'out of the blue', '?']

The memory overhead comes from an English Wikipedia data set. While bulky, this data set provides a huge number of named entities. To load from a specific English source, use:

\>\>\> pa = partitioner(language="en", source="wiktionary")

or one of the other data sets. To view all of the available datasets, check out:

\>\>\> pa.datasets

To load all sets from a specific language (assuming data has been added beyond the starter data, which comes from Wikipedia), use:

\>\>\> pa = partitioner(language="es", source="")

## Motivation

The original goal of the partitioner project was to create a fast, efficient, and general algorithm that segments texts into the smallest-possible meaningful units, which we refer to as phrases. This essentially coincides with the NLP task for comprehensive MWE segmentation segmentation. Reference for this modules function may be found in the following article:

https://arxiv.org/pdf/1608.02025.pdf

## Installation

Using pip from the command line:

\>\>\> pip install partitioner

Alternatively, if using git from a command line first clone the repository:

\>\>\> git clone https://github.com/jakerylandwilliams/partitioner.git

then navigate the repository's main directory and run:

\>\>\> sudo python setup.py install

## Contributors

Jake Ryland Williams and Andy Reagan

## License

Apache