# Preprocessing

This directory contains extracted and stemmed reviews and codes for them.

## [stemmer](stemmer/)
This is a Qt project of command line stemmer. Algorithm and some codes were adopted from [Porter2](http://snowball.tartarus.org/algorithms/english/stemmer.html). Build with any Qt version (>= 4), most operating systems are supported.

### [json_extract.py](json_extract.py)
Download, extract and parse gzipped json file of Amazon reviews.

### [output](output/)
* [extracted.log](output/extracted.log): Log file for extraction
* extracted.tsv: Extracted raw reviews
* stemmed.tsv: Stemmed reviews

## License
[LGPL](LICENSE)
