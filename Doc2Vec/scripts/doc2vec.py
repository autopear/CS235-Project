# Use Doc2Vec algorithm to convert all reviews to vectors


import glob
import gzip
import numpy
import os
import shutil
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument


num_threads = 4  # Number of threads to be used
num_epochs = 10  # Number of iterations


def to_system_path(path):
    """ Convert an input path to the current system style, \ for Windows, / for others """
    if os.name == "nt":
        return path.replace("/", "\\")
    else:
        return path.replace("\\", "/")


def to_standard_path(path):
    """ Convert \ to / in path (mainly for Windows) """
    return path.replace("\\", "/")


dir_path = to_standard_path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # Module directory
src_dir = to_standard_path("{0}/fastText/data".format(os.path.dirname(dir_path)))
data_dir = "{0}/data".format(dir_path)
tmp_dir = "{0}/tmp".format(dir_path)

# Create temporary workspace
if not os.path.isdir(tmp_dir):
    os.makedirs(tmp_dir)

# Read number of folds dynamically
folds = []
for fp in glob.glob(to_system_path("{0}/fold-*.tsv.gzip".format(src_dir))):
    folds.append(int(os.path.basename(fp)[5:-9]))
folds.sort()

# Generate the input file needed

model_path = to_system_path("{0}/model".format(tmp_dir))
doc_path = to_system_path("{0}/docs.txt".format(tmp_dir))
map_path = to_system_path("{0}/map.txt".format(tmp_dir))

outf = open(doc_path, "w")  # Merged documents in one file
labels = []
num_docs = 0  # Number of reviews per fold

for fold in folds:
    with gzip.open(to_system_path("{0}/fold-{1}.tsv.gzip".format(src_dir, fold)), "rt") as inf:
        for line in inf:
            line = line[:-1]
            components = line.split("\t")
            if len(components) != 3:
                continue
            score = components[1]
            words = components[2]

            labels.append(score)
            outf.write("{0}\n".format(words))
    inf.close()
    if num_docs == 0:
        num_docs = len(labels)
outf.close()
print("Done preprocessing")

sentences = TaggedLineDocument(doc_path)  # Read all documents
model = Doc2Vec(size=100, window=8, min_count=0, workers=num_threads)  # Create model without initialization
model.build_vocab(sentences)  # Scan for words
model.train(sentences, total_examples=len(labels), epochs=num_epochs)  # Train
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)  # Save memory
print("Model created")

for fold in folds:
    mtx_path = to_system_path("{0}/vectors-{1}.npy".format(data_dir, fold))
    label_path = to_system_path("{0}/labels-{1}.txt".format(data_dir, fold))
    
    vecs = numpy.zeros((num_docs, 100), dtype=float) # Save document vectors into numpy 2D array
    start_idx = num_docs*(fold-1)
    for i in range(0, num_docs):
        vecs[i] = model.docvecs[start_idx+i]
    numpy.save(mtx_path, mtx) # Save vectors to binary file

    # Save labels
    labelf = open(label_path, "w")
    labelf.write("{0}\n".format("\n".join(labels[start_idx:start_idx+num_docs])))
    labelf.close()

    del vecs

    print("Done fold {0}".format(fold))

del model

# Remove the temporary workspace
shutil.rmtree(tmp_dir)

print("Done")
