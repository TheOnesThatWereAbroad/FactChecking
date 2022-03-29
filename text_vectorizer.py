import numpy as np
import pandas as pd
import os
import glob
from urllib import request
import zipfile
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

class NotAdaptedError(Exception):
    pass


class TextVectorizer:
    def __init__(
        self,
        glove_url="http://nlp.stanford.edu/data/glove.6B.zip",
        embedding_dim=100,
        embedding_folder="glove",
        max_sentence_length=300,
    ):
        """
        This class parses the GloVe embeddings, the input documents are expected
        to be in the form of a list of lists.
        [["word1", "word2", ...], ["word1", "word2", ...], ...]

        Parameters
        ----------
        glove_url : The url of the GloVe embeddings.
        embedding_dim : The dimension of the embeddings (pick one of 50, 100, 200, 300).
        embedding_folder : folder where the embedding will be downloaded
        max_sentence_length : The maximum size of the documents.
        """
        self.embedding_dim = embedding_dim
        self.download_glove_if_needed(
            glove_url=glove_url, embedding_folder=embedding_folder
        )
        self.max_sentence_length = max_sentence_length

        # create the vocabulary
        self.vocabulary = self.parse_glove(embedding_folder)

    def download_glove_if_needed(self, glove_url, embedding_folder):
        """
        Downloads the glove embeddings from the internet

        Parameters
        ----------
        glove_url : The url of the GloVe embeddings.
        embedding_folder: folder where the embedding will be downloaded
        """
        # create embedding folder if it does not exist
        if not os.path.exists(embedding_folder):
            os.makedirs(embedding_folder)

        # extract the embedding if it is not extracted
        if not glob.glob(
            os.path.join(embedding_folder, "**/glove*.txt"), recursive=True
        ):

            # download the embedding if it does not exist
            embedding_zip = os.path.join(embedding_folder, glove_url.split("/")[-1])
            if not os.path.exists(embedding_zip):
                print("Downloading the GloVe embeddings...")
                request.urlretrieve(glove_url, embedding_zip)
                print("Successful download!")

            # extract the embedding
            print("Extracting the embeddings...")
            with zipfile.ZipFile(embedding_zip, "r") as zip_ref:
                zip_ref.extractall(embedding_folder)
                print("Successfully extracted the embeddings!")
            os.remove(embedding_zip)

    def parse_glove(self, embedding_folder):
        """
        Parses the GloVe embeddings from their files, filling the vocabulary.

        Parameters
        ----------
        embedding_folder : folder where the embedding files are stored

        Returns
        -------
        dictionary representing the vocabulary from the embeddings
        """
        vocabulary = {"<pad>": np.zeros(self.embedding_dim)}
        embedding_file = os.path.join(
            embedding_folder, "glove.6B." + str(self.embedding_dim) + "d.txt"
        )
        with open(embedding_file, encoding="utf8") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                vocabulary[word] = coefs
        return vocabulary

    def adapt(self, dataset, columns=None):
        """
        Computes the OOV words for a single data split, and adds them to the vocabulary.

        Parameters
        ----------
        documents : The data split (might be training set, validation set, or test set).
        """
        # create a set containing words from the documents in a given data split
        words = {word for column in columns for sentence in dataset[column] for word in sentence}
        oov_words = words - self.vocabulary.keys()
        # add the OOV words to the vocabulary giving them a random encoding
        for word in oov_words:
            self.vocabulary[word] = np.random.uniform(-1, 1, size=self.embedding_dim)
        self.encoding = {word: i for i, word in enumerate(self.vocabulary.keys())}
        self.embedding_matrix = np.array(list(self.vocabulary.values()))
        print(f"Generated embeddings for {len(oov_words)} OOV words.")
        return self.embedding_matrix

    def transform(self, dataset, columns=None, to_embedding=False):
        """
        Transform the data into the input structure for the training. This method should be used always after the adapt method.

        Parameters
        ----------
        documents : The data split (might be training set, validation set, or test set).

        Returns
        -------
        Array of shape (number of documents, number of words, embedding dimension)
        """
        X_claim, X_evidence = [], []
        for index, row in tqdm(dataset.iterrows()):
            if to_embedding:
                X_claim.append(self._transform_document_to_embedding(row["Claim"]))
                X_evidence.append(self._transform_document_to_embedding(row["Evidence"]))
            else:
                X_claim.append(self._transform_document_to_encoding(row["Claim"]))
                X_evidence.append(self._transform_document_to_encoding(row["Evidence"]))
        return np.stack(X_claim), np.stack(X_evidence)


    def _transform_document_to_encoding(self, document):
        """
        Transforms a single document to the matrix encoding

        Parameters
        ----------
        document : The document to be transformed.

        Returns
        -------
        Numpy array of shape (number of words)
        """
        return np.hstack((np.array([self.encoding[word] for word in document]), np.zeros(self.max_sentence_length - len(document))))

    def _transform_document_to_embedding(self, document):
        """
        Transforms a single document to the GloVe embedding

        Parameters
        ----------
        document : The document to be transformed.

        Returns
        -------
        Numpy array of shape (number of words, embedding dimension)
        """
        try:
            if not document or len(document) > self.max_sentence_length:
                return None
            # return np.array([self.vocabulary[word] for word in document])
            return np.vstack((np.array([self.vocabulary[word] for word in document]),np.zeros((self.max_sentence_length - len(document), self.embedding_dim))))
        except KeyError:
            raise NotAdaptedError(
                f"The whole document is not in the vocabulary. Please adapt the vocabulary first."
            )
        except TypeError: # NaNs
            return None


def encode_target(target_series):
    """
    Encodes the target column of the dataset
    """
    return target_series.apply(lambda x: 1 if x == "SUPPORTS" else 0)

if __name__ == "__main__":
    data = pd.read_csv('dataset/train_pairs.csv', index_col=0)
    tv = TextVectorizer()
    tv.adapt(data, ["Claim", "Evidence"])
    data = tv.transform(data, ["Claim", "Evidence"])
    data.dropna(inplace=True)
    X_train_claim = np.stack(data["Claim"])
    X_train_evidence = np.stack(data["Evidence"])
    y_train = data["Label"].to_numpy()
    np.savetxt("X_train_claim.txt", X_train_claim)
    np.savetxt("X_train_evidence.txt", X_train_evidence)
    np.savetxt("y_train.txt", y_train)
    
