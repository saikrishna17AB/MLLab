import ast
import numpy as np
from scipy.sparse.linalg import svds
import pandas as pd


DATASET_FILE = "dataset.txt"
K = 2
TOP_N = 3

def load_dataset(filename):
    """
    Load the Python-style dataset supplied in dataset.txt.

    Only the value assigned to `dataset` is extracted.
    """

    with open(filename, "r", encoding="utf-8") as file:
        source = file.read()

    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "dataset":
                    return ast.literal_eval(node.value)

    raise ValueError(
        "Could not find a variable named 'dataset' in dataset.txt."
    )


def makeDocumentTermMatrix(data):
    """
    Construct a Words x Documents matrix.

    Rows    -> unique words
    Columns -> documents
    Values  -> exact count of a word in a document
    """

    vocabulary = []

    for document in data:
        for word in document["words"]:
            if word not in vocabulary:
                vocabulary.append(word)

    word_to_index = {
        word: index
        for index, word in enumerate(vocabulary)
    }

    number_of_words = len(vocabulary)
    number_of_documents = len(data)

    matrix = np.zeros(
        (number_of_words, number_of_documents),
        dtype=float
    )

    for document_index, document in enumerate(data):

        for word in document["words"]:

            word_index = word_to_index[word]

            matrix[
                word_index,
                document_index
            ] += 1

    return matrix, vocabulary


def normalize_matrix(matrix):
    """
    Apply log-entropy weighting.

    Local weight:
        L_ij = log(1 + A_ij)

    Row probability:
        p_ij = A_ij / sum_j(A_ij)

    Global entropy:
        E_i = -sum_j p_ij * log(p_ij) / log(N)

    Global weight:
        G_i = 1 - E_i

    Final weight:
        W_ij = L_ij * G_i

    For a single-document corpus, entropy is defined as 0,
    giving a global weight of 1 because log(N) = 0.
    """

    matrix = np.asarray(
        matrix,
        dtype=float
    )

    number_of_documents = matrix.shape[1]


    local_factor = np.log1p(matrix)


    global_factor = np.ones(
        matrix.shape[0],
        dtype=float
    )

    if number_of_documents > 1:

        row_sums = matrix.sum(
            axis=1,
            keepdims=True
        )

        probabilities = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix),
            where=row_sums != 0
        )

        entropy_terms = np.zeros_like(
            probabilities
        )

        nonzero = probabilities > 0

        entropy_terms[nonzero] = (
            probabilities[nonzero]
            * np.log(probabilities[nonzero])
        )

        entropy = -(
            entropy_terms.sum(axis=1)
            / np.log(number_of_documents)
        )

        global_factor = 1 - entropy


    normalized_matrix = (
        local_factor
        * global_factor[:, np.newaxis]
    )

    return (
        normalized_matrix,
        local_factor,
        global_factor
    )


def perform_svd(normalized_matrix, k=2):
    """
    Perform:

        A ≈ U Sigma V^T

    using truncated SVD.
    """

    rows, columns = normalized_matrix.shape

    if min(rows, columns) <= k:

        print(
            "\nWARNING:"
            f" The current dataset has matrix shape "
            f"{normalized_matrix.shape}."
        )

        print(
            f"A true truncated SVD with k={k} requires "
            f"k < min(words, documents)."
        )

        print(
            "Using full NumPy SVD so the pipeline can still "
            "run on the supplied corpus."
        )

        U_full, singular_values, Vt_full = np.linalg.svd(
            normalized_matrix,
            full_matrices=False
        )

        actual_k = min(
            k,
            len(singular_values)
        )

        U = U_full[:, :actual_k]
        singular_values = singular_values[:actual_k]
        Vt = Vt_full[:actual_k, :]

    else:

        U, singular_values, Vt = svds(
            normalized_matrix,
            k=k
        )

        order = np.argsort(
            singular_values
        )[::-1]

        singular_values = singular_values[
            order
        ]

        U = U[
            :,
            order
        ]

        Vt = Vt[
            order,
            :
        ]

        actual_k = k

    Sigma = np.diag(
        singular_values
    )

    return (
        U,
        Sigma,
        Vt,
        actual_k
    )


def project_words(normalized_matrix, Vt):
    """
    Project words into the low-dimensional concept space.

    The problem statement specifies:

        Projected Words = A * V

    Since Vt = V^T:

        V = Vt^T
    """

    V = Vt.T

    projected_words = (
        normalized_matrix @ V
    )

    return projected_words


def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.
    """

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b)
        / (norm_a * norm_b)
    )


def find_closest_words(
    target_word,
    vocabulary,
    projected_words,
    top_n=3
):
    """
    Find the top N words closest to target_word
    using cosine similarity.
    """

    if target_word not in vocabulary:
        raise ValueError(
            f"Word '{target_word}' was not found "
            "in the vocabulary."
        )

    target_index = vocabulary.index(
        target_word
    )

    target_vector = projected_words[
        target_index
    ]

    similarities = []

    for index, word in enumerate(vocabulary):

        if word == target_word:
            continue

        similarity = cosine_similarity(
            target_vector,
            projected_words[index]
        )

        similarities.append(
            (word, similarity)
        )

    similarities.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return similarities[:top_n]



def main():

    print("=" * 70)
    print("LATENT SEMANTIC ANALYSIS (LSA)")
    print("=" * 70)


    data = load_dataset(
        DATASET_FILE
    )

    print("\nNumber of documents:", len(data))

    print("\nDocuments:")

    for document in data:
        print(
            " -",
            document["filename"]
        )


    print("\n" + "=" * 70)
    print("STEP 1 - DOCUMENT-TERM MATRIX")
    print("=" * 70)

    matrix, vocabulary = (
        makeDocumentTermMatrix(data)
    )

    print(
        "\nVocabulary size:",
        len(vocabulary)
    )

    print(
        "Number of documents:",
        len(data)
    )

    print(
        "Matrix shape:",
        matrix.shape
    )

    print("\nVocabulary:")
    print(vocabulary)

    print("\nDocument-Term Matrix:")

    matrix_df = np.asarray(
        matrix,
        dtype=int
    )

    print(matrix_df)


    (
        normalized_matrix,
        local_factor,
        global_factor
    ) = normalize_matrix(
        matrix
    )

    print("\nLocal Factor:")
    print(local_factor)

    print("\nGlobal Factor:")
    print(global_factor)

    print("\nLog-Entropy Normalized Matrix:")
    print(
        np.round(
            normalized_matrix,
            4
        )
    )

    print("\n" + "=" * 70)
    print("STEP 2 - TRUNCATED SVD")
    print("=" * 70)

    (
        U,
        Sigma,
        Vt,
        actual_k
    ) = perform_svd(
        normalized_matrix,
        K
    )

    print("\nRequested k:", K)
    print("Actual number of components:", actual_k)

    print("\nLeft Singular Vectors (U):")
    print(
        np.round(
            U,
            4
        )
    )

    print("\nSingular Values:")
    print(
        np.round(
            np.diag(Sigma),
            4
        )
    )

    print("\nSigma:")
    print(
        np.round(
            Sigma,
            4
        )
    )

    print("\nRight Singular Vectors (V^T):")
    print(
        np.round(
            Vt,
            4
        )
    )


    projected_words = project_words(
        normalized_matrix,
        Vt
    )

    print("\nProjected Word Coordinates:")

    projected_df = pd.DataFrame(
        projected_words,
        index=vocabulary
    )

    print(
        projected_df.round(4)
    )


    print("\n" + "=" * 70)
    print("STEP 3 - LATENT SIMILARITY MAPPING")
    print("=" * 70)


    if "machine" in vocabulary:
        target_word = "machine"

    elif "data" in vocabulary:
        target_word = "data"

    else:
        target_word = vocabulary[0]

    closest_words = find_closest_words(
        target_word,
        vocabulary,
        projected_words,
        TOP_N
    )

    print(
        f"\nTarget word: '{target_word}'"
    )

    print(
        f"\nTop {TOP_N} closest words:"
    )

    for rank, (word, similarity) in enumerate(
        closest_words,
        start=1
    ):
        print(
            f"{rank}. {word:<20} "
            f"Cosine Similarity = {similarity:.6f}"
        )


    document_term_df = pd.DataFrame(
        matrix,
        index=vocabulary,
        columns=[
            document["filename"]
            for document in data
        ]
    )

    normalized_df = pd.DataFrame(
        normalized_matrix,
        index=vocabulary,
        columns=[
            document["filename"]
            for document in data
        ]
    )

    projected_df.to_csv(
        "projected_words.csv"
    )

    document_term_df.to_csv(
        "document_term_matrix.csv"
    )

    normalized_df.to_csv(
        "normalized_matrix.csv"
    )

    print("\n" + "=" * 70)
    print("LSA PIPELINE COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")
    print("1. document_term_matrix.csv")
    print("2. normalized_matrix.csv")
    print("3. projected_words.csv")

if __name__ == "__main__":
    main()