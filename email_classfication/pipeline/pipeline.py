from sklearn.pipeline import Pipeline
from preprocessing.transformers import EmailToWordCounterTransformer, WordCounterToVectorTransformer

def build_pipeline(vocab_size=1000):
    return Pipeline([
        ("email_to_wordcount", EmailToWordCounterTransformer()),
        ("wordcount_to_vector", WordCounterToVectorTransformer(vocabulary_size=vocab_size)),
    ])
