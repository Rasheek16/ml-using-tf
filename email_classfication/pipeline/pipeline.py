from sklearn.pipeline import Pipeline # type: ignore 
from preprocessing.transformers import EmailToWordCounterTransformer, WordCounterToVectorTransformer # type: ignore 

def build_pipeline(vocab_size=1000):
    return Pipeline([
        ("email_to_wordcount", EmailToWordCounterTransformer()),
        ("wordcount_to_vector", WordCounterToVectorTransformer(vocabulary_size=vocab_size)),
    ])
