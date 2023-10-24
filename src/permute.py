from sklearn.pipeline import Pipeline


def permute(transformers: [Pipeline], classifiers: [Pipeline]) -> [Pipeline]:
    """
    Creates a new array with a permutation of all the transformers pipelines & all the classifier pipelines provided.
    """
    pipelines = []

    for transformer in transformers:
        for classifier in classifiers:
            pipelines.append(Pipeline([
                ('transformers', transformer),
                ('classifier', classifier),
            ]))

    return pipelines
