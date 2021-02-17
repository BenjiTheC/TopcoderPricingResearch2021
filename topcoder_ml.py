""" Natural Language Processing logic for building the price model."""
import pathlib
import itertools
from collections.abc import Iterable

import numpy as np
from gensim import corpora, models, similarities

import topcoder_mongo as DB
import util as U

def get_ngrams(s: str, n: int) -> list[str]:
    """ Get N-gram from a string."""
    return [''.join(gram) for gram in zip(*[s[i:] for i in range(n)])]

def group_challenge_tags():
    """ Use TF-IDF model to group the tag that are similar.
        Perfect example of over-engineering.

        And it's not working so forget about it...
    """

    query = [
        {'$match': {
            'status': 'Completed',
            'track': 'Development',
            'type': 'Challenge',
            'end_date': {'$lte': U.year_end(2020)},     
        }},
        {'$unwind': '$tags'},
        {'$group': {'_id': {'tag': '$tags'}, 'tag': {'$first': '$tags'}}},
        {'$project': {'_id': False}},
    ]
    unique_tags: list[str] = [tag['tag'] for tag in DB.TopcoderMongo.run_challenge_aggregation(query)]
    unique_tags.remove('Other')  # 'Other' tag is meaningless

    unique_tags_no_space: list[str] = [''.join(tag.lower().split()) for tag in unique_tags]
    unique_tags_multigram: list[Iterable[str]] = [
        list(itertools.chain(*[get_ngrams(tag, i) for i in range(3)])) for tag in unique_tags_no_space
    ]

    # tag_to_multigram = dict(zip(unique_tags, unique_tags_multigram))
    grams_to_id = corpora.Dictionary(unique_tags_multigram)
    unique_tags_bow = [grams_to_id.doc2bow(multigram) for multigram in unique_tags_multigram]

    tfidf = models.TfidfModel(unique_tags_bow, dictionary=grams_to_id)
    unique_tags_tfidf = tfidf[unique_tags_bow]
    similarity_index = similarities.SparseMatrixSimilarity(unique_tags_tfidf, num_features=len(grams_to_id))

    tag_to_tfidf = dict(zip(unique_tags, unique_tags_tfidf))

    return tag_to_tfidf, grams_to_id, similarity_index


def challenge_tag_word2vec():
    """ Train a Word2Vec model in the fields of challenge tags
        Another perfect example of over engineering

        Not very meaingful either....
    """
    model_path: pathlib.Path = pathlib.Path('./models/challenge_tag_word2vec')

    tag_count_query = [
        *DB.TopcoderMongo.scoped_challenge_query,
        {'$unwind': '$tags'},
        {'$group': {'_id': {'tag': '$tags'}, 'count': {'$sum': 1}}},
        {'$replaceRoot': {'newRoot': {'$mergeObjects': ['$_id', {'count': '$count'}]}}},
    ]
    challenge_tag_query = [
        *DB.TopcoderMongo.scoped_challenge_query,
        {'$project': {'id': True, 'tags': True}},
    ]

    tag_count = {
        tag['tag']: tag['count']
        for tag in DB.TopcoderMongo.run_challenge_aggregation(tag_count_query)
        if tag['count'] >= 5
    }
    challenge_tags = {
        challenge['id']: sorted(tag for tag in challenge['tags'] if tag in tag_count and tag != 'Other')
        for challenge in DB.TopcoderMongo.run_challenge_aggregation(challenge_tag_query)
    }

    if model_path.exists():
        print('Returning existed model')
        word2vec = models.Word2Vec.load(str(model_path.resolve()))
    else:
        print('Returning newly trained model')
        word2vec = models.Word2Vec(sentences=challenge_tags.values(), workers=4)
        word2vec.save(str(model_path.resolve()))

    return tag_count, challenge_tags, word2vec


def softmax(x: np.ndarray):
    """ Compute softmax values for the array."""
    return np.exp(x) / np.sum(np.exp(x))
