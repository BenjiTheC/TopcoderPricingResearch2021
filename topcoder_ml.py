""" Natural Language Processing logic for building the price model."""
import pathlib
import itertools
from collections.abc import Iterable

import numpy as np
import pandas as pd
from gensim import corpora, models, similarities

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator

import topcoder_mongo as DB
import util as U
import static_var as S


def get_ngrams(s: str, n: int) -> list[str]:
    """ Get N-gram from a string."""
    return [''.join(gram) for gram in zip(*[s[i:] for i in range(n)])]


def group_challenge_tags() -> tuple[dict, corpora.Dictionary, similarities.SparseMatrixSimilarity]:
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


def challenge_tag_word2vec() -> tuple[dict, dict, models.Word2Vec]:
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


def softmax(x: np.ndarray) -> np.ndarray:
    """ Compute softmax values for the array."""
    return np.exp(x) / np.sum(np.exp(x))


def mean_magnitude_of_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Return the single float of MMRE."""
    return np.mean(np.abs(y_pred - y_true) / y_true)


def get_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Retrieve engineered feature matrix X and y from database.
        The order of array concatenation: numeric -> one hot encoded -> docvec
    """
    challenge_prize_query = [
        DB.TopcoderMongo.filter_valid_docvec_query(),
        {'$project': {'_id': False, 'id': True, 'top2_prize': True}},
    ]
    challenge_feature_query = [
        DB.TopcoderMongo.filter_valid_docvec_query(),
        {'$project': {
            '_id': False, 'id': True,
            'vector': {
                '$concatArrays': [
                    '$metadata',
                    ['$num_of_competing_challenges'],
                    f'$softmax_dim{S.CHALLENGE_TAG_OHE_DIM}',
                    f'$one_hot_dim{S.CHALLENGE_TAG_OHE_DIM}',
                    f'${S.DV_FEATURE_NAME}',
                ],
            },
        }},
    ]
    challenge_prize = (pd.DataFrame
                       .from_records(DB.TopcoderMongo.run_feature_aggregation(challenge_prize_query))
                       .set_index('id'))
    challenge_feature = (pd.DataFrame
                         .from_records((DB.TopcoderMongo.run_feature_aggregation(challenge_feature_query)))
                         .set_index('id'))
    challenge_feature = (pd.DataFrame
                         .from_records(data=challenge_feature.vector, index=challenge_feature.index)
                         .rename(columns=dict(enumerate(S.FEATURE_MATRIX_COLUMNS)))
                         .reindex(S.FEATURE_MATRIX_REINDEX, axis=1))

    feature_and_target = challenge_feature.join(challenge_prize)

    prize_lower_bound = challenge_prize['top2_prize'].quantile(0.05)
    prize_upper_bound = challenge_prize['top2_prize'].quantile(0.95)
    challenge_by_project_scale: list[str] = (pd.DataFrame.from_records(DB.TopcoderMongo.get_project_scale([0, 10]))
                                             .set_index('tag')
                                             .loc['>=10', 'challenge_lst'])

    selected_feature_and_target = feature_and_target.loc[
        (feature_and_target['top2_prize'] >= prize_lower_bound) &
        (feature_and_target['top2_prize'] <= prize_upper_bound) &
        feature_and_target.index.isin(challenge_by_project_scale)
    ]

    return (
        selected_feature_and_target.reindex(challenge_feature.columns, axis=1),
        selected_feature_and_target.reindex(['top2_prize'], axis=1),
    )


def get_challenge_prize_range() -> pd.DataFrame:
    """ Return the challenge id list with tag of prize range.
        This function is written to aggregate the challenge by prize range, and eventually
        using the prize range to split the dataset. However, this approach assume that we
        "know" the testing dataset distribution and is wrong, so it shoud NOT be used.
    """
    feature, target = get_training_data()
    target_min, target_max = target['top2_prize'].min(), target['top2_prize'].max()
    target_challenge = target.index.tolist()

    prize_interval_points = np.linspace(target_min, target_max, int((target_max - target_min) / 50) + 1)
    branch_query = [{
        'case': {'$and': [
            {'$gte': ['$top2_prize', intv_start]},
            {'$lte' if intv_end == prize_interval_points[-1] else '$lt': ['$top2_prize', intv_end]}]},
        'then': f'[{intv_start}, {intv_end}' + (']' if intv_end == prize_interval_points[-1] else ')'),
    } for intv_start, intv_end in zip(prize_interval_points, prize_interval_points[1:])]

    query = [
        DB.TopcoderMongo.filter_valid_docvec_query(),
        {'$set': {'prize_range': {'$switch': {'branches': branch_query, 'default': 'out_of_range'}}}},
        {'$match': {'prize_range': {'$ne': 'out_of_range'}, 'id': {'$in': target_challenge}}},
        {'$group': {
            '_id': {'prize_range': '$prize_range'},
            'challenge_ids': {'$push': '$id'},
        }},
        {'$replaceRoot': {'newRoot': {'$mergeObjects': ['$_id', {'challenge_ids': '$challenge_ids'}]}}},
    ]

    return pd.concat([
        pd.DataFrame({'id': doc['challenge_ids'], 'prize_range': doc['prize_range']})
        for doc in DB.TopcoderMongo.run_feature_aggregation(query)
    ]).reset_index(drop=True)


def get_train_test_index(test_size: float = 0.15) -> tuple[list[str], list[str]]:
    """ Get the training and testing challenge id list.
        This function is written to group and resample the testing dataset. However, this
        approach assume that we "know" the testing dataset distribution and is wrong, so
        it shoud NOT be used.
    """
    challenge_prize_range = get_challenge_prize_range()
    test_challenge_id: list[str] = (challenge_prize_range.groupby('prize_range')
                                                         .sample(frac=test_size, random_state=42)
                                                         .loc[:, 'id']
                                                         .to_list())
    train_challenge_id: list[str] = (challenge_prize_range
                                     .loc[~challenge_prize_range['id'].isin(test_challenge_id)]['id']
                                     .to_list())

    return train_challenge_id, test_challenge_id


def construct_training_pipeline(
    estimator: BaseEstimator = GradientBoostingRegressor,
    est_name: str = 'est',
    est_param: dict = {},
) -> Pipeline:
    """ Construct a sklearn.pipeline.Pipeline with ColumnTransformer.
        Potentially more
    """
    return Pipeline([
        ('col', ColumnTransformer(
            [('standardization', StandardScaler(), S.NUMERIC_FEATURES)],
            remainder='passthrough',
        )),
        (est_name, estimator(**est_param)),
    ])
