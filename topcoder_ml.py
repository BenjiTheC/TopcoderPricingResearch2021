""" Natural Language Processing logic for building the price model."""
import pathlib
import argparse
from pprint import pprint
from itertools import chain

import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.sklearn_api import D2VTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.metrics import make_scorer

import topcoder_mongo as DB
import util as U
import static_var as S


def softmax(x: np.ndarray) -> np.ndarray:
    """ Compute softmax values for the array."""
    return np.exp(x) / np.sum(np.exp(x))


def mean_magnitude_of_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Return the single float of MMRE."""
    return np.mean(np.abs(y_pred - y_true) / y_true)


def get_training_data(
    excluded_metadata: list = [],
    excluded_global_context: list = [],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Retrieve engineered feature matrix X and y from database.
        The order of array concatenation: numeric -> one hot encoded -> docvec
    """
    challenge_prize_query = [
        {'$project': {'_id': False, 'id': True, 'top2_prize': True}},
    ]
    challenge_feature_query = [
        {'$project': {
            '_id': False, 'id': True,
            'vector': {
                '$concatArrays': [
                    '$metadata',
                    [f'${col}' for col in S.GLOBAL_CONTEXT_COLUMNS],
                    f'$softmax_dim{S.CHALLENGE_TAG_OHE_DIM}',
                    f'$one_hot_dim{S.CHALLENGE_TAG_OHE_DIM}',
                ],
            },
        }},
    ]

    # get the challenge description text and process it into tokens
    challenge_desc: pd.DataFrame = (pd.DataFrame
                                    .from_records(DB.TopcoderMongo.get_challenge_description())
                                    .set_index('id'))
    challenge_desc['tokens'] = challenge_desc['processed_paragraph'].apply(simple_preprocess)
    challenge_desc = (challenge_desc
                      .loc[challenge_desc['tokens'].apply(lambda t: len(t) > S.DOC2VEC_CONFIG.token_length)]
                      .reindex(['tokens'], axis=1)
                      .rename(columns={'tokens': 'processed_paragraph'}))

    challenge_prize = (pd.DataFrame
                       .from_records(DB.TopcoderMongo.run_feature_aggregation(challenge_prize_query))
                       .set_index('id'))

    # `vector` column is a Series of lists, so we need to `from_records` to flat out the matrix
    challenge_feature = (pd.DataFrame
                         .from_records((DB.TopcoderMongo.run_feature_aggregation(challenge_feature_query)))
                         .set_index('id'))
    challenge_feature = (pd.DataFrame
                         .from_records(data=challenge_feature.vector, index=challenge_feature.index)
                         .rename(columns=dict(enumerate(S.FEATURE_MATRIX_COLUMNS)))
                         .reindex(S.FEATURE_MATRIX_REINDEX, axis=1))

    # challenge_desc can be potentially less than other features when `DOC2VEC_CONFIG.token_length` changes.
    feature_and_target = challenge_desc.join(challenge_feature.join(challenge_prize))

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
        selected_feature_and_target.reindex(
            [col for col in S.FEATURE_MATRIX_COLUMNS
             if col not in (excluded_metadata + excluded_global_context)] + ['processed_paragraph'],
            axis=1,
        ),
        selected_feature_and_target.reindex(['top2_prize'], axis=1),
    )


def construct_training_pipeline(
    estimator: BaseEstimator = GradientBoostingRegressor,
    est_name: str = 'est',
    est_param: dict = {},
    excluded_metadata: list = [],
    excluded_global_context: list = [],
) -> Pipeline:
    """ Construct a sklearn.pipeline.Pipeline with ColumnTransformer.
        Potentially more
    """
    return Pipeline([
        ('col', ColumnTransformer(
            [
                ('standardization',
                 StandardScaler(),
                 [col for col in S.NUMERIC_FEATURES if col not in (excluded_metadata + excluded_global_context)]),
                ('doc2vec',
                 Pipeline([
                     ('txt_extraction', FunctionTransformer(lambda df: df['processed_paragraph'].to_list())),
                     ('doc2vec', D2VTransformer(size=S.DOC2VEC_CONFIG.dimension, min_count=5, iter=10)),
                 ]),
                 ['processed_paragraph']),
            ],
            remainder='passthrough',
        )),
        (est_name, estimator(**est_param)),
    ])


def cross_validation_with_time_window(
    feature: pd.DataFrame,
    target: pd.DataFrame,
    train_time_span: int = 1,
) -> list[tuple[pd.Timestamp, np.float64]]:
    """ Cross validate the model using specific time window."""
    query = [
        *DB.TopcoderMongo.scoped_challenge_with_text_query,
        {'$group': {
            '_id': {'$dateToString': {'format': '%Y-%m', 'date': '$end_date'}},
            'id_lst': {'$addToSet': '$id'},
        }},
        {'$replaceRoot': {'newRoot': {
            'month': '$_id',
            'challenges': '$id_lst',
        }}},
    ]
    challenge_by_month = (pd.DataFrame
                          .from_records(DB.TopcoderMongo.run_challenge_aggregation(query))
                          .set_index('month')
                          .sort_index())
    challenge_by_month.index = pd.to_datetime(challenge_by_month.index)

    cv_scores = []
    for time_window in U.slide_window(challenge_by_month.index, train_time_span + 1):
        train_idx = list(chain.from_iterable(
            challenge_by_month
            .loc[time_window[:train_time_span], 'challenges']
            .to_list()
        ))
        test_idx = challenge_by_month.loc[time_window[-1], 'challenges']

        train_feature = feature.loc[feature.index.isin(train_idx)]
        test_feature = feature.loc[feature.index.isin(test_idx)]

        train_target = target.loc[target.index.isin(train_idx)]
        test_target = target.loc[target.index.isin(test_idx)]

        est = construct_training_pipeline(est_param=dict(random_state=42))
        est.fit(train_feature, train_target.to_numpy().reshape(-1))
        pred_target = est.predict(test_feature)
        score = mean_magnitude_of_relative_error(test_target.to_numpy().reshape(-1), pred_target)

        cv_scores.append((time_window[-1], score))

    return cv_scores


def command_line_cross_validate():
    """ Function to call in shell to run cross validation."""
    parser = argparse.ArgumentParser(description='Run cross validation of pricing model with different config')
    parser.add_argument(
        '--exclude-metadata', '-m',
        dest='excluded_metadata',
        nargs='*',
        default=[],
        help='Metadata features to exclude.',
    )
    parser.add_argument(
        '--exclude-global-context', '-g',
        dest='excluded_global_context',
        nargs='*',
        default=[],
        help='Global context features to exclude.',
    )
    args = parser.parse_args()

    feature, target = get_training_data(
        excluded_metadata=args.excluded_metadata,
        excluded_global_context=args.excluded_global_context,
    )
    score = np.abs(np.mean(cross_val_score(
        construct_training_pipeline(
            excluded_metadata=args.excluded_metadata,
            excluded_global_context=args.excluded_global_context,
        ),
        feature, target.to_numpy().reshape(-1),
        scoring=make_scorer(mean_magnitude_of_relative_error, greater_is_better=False),
        cv=ShuffleSplit(n_splits=10, test_size=0.3, random_state=42),
    )))

    result = {
        'excluded_metadata': args.excluded_metadata,
        'excluded_global_context': args.excluded_global_context,
        'tag_ohe_dimension': S.CHALLENGE_TAG_OHE_DIM,
        'doc2vec_dimension': S.DOC2VEC_CONFIG.dimension,
        'mmre': score,
    }
    pprint(result)
    U.json_list_append(result, pathlib.Path('./cross_validation_result.json'))


if __name__ == '__main__':
    command_line_cross_validate()
