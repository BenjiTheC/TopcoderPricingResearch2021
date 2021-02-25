""" Some logic that's meant to run in the jupyter notebook."""
import typing
import pathlib
import itertools
import numpy as np
import pandas as pd
import static_var as S
import topcoder_ml as TML
import topcoder_mongo as DB
from collections.abc import Iterator
from pandas.api.types import CategoricalDtype
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def get_challenge_tag_combination_count() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Extract challenge tag list and get the combination of 2 and 3 tags."""
    def count_tag_combination(challenge_tag_it: Iterator) -> pd.DataFrame:
        tag_combinatio_count = (pd.DataFrame
                                .from_records(
                                    pd.Series(challenge_tag_it)
                                    .apply(lambda l: [c if isinstance(c, str) else tuple(sorted(c)) for c in l])
                                )
                                .fillna('')
                                .stack()
                                .value_counts()
                                .to_frame()
                                .reset_index()
                                .rename(columns={'index': 'tag', 0: 'count'}))
        return tag_combinatio_count.loc[tag_combinatio_count['tag'].astype(bool)].reset_index(drop=True)

    challenge_tags_cursor = DB.TopcoderMongo.run_challenge_aggregation([
        *DB.TopcoderMongo.scoped_challenge_with_text_query,
        {'$project': {'tags': True, '_id': False}},
    ])

    it0, it1, it2, it3 = itertools.tee((doc['tags'] for doc in challenge_tags_cursor), 4)
    return (
        count_tag_combination(it0),
        count_tag_combination(itertools.combinations(tags, 2) for tags in it1),
        count_tag_combination(itertools.combinations(tags, 3) for tags in it2),
        count_tag_combination(itertools.combinations(tags, 4) for tags in it3),
    )


def get_tag_combination_softmax() -> list[pd.DataFrame]:
    """ Calculate Tag combination's softmax score from frequency count.
        Separate this part of logic from `get_challenge_tag_combination_count`
        to preserve the total combination count df.
    """
    def compute_softmax(tag_combination: pd.DataFrame):
        """ Calculate softmax for tag combination DataFrame."""
        top25 = tag_combination.head(25).copy()
        top25['count_softmax'] = TML.softmax(np.log(top25['count']))
        return top25

    return [compute_softmax(tag_combination) for tag_combination in get_challenge_tag_combination_count()]


def compute_tag_feature() -> list[dict]:
    """ Use the tag combination softmax table to caluate the softmax score of a
        challenge's tags. And encode the binary array.
    """
    tag_comb_softmax: list[pd.DataFrame] = get_tag_combination_softmax()
    challenge_tag = DB.TopcoderMongo.run_challenge_aggregation([
        *DB.TopcoderMongo.scoped_challenge_with_text_query,
        {'$project': {'id': True, 'tags': True, '_id': False}},
    ])

    def map_tag_lst_to_softmax(tags: list[str]) -> dict[str, dict]:
        """ Encode the tag list into one-hot list and sum of softmax.
            Short var name `tc` stands for `tag_combination`.
        """
        feature_dct = {}
        for comb_r, tc_softmax in enumerate(tag_comb_softmax, 1):
            tc_lst = tags if comb_r == 1 else [tuple(sorted(tc)) for tc in itertools.combinations(tags, comb_r)]
            softmax_score = tc_softmax.loc[tc_softmax['tag'].isin(tc_lst), 'count_softmax'].sum()
            one_hot_array = tc_softmax['tag'].isin(tc_lst).astype(int).to_numpy()
            feature_dct.update({
                f'tag_comb{comb_r}_softmax_score': softmax_score,
                f'tag_comb{comb_r}_one_hot_array': one_hot_array,
            })
        return feature_dct

    return [{**cha, **map_tag_lst_to_softmax(cha['tags'])} for cha in challenge_tag]


def train_challenge_desc_doc2vec(
    similarity_threshold: typing.Optional[float] = None,
    frequency_threshold: typing.Optional[float] = None,
    token_len_threshold: int = 0,
) -> tuple[Doc2Vec, list[TaggedDocument]]:
    """ Retrieve challenge description from meaningful processed description."""
    challenge_description = pd.DataFrame.from_records(
        DB.TopcoderMongo.get_challenge_description(similarity_threshold, frequency_threshold)
    )
    challenge_description['tokens'] = challenge_description['processed_paragraph'].apply(simple_preprocess)

    corpus = [
        TaggedDocument(words=row.tokens, tags=[row.id])
        for row in (challenge_description
                    .loc[challenge_description['tokens'].apply(lambda t: len(t)) > token_len_threshold]
                    .itertuples())
    ]

    model_path: pathlib.Path = (
        S.MODEL_PATH / 
        ('challenge_desc_docvecs_' +
            f'sim{similarity_threshold}freq{frequency_threshold}tkl{token_len_threshold}')
    )

    if model_path.exists():
        return Doc2Vec.load(str(model_path.resolve())), corpus

    model = Doc2Vec(vector_size=100, min_count=5, epochs=10)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(str(model_path.resolve()))

    return model, corpus


def compute_challenge_desc_docvec(
    similarity_threshold: typing.Optional[float] = None,
    frequency_threshold: typing.Optional[float] = None,
    token_len_threshold: int = 0,
) -> dict[str, list]:
    """ Compute the document vector representation of a challenge description with
        given similarity, frequency and token length threshold.
    """
    model, corpus = train_challenge_desc_doc2vec(similarity_threshold, frequency_threshold, token_len_threshold)
    return {doc.tags[0]: model.docvecs[doc.tags[0]].tolist() for doc in corpus}


def compute_challenge_metadata():
    """ Compute challenge metadata:
        - Challenge duration (by full days)
        - Project id (categorically encoded)
        - Legacy sub track (categorically encoded)
    """
    proj_id_query = [
        *DB.TopcoderMongo.scoped_challenge_with_text_query,
        {'$group': {'_id': None, 'project_ids': {'$addToSet': '$project_id'}}},
    ]
    sub_track_query = [
        {'$match': {'legacy.track': 'DEVELOP'}},
        {'$group': {'_id': None, 'legacy_sub_tracks': {'$addToSet': '$legacy.sub_track'}}},
    ]
    project_ids = next(
        DB.TopcoderMongo.run_challenge_aggregation(proj_id_query)
    )['project_ids']
    sub_tracks = next(
        (DB.TopcoderMongo
            .run_challenge_aggregation(sub_track_query))
    )['legacy_sub_tracks']

    project_id_cat = CategoricalDtype(sorted(project_ids))
    sub_track_cat = CategoricalDtype(sorted(sub_tracks))

    challenge_metadata_query = [
        *DB.TopcoderMongo.scoped_challenge_with_text_query,
        {'$project': {
            '_id': False, 'id': True, 'project_id': True,
            'legacy_sub_track': '$legacy.sub_track',
            'duration': {'$toInt': {
                '$divide': [
                    {'$subtract': ['$end_date', '$start_date']},
                    24 * 60 * 60 * 1000,
                ],
            }},
        }},
    ]
    challenge_metadata = (pd.DataFrame
                            .from_records(
                                DB.TopcoderMongo.run_challenge_aggregation(challenge_metadata_query)
                            ).astype({
                                'project_id': project_id_cat,
                                'legacy_sub_track': sub_track_cat
                            }))

    challenge_metadata['project_id_encode'] = challenge_metadata['project_id'].cat.codes
    challenge_metadata['legacy_sub_track_encode'] = challenge_metadata['legacy_sub_track'].cat.codes

    return challenge_metadata


def compute_competing_challenges():
    """ For each challenge, find all competing challenges."""
    challenge_start_end_query = [
        *DB.TopcoderMongo.scoped_challenge_with_text_query,
        {'$project': {'_id': False, 'id': True, 'start_date': True, 'end_date': True}},
    ]
    challenge_start_end = [
        (cha['id'], cha['start_date'], cha['end_date'])
        for cha in DB.TopcoderMongo.run_challenge_aggregation(challenge_start_end_query)
    ]

    competing_challenge_facet = {'$facet': {cha_id: [
        {'$match': {
            'id': {'$ne': cha_id},
            '$expr': {'$or': [  # Either the start date or end date is in the given challenge's duration.
                {'$and': [{'$gt': ['$start_date', start_date]}, {'$lt': ['$start_date', end_date]}]},
                {'$and': [{'$gt': ['$end_date', start_date]}, {'$lt': ['$end_date', end_date]}]},
            ]},
        }},
        {'$group': {'_id': None, 'competing_challenge_ids': {'$push': '$id'}}},
        {'$project': {
            '_id': False,
            'competing_challenge_ids': True,
            'num_of_competing_challenges': {'$size': '$competing_challenge_ids'},
        }},
    ] for cha_id, start_date, end_date in challenge_start_end}}

    # We are not unpacking the facet as there will be challenges with NO competing challenges.
    get_competing_challenge_query = [
        *DB.TopcoderMongo.scoped_challenge_with_text_query,
        competing_challenge_facet,  # This is 5,000+ pipelines in parallel, so it CAN blow the memory out
    ]

    competing_challenges: dict = list(DB.TopcoderMongo.run_challenge_aggregation(get_competing_challenge_query))[0]
    empty_competing_cha = {'competing_challenge_ids': [], 'num_of_competing_challenges': 0}

    return [
        {'id': cha_id, **(empty_competing_cha if competing_cha == [] else competing_cha[0])}
        for cha_id, competing_cha in competing_challenges.items()
    ]
