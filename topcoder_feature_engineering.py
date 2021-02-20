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
        *DB.TopcoderMongo.scoped_challenge_query,
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


def compute_description_doc2vec(
    similarity_threshold: typing.Optional[float] = None,
    frequency_threshold: typing.Optional[float] = None,
    token_len_threshold: int = 0,
) -> list[dict]:
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