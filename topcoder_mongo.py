""" Retrieve data from mongoDB."""
import re
import typing
import pymongo
import pymongo.errors
import pymongo.cursor
import pymongo.database
import pymongo.collection
import pandas as pd
import numpy as np
import topcoder_feature_engineering as FE
import static_var as S
import util as U
from url import URL
from pprint import pprint
from dateutil.parser import isoparse


MONGO_CLIENT = None


def construct_mongo_url():
    """ Construct URL for connecting to MongoDB."""
    url = URL('')
    if S.MONGO_CONFIG.host in ['127.0.0.1', 'localhost']:
        url.scheme = 'mongodb'
        url.netloc = f'{S.MONGO_CONFIG.host}:{S.MONGO_CONFIG.port}'
    else:
        url.scheme = 'mongodb+srv'
        url.netloc = f'{S.MONGO_CONFIG.username}:{S.MONGO_CONFIG.password}@{S.MONGO_CONFIG.host}'
        url.path = S.MONGO_CONFIG.database
        url.query_param.set('retryWrites', 'true')
        url.query_param.set('w', 'majority')
    return str(url)


def connect() -> pymongo.database.Database:
    """ Connect to MongoDB."""
    global MONGO_CLIENT
    if MONGO_CLIENT is None:
        MONGO_CLIENT = pymongo.MongoClient(construct_mongo_url())

    database = MONGO_CLIENT[S.MONGO_CONFIG.database]
    return database


def get_collection(collection_name: str) -> pymongo.collection.Collection:
    """ Get the collection by collection name."""
    database = connect()
    return database.get_collection(collection_name)


class TopcoderMongo:
    """ Retrieve data from MongoDB."""
    challenge = get_collection('challenge')
    project = get_collection('project')
    feature = get_collection('feature')

    scoped_challenge_query = [  # This query select the 5,996 challenges fall into the scope
        {'$match': {
            'status': 'Completed',
            'track': 'Development',
            'type': 'Challenge',
            'end_date': {'$lte': U.year_end(2020)},
            'legacy.sub_track': {'$exists': True},
            'project_id': {'$ne': None},
            '$expr': {'$lte': ['$start_date', '$end_date']},
        }},
        {'$unwind': '$prize_sets'},
        {'$match': {'prize_sets.type': 'placement'}},
        {'$set': {'num_of_placements': {'$size': '$prize_sets.prizes'}}},
        {'$match': {'num_of_placements': {'$gt': 0}}},
    ]

    scoped_challenge_with_text_query = [  # This query select the 3,316 challenges fall into the scope
        *scoped_challenge_query,
        {'$set': {'filtered_processed_desc': {
            '$filter': {
                'input': '$processed_description',
                'as': 'desc',
                'cond': {'$not': [{'$in': ['$$desc.name', ['Final Submission Guidelines', 'Payments']]}]},
            }
        }}},
        {'$set': {'size_of_filtered_processed_desc': {'$size': '$filtered_processed_desc'}}},
        {'$match': {'size_of_filtered_processed_desc': {'$gt': 0}}},
    ]

    @classmethod
    def filter_valid_docvec_query(cls) -> dict:
        """ Query for challenges with valid docvec."""
        return {
            '$match': {
                S.DV_FEATURE_NAME: {'$exists': True},
                'top2_prize': {'$gt': 0},
            },
        }

    @classmethod
    def run_challenge_aggregation(self, query: list[dict]) -> typing.Any:
        """ Run mongo aggregation"""
        try:
            result = self.challenge.aggregate(query)
        except pymongo.errors.OperationFailure as e:
            print('Challenge aggregation failed. Details:')
            pprint(e.details)
        else:
            return result

    @classmethod
    def run_project_aggregation(self, query: list[dict]) -> typing.Any:
        """ Run mongo aggregation"""
        try:
            result = self.project.aggregate(query)
        except pymongo.errors.OperationFailure as e:
            print('Project aggregation failed. Details:')
            pprint(e.details)
        else:
            return result

    @classmethod
    def run_feature_aggregation(self, query: list[dict]) -> typing.Any:
        """ Run mongo aggregation"""
        try:
            result = self.feature.aggregate(query)
        except pymongo.errors.OperationFailure as e:
            print('Feature aggregation failed. Details:')
            pprint(e.details)
        else:
            return result

    @classmethod
    def get_challenge_count(
        cls,
        granularity: str = 'year',
        fmt: str = '%Y',
        status_filter: list[str] = S.STATUS
    ) -> list[dict]:
        """ Get the overview count of challenges by year/month/other granularity."""
        query = [
            {'$match': {'end_date': {'$lte': U.year_end(2020)}}},
            {
                '$project': {
                    'status': {'$arrayElemAt': [{'$split': ['$status', ' - ']}, 0]},
                    'end_date': '$end_date',
                },
            },
            {'$match': {'$expr': {'$in': ['$status', status_filter]}}},
            {
                '$group': {
                    '_id': {
                        granularity: {'$dateToString': {'format': fmt, 'date': '$end_date'}},
                        'status': '$status',
                    },
                    'count': {'$sum': 1},
                },
            },
            {'$replaceRoot': {'newRoot': {'$mergeObjects': ['$_id', {'count': '$count'}]}}},
        ]

        result = [{**cnt, granularity: isoparse(cnt[granularity])} for cnt in cls.challenge.aggregate(query)]

        return result

    @classmethod
    def get_cancelled_challenges_count(cls, granularity: str = 'year', fmt: str = '%Y') -> list[dict]:
        """ Get only Cancelled challenge."""
        query = [
            {'$match': {'status': {'$regex': re.compile(r'^Cancelled')}}},
            {
                '$group': {
                    '_id': {
                        granularity: {'$toInt': {'$dateToString': {'format': fmt, 'date': '$end_date'}}},
                        'status': '$status',
                    },
                    'count': {'$sum': 1},
                }
            },
            {'$replaceRoot': {'newRoot': {'$mergeObjects': ['$_id', {'count': '$count'}]}}},
        ]

        result = cls.challenge.aggregate(query)

        return list(result)

    @classmethod
    def get_project_section_similarity(cls) -> list[dict]:
        """ Get project common section similarity table for display."""
        query = [
            {'$project': {'id': True, 'section_similarity': True}},
            {'$unwind': '$section_similarity'},
            {'$replaceRoot': {'newRoot': {'$mergeObjects': [{'project_id': '$id'}, '$section_similarity']}}},
        ]

        return list(cls.project.aggregate(query))

    @classmethod
    def get_project_repeated_sections(
        cls,
        similarity_threshold: float = 1.1,
        frequency_threshold: float = 1.1,
    ) -> dict[str: list[str]]:
        """ Get the project section names by frequency and similarity threshold
            for conostructing challenge description
        """
        query = [
            {'$project': {'id': True, 'section_similarity': True}},
            {'$unwind': '$section_similarity'},
            {'$replaceRoot': {'newRoot': {'$mergeObjects': [{'project_id': '$id'}, '$section_similarity']}}},
            {'$match': {
                'similarity': {'$gte': similarity_threshold},
                'frequency': {'$gte': frequency_threshold},
                'name': {'$not': {'$in': ['Final Submission Guidelines', 'Payments']}},
            }},
            {'$group': {
                '_id': {'project_id': '$project_id'},
                'names': {'$push': '$name'},
            }},
            {'$replaceRoot': {'newRoot': {'$mergeObjects': [
                '$_id',
                {'names': '$names'},
            ]}}},
        ]

        return {proj['project_id']: proj['names'] for proj in cls.project.aggregate(query)}

    @classmethod
    def get_project_scale(cls, interval_points: list[int] = [0, 10, 50, 100]) -> pymongo.cursor.Cursor:
        """ Return a cursor (iterable) for tagged projects.
            This function take ONLY Dvelopment track challenge into consideration.
            And capitalized words 'Challenge' 'First2Finish' 'Task' stand for challenge type.
        """
        intervals = zip(interval_points, [i - 1 for i in interval_points[1:]])
        branch_query = [
            {
                'case': {'$and': [
                    {'$gte': ['$num_of_challenge', intv_start]},
                    {'$lte': ['$num_of_challenge', intv_end]}]},
                'then': f'{intv_start}~{intv_end}'
            } for intv_start, intv_end in intervals
        ]
        branch_query.append({
            'case': {'$gte': ['$num_of_challenge', interval_points[-1]]},
            'then': f'>={interval_points[-1]}'}
        )

        query = [
            {'$match': {'$expr': {'$in': ['Development', '$tracks']}}},
            {'$unwind': '$num_of_challenge'},
            {'$match': {'num_of_challenge.track': 'Development'}},
            {'$unwind': '$completion_ratio'},
            {'$match': {'completion_ratio.track': 'Development'}},
            {
                '$set': {
                    'dev_cha_lst': {
                        '$filter': {
                            'input': '$challenge_lst',
                            'as': 'cha',
                            'cond': {'$eq': ['$$cha.track', 'Development']},
                        },
                    },
                    'num_of_completed': {  # number of completed development challenge
                        '$size': {
                            '$filter': {
                                'input': '$challenge_lst',
                                'as': 'cha',
                                'cond': {'$and': [
                                    {'$eq': ['$$cha.status', 'Completed']},
                                    {'$eq': ['$$cha.track', 'Development']},
                                ]},
                            }
                        },
                    },
                    **{f'num_of_completed_{challenge_type}': {
                        '$size': {
                            '$filter': {
                                'input': '$challenge_lst',
                                'as': 'cha',
                                'cond': {'$and': [
                                    {'$eq': ['$$cha.status', 'Completed']},
                                    {'$eq': ['$$cha.track', 'Development']},
                                    {'$eq': ['$$cha.type', challenge_type]},
                                ]}
                            }
                        }
                    } for challenge_type in S.TYPE},
                    **{f'num_of_{challenge_type}': {
                        '$size': {
                            '$filter': {
                                'input': '$challenge_lst',
                                'as': 'cha',
                                'cond': {'$and': [
                                    {'$eq': ['$$cha.track', 'Development']},
                                    {'$eq': ['$$cha.type', challenge_type]},
                                ]}
                            }
                        }
                    } for challenge_type in S.TYPE},
                },
            },
            {
                '$project': {
                    'num_of_challenge': '$num_of_challenge.count',
                    'completion_ratio': '$completion_ratio.ratio',
                    'num_of_completed': True,
                    **{f'num_of_completed_{challenge_type}': True for challenge_type in S.TYPE},
                    **{f'num_of_{challenge_type}': True for challenge_type in S.TYPE},
                    'challenge_lst': '$dev_cha_lst.id',
                }
            },
            {'$set': {'num_of_cha_range': {'$switch': {'branches': branch_query}}}},
            {
                '$group': {
                    '_id': '$num_of_cha_range',
                    'num_of_project': {'$sum': 1},
                    'num_of_challenge': {'$sum': '$num_of_challenge'},
                    'num_of_completed': {'$sum': '$num_of_completed'},
                    **{f'num_of_completed_{challenge_type}': {
                        '$sum': f'$num_of_completed_{challenge_type}'
                    } for challenge_type in S.TYPE},
                    **{f'num_of_{challenge_type}': {
                        '$sum': f'$num_of_{challenge_type}'
                    } for challenge_type in S.TYPE},
                    'avg_completion_ratio': {'$avg': '$completion_ratio'},
                    'all_challenge_lst': {'$push': '$challenge_lst'}
                },
            },
            {
                '$project': {
                    'tag': '$_id',
                    'num_of_project': True,
                    'num_of_challenge': True,
                    'num_of_completed': True,
                    **{f'num_of_completed_{challenge_type}': True for challenge_type in S.TYPE},
                    **{f'num_of_{challenge_type}': True for challenge_type in S.TYPE},
                    'avg_completion_ratio': {'$round': ['$avg_completion_ratio', 3]},
                    'challenge_lst': {
                        '$reduce': {
                            'input': '$all_challenge_lst',
                            'initialValue': [],
                            'in': {'$concatArrays': ['$$value', '$$this']}
                        }
                    }
                },
            },
            {'$sort': {'tag': pymongo.ASCENDING}},
            {'$project': {'_id': False}}
        ]
        return cls.project.aggregate(query)

    @classmethod
    def get_challenge_description(cls) -> pymongo.cursor.Cursor:
        """ Retrieve the challenge description text paragraph from filtered processed description section."""
        project_section_names = cls.get_project_repeated_sections(
            S.DOC2VEC_CONFIG.similarity,
            S.DOC2VEC_CONFIG.frequency
        )
        scoped_project_section_names: dict[str, list[str]] = {
            project['project_id']: project_section_names.get(project['project_id'], [])
            for project in cls.challenge.aggregate([
                *cls.scoped_challenge_with_text_query,
                {'$group': {'_id': None, 'project_id': {'$addToSet': '$project_id'}}},
                {'$unwind': '$project_id'},
                {'$project': {'_id': False}},
            ])
        }

        facet_query = {'$facet': {str(proj_id): [
            {'$match': {'project_id': proj_id}},
            {'$project': {
                '_id': False, 'id': True,
                'processed_paragraph': {
                    '$reduce': {
                        'input': {
                            '$map': {
                                'input': {
                                    '$filter': {
                                        'input': '$filtered_processed_desc',
                                        'as': 'fpd',
                                        'cond': {'$not': [{'$in': ['$$fpd.name', section_names]}]},
                                    },
                                },
                                'as': 'fpd',
                                'in': {'$concat': [
                                    {'$cond': [{'$eq': ['$$fpd.name', 'null']}, '', '$$fpd.name']},
                                    '$$fpd.text',
                                ]},
                            },
                        },
                        'initialValue': '',
                        'in': {'$concat': ['$$value', '$$this']},
                    }
                }
            }},
        ] for proj_id, section_names in scoped_project_section_names.items()}}

        facet_unpack_query = [
            {'$project': {'tmp_root': {'$setUnion': [f'${proj_id}' for proj_id in scoped_project_section_names]}}},
            {'$unwind': '$tmp_root'},
            {'$replaceRoot': {'newRoot': '$tmp_root'}},
        ]

        return cls.challenge.aggregate([
            *cls.scoped_challenge_with_text_query,
            facet_query,
            *facet_unpack_query,
        ])

    @classmethod
    def write_training_feature(cls, drop_before_write: bool = True) -> None:
        """ Bundle the writing method for `feature` collections."""
        if drop_before_write:
            cls.feature.drop()
        cls.write_prize_target()
        cls.write_tag_feature()
        cls.write_docvec_feature()
        cls.write_metadata_feature()

    @classmethod
    def write_prize_target(cls) -> None:
        """ Write the training target Challenge top2 prize."""
        query = [
            *cls.scoped_challenge_with_text_query,
            {'$unwind': '$prize_sets'},
            {'$match': {'prize_sets.type': 'placement'}},
            {'$project': {
                '_id': False,
                'id': True,
                'top2_prize': {'$sum': {'$slice': ['$prize_sets.prizes.value', 2]}},
            }},
        ]
        for challenge in cls.challenge.aggregate(query):
            cls.feature.update_one(
                {'id': challenge['id']},
                {'$set': {'id': challenge['id'], 'top2_prize': challenge['top2_prize']}},
                upsert=True,
            )

    @classmethod
    def write_tag_feature(cls) -> None:
        """ Write the engineered feature into database."""
        tag_feature: pd.DataFrame = pd.DataFrame.from_records(FE.compute_tag_feature())
        for _, row in tag_feature.iterrows():
            cls.feature.update_one(
                {'id': row['id']},
                {'$set': {
                    'id': row['id'],
                    f'softmax_dim{S.CHALLENGE_TAG_OHE_DIM}': row.reindex(
                        [f'tag_comb{i}_dim{S.CHALLENGE_TAG_OHE_DIM}_softmax_score' for i in range(1, 5)]).tolist(),
                    f'one_hot_dim{S.CHALLENGE_TAG_OHE_DIM}':
                        np.concatenate(
                            row.reindex([
                                f'tag_comb{i}_dim{S.CHALLENGE_TAG_OHE_DIM}_one_hot_array'
                                for i in range(1, 5)
                            ]).to_numpy()
                        ).tolist(),
                }},
                upsert=True,
            )

    @classmethod
    def write_docvec_feature(cls) -> None:
        """ Write the engineered challenge docvec into database."""
        challenge_desc_docvecs = FE.compute_challenge_desc_docvec()

        for cha_id, cha_docvec in challenge_desc_docvecs.items():
            cls.feature.update_one(
                {'id': cha_id},
                {'$set': {
                    'id': cha_id,
                    S.DV_FEATURE_NAME: cha_docvec
                }},
                upsert=True,
            )

    @classmethod
    def write_metadata_feature(cls) -> None:
        """ Write the engineered metadata feature into database."""
        challenge_metadata = FE.compute_challenge_metadata()
        for row in challenge_metadata.itertuples(index=False):
            cls.feature.update_one(
                {'id': row.id},
                {'$set': {
                    'id': row.id,
                    'metadata': [row.project_id_encode, row.legacy_sub_track_encode, row.duration],
                }},
                upsert=True,
            )

    @classmethod
    def write_competing_challenges(cls):
        """ Write the competing challenge id list and number of competing challenges into database.
            This method takes longer to run than it looks ;-)
        """
        for challenge in FE.compute_competing_challenges():
            cls.feature.update_one({'id': challenge['id']}, {'$set': challenge}, upsert=True)


if __name__ == '__main__':
    TopcoderMongo.write_training_feature()
