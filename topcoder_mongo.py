""" Retrieve data from mongoDB."""
import re
import typing
import pymongo
import pymongo.errors
import pymongo.database
import pymongo.collection
import static_var as S
import util as U
from pprint import pprint
from dateutil.parser import isoparse


MONGO_CLIENT = None


def connect() -> pymongo.database.Database:
    """ Connect to MongoDB."""
    global MONGO_CLIENT
    if MONGO_CLIENT is None:
        MONGO_CLIENT = pymongo.MongoClient(S.MONGO_CONFIG.host, S.MONGO_CONFIG.port)

    database = MONGO_CLIENT[S.MONGO_CONFIG.database]
    return database


def get_collection(collection_name: str) -> pymongo.collection.Collection:
    database = connect()
    return database.get_collection(collection_name)


class TopcoderMongo:
    """ Retrieve data from MongoDB."""
    challenge = get_collection('challenge')
    project = get_collection('project')

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
    def get_challenge_count(cls, granularity: str = 'year', fmt: str = '%Y', status_filter: list[str] = S.STATUS) -> list[dict]:
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
        """ Get project common section similarity."""
        query = [
            {'$project': {'id': True, 'section_similarity': True}},
            {'$unwind': '$section_similarity'},
            {'$replaceRoot': {'newRoot': {'$mergeObjects': [{'project_id': '$id'}, '$section_similarity']}}},
        ]

        return list(cls.project.aggregate(query))
