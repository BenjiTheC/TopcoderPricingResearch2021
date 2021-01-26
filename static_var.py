""" Static variable"""
import pathlib
from collections import namedtuple

DATA_PATH = pathlib.Path('data')

# Hardcode the mongoDB info here, .env or other environmental variable should be used for remote database
MongoConfig = namedtuple('MongoConfig', ['host', 'port', 'database'])
MONGO_CONFIG = MongoConfig('127.0.0.1', 27017, 'topcoder')

# Some meta data from topcoder.com, manually written here because it's pretty short
DETAILED_STATUS = [
    'New',
    'Draft',
    'Cancelled',
    'Active',
    'Completed',
    'Deleted',
    'Cancelled - Failed Review',
    'Cancelled - Failed Screening',
    'Cancelled - Zero Submissions',
    'Cancelled - Winner Unresponsive',
    'Cancelled - Client Request',
    'Cancelled - Requirements Infeasible',
    'Cancelled - Zero Registrations',
]
STATUS = ['Completed', 'Cancelled', 'Deleted', 'Draft', 'New', 'Active']
TRACK = ['Data Science', 'Design', 'Development', 'Quality Assurance']
TYPE = ['Challenge', 'First2Finish', 'Task']
