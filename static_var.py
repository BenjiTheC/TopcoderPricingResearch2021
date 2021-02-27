""" Static variable"""
import os
import pathlib
from collections import namedtuple
from dotenv import load_dotenv
load_dotenv()

DATA_PATH = pathlib.Path('data')
MODEL_PATH = pathlib.Path('models')

# Hardcode the mongoDB info here, .env or other environmental variable should be used for remote database
MongoConfig = namedtuple('MongoConfig', ['host', 'port', 'username', 'password', 'database'])
MONGO_CONFIG = MongoConfig(
    host=os.getenv("MONGO_HOST"),
    port=os.getenv("MONGO_PORT") and int(os.getenv("MONGO_PORT")),
    username=os.getenv('MONGO_USER'),
    password=os.getenv('MONGO_PSWD'),
    database=os.getenv("MONGO_DATABASE"),
)

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
