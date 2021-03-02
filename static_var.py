""" Static variables"""
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

Doc2VecConfig = namedtuple('Doc2VecConfig', ['similarity', 'frequency', 'token_length'])
DOC2VEC_CONFIG = Doc2VecConfig(
    similarity=(os.getenv('CHALLENGE_DESC_SIMILARITY') and float(os.getenv('CHALLENGE_DESC_SIMILARITY'))) or None,
    frequency=(os.getenv('CHALLENGE_DESC_FREQUENCY') and float(os.getenv('CHALLENGE_DESC_FREQUENCY'))) or None,
    token_length=(os.getenv('CHALLENGE_DESC_TOKEN_LEN') and int(os.getenv('CHALLENGE_DESC_TOKEN_LEN'))) or 0,
)
DV_FEATURE_NAME = 'docvec_sim{similarity}freq{frequency}tkl{token_len}'.format(
    similarity=DOC2VEC_CONFIG.similarity and int(DOC2VEC_CONFIG.similarity * 100),
    frequency=DOC2VEC_CONFIG.frequency and int(DOC2VEC_CONFIG.frequency * 100),
    token_len=DOC2VEC_CONFIG.token_length,
)
DV_MODEL_NAME = 'challenge_desc_docvecs_sim{similarity}freq{frequency}tkl{token_len}'.format(
    similarity=DOC2VEC_CONFIG.similarity,
    frequency=DOC2VEC_CONFIG.frequency,
    token_len=DOC2VEC_CONFIG.token_length,
)

FEATURE_MATRIX_COLUMNS = (
    ['project_id', 'sub_track', 'duration', 'num_of_competing_challenges'] +
    [f'softmax_c{i + 1}' for i in range(4)] + 
    [f'ohe{i}' for i in range(100)] + [f'dv{i}' for i in range(100)]
)
NUMERIC_FEATURES = ['duration', 'num_of_competing_challenges', *[f'softmax_c{i + 1}' for i in range(4)]]
FEATURE_MATRIX_REINDEX = (  # Move numeric feature columns to the front so the order is intact from ColumnsTransformer
    NUMERIC_FEATURES + ['project_id', 'sub_track'] +
    [f'ohe{i}' for i in range(100)] + [f'dv{i}' for i in range(100)]
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
