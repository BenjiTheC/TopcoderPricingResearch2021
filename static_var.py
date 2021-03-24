""" Static variables"""
import os
import pathlib
from collections import namedtuple
from dotenv import load_dotenv
load_dotenv()

DATA_PATH = pathlib.Path('data')
MODEL_PATH = pathlib.Path('models')

MongoConfig = namedtuple('MongoConfig', ['host', 'port', 'username', 'password', 'database'])
MONGO_CONFIG = MongoConfig(
    host=os.getenv("MONGO_HOST"),
    port=os.getenv("MONGO_PORT") and int(os.getenv("MONGO_PORT")),
    username=os.getenv('MONGO_USER'),
    password=os.getenv('MONGO_PSWD'),
    database=os.getenv("MONGO_DATABASE"),
)

Doc2VecConfig = namedtuple('Doc2VecConfig', ['similarity', 'frequency', 'token_length', 'dimension'])
DOC2VEC_CONFIG = Doc2VecConfig(
    similarity=(os.getenv('CHALLENGE_DESC_SIMILARITY') and float(os.getenv('CHALLENGE_DESC_SIMILARITY'))) or None,
    frequency=(os.getenv('CHALLENGE_DESC_FREQUENCY') and float(os.getenv('CHALLENGE_DESC_FREQUENCY'))) or None,
    token_length=(os.getenv('CHALLENGE_DESC_TOKEN_LEN') and int(os.getenv('CHALLENGE_DESC_TOKEN_LEN'))) or 0,
    dimension=(os.getenv('DOCVEC_DIM') and int(os.getenv('DOCVEC_DIM'))),
)
DV_FEATURE_NAME = 'docvec_sim{similarity}freq{frequency}tkl{token_len}dim{dimension}'.format(
    similarity=DOC2VEC_CONFIG.similarity and int(DOC2VEC_CONFIG.similarity * 100),
    frequency=DOC2VEC_CONFIG.frequency and int(DOC2VEC_CONFIG.frequency * 100),
    token_len=DOC2VEC_CONFIG.token_length,
    dimension=DOC2VEC_CONFIG.dimension,
)
DV_MODEL_NAME = 'challenge_desc_docvecs_sim{similarity}freq{frequency}tkl{token_len}dim{dimension}'.format(
    similarity=DOC2VEC_CONFIG.similarity,
    frequency=DOC2VEC_CONFIG.frequency,
    token_len=DOC2VEC_CONFIG.token_length,
    dimension=DOC2VEC_CONFIG.dimension,
)
CHALLENGE_TAG_OHE_DIM = os.getenv('TAG_OHE_DIM') and int(os.getenv('TAG_OHE_DIM'))
CHALLENGE_TAG_COMB_TOP = CHALLENGE_TAG_OHE_DIM // 4

META_DATA_COLUMNS = ['project_id', 'sub_track', 'duration']
GLOBAL_CONTEXT_COLUMNS = [
    'num_of_competing_challenges',
    'competing_same_proj',
    'competing_same_sub_track',
    'competing_avg_overlapping_tags',
]
TAG_SOFTMAX_COLUMNS = [f'softmax_c{i + 1}' for i in range(4)]
TAG_OHE_COLUMNS = [f'ohe{i}' for i in range(CHALLENGE_TAG_OHE_DIM)]
DOCVEC_COLUMNS = [f'dv{i}' for i in range(DOC2VEC_CONFIG.dimension)]
FEATURE_MATRIX_COLUMNS = META_DATA_COLUMNS + GLOBAL_CONTEXT_COLUMNS + TAG_SOFTMAX_COLUMNS + TAG_OHE_COLUMNS
NUMERIC_FEATURES = ['duration'] + GLOBAL_CONTEXT_COLUMNS + TAG_SOFTMAX_COLUMNS
FEATURE_MATRIX_REINDEX = NUMERIC_FEATURES + ['project_id', 'sub_track'] + TAG_OHE_COLUMNS
FEATURE_MATRIX_COLUMNS_WITH_DV = NUMERIC_FEATURES + DOCVEC_COLUMNS + ['project_id', 'sub_track'] + TAG_OHE_COLUMNS

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
