"""[summary]
"""
import argparse
import json
from dotenv import load_dotenv


from clustering_ex import utils, job

parser = argparse.ArgumentParser()
parser.add_argument("--k", help="clustering size k", type=int)
parser.add_argument("--iteration", help="max iteration number", type=int)
parser.add_argument("--seed", help="randomization seed value", type=int)


if __name__ == "__main__":
    load_dotenv()
    args = parser.parse_args()
    spark_session, logger, settings = utils.start_spark(**vars(args))
    logger.info(f'default settings: {json.dumps(settings)}')
    job.run(spark_session, settings)
