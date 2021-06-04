"""[summary]
"""
import argparse
import json
from dotenv import load_dotenv


from clustering_ex import utils
from clustering_ex.lda import job as job_lda
from clustering_ex.node2vec import job as job_node2vec

parser = argparse.ArgumentParser()
parser.add_argument("--job", help="pick etl job", type=str)
parser.add_argument("--k", help="clustering size k", type=int)
parser.add_argument("--iteration", help="max iteration number", type=int)
parser.add_argument("--seed", help="randomization seed value", type=int)

JOBS = {
    "lda": job_lda,
    "node2vec": job_node2vec
}

if __name__ == "__main__":
    load_dotenv()
    args = parser.parse_args()
    spark_session, logger, settings = utils.start_spark(**vars(args))
    logger.info(f'default settings: {json.dumps(settings)}')
    job = JOBS.get(args.job, "lda")
    job.run(spark_session, settings)
