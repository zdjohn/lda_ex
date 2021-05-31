from clustering_ex import utils
import pytest


@pytest.fixture()
def spark_resource(request):
    session, logger, _ = utils.start_spark()

    def teardown():
        logger.info("tearing down")
        session.stop()

    request.addfinalizer(teardown)

    return session
