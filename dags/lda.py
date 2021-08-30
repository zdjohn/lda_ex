from datetime import timedelta
from datetime import datetime

from airflow.models.dag import dag
from airflow.providers.amazon.aws.operators.emr_add_steps import EmrAddStepsOperator
from airflow.providers.amazon.aws.operators.emr_create_job_flow import EmrCreateJobFlowOperator
from airflow.providers.amazon.aws.operators.emr_terminate_job_flow import EmrTerminateJobFlowOperator
from airflow.providers.amazon.aws.sensors.emr_step import EmrStepSensor

BASE_S3 = 's3://zip-ex'
ETL_ARTIFACTS_DIST = f'{BASE_S3}/artifact'
ETL_ARTIFACTS_SETTINGS = f'{ETL_ARTIFACTS_DIST}/settings.json'
ETL_ARTIFACTS_MAIN = f'{ETL_ARTIFACTS_DIST}/main.py'
ETL_ARTIFACTS_BOOTSTRAP = f'{ETL_ARTIFACTS_DIST}/bootstrap.sh'

EMR_INSTANCE_GROUPS = [
    {
        'Name': "Master Instance Group",
        'EbsConfiguration': {
            'EbsBlockDeviceConfigs': [
                            {
                                'VolumeSpecification': {
                                    'SizeInGB': 32,
                                    'VolumeType': 'gp2'
                                },
                                'VolumesPerInstance': 2
                            }
            ]
        },
        'InstanceRole': 'MASTER',
        'InstanceType': 'c5.xlarge',
        'InstanceCount': 1,
    },
    {
        'Name': "Core Instance Group",
        'EbsConfiguration': {
            'EbsBlockDeviceConfigs': [
                            {
                                'VolumeSpecification': {
                                    'SizeInGB': 32,
                                    'VolumeType': 'gp2'
                                },
                                'VolumesPerInstance': 2
                            }
            ]
        },
        'InstanceRole': 'CORE',
        'InstanceType': 'c5.xlarge',
        'InstanceCount': 2,
    }
]

EMR_BOOTSTRAP_ACTIONS = [
    {
        'Name': 'Install deps',
        'ScriptBootstrapAction': {
                'Path': ETL_ARTIFACTS_BOOTSTRAP
        }
    },
]

JOB_FLOW_OVERRIDES = {
    'Name': 'PiCalc',
    'ReleaseLabel': 'emr-5.29.0',
    'Instances': {
        'InstanceGroups': EMR_INSTANCE_GROUPS,
        'KeepJobFlowAliveWhenNoSteps': True,
        'TerminationProtected': False,
    },
    'BootstrapActions': EMR_BOOTSTRAP_ACTIONS,
    'JobFlowRole': 'EMR_EC2_DefaultRole',
    'ServiceRole': 'EMR_DefaultRole',
}


DEFAULT_ARGS = {
    'owner': 'zdjohn',
    'depends_on_past': False,
    "start_date": datetime(2021, 6, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

METRIC_ETL_DAG_ID = 'metrics_etl_emr'


@dag(dag_id=METRIC_ETL_DAG_ID, default_args=DEFAULT_ARGS, tags=['lda'])
def metrics_etl_emr():

    def _metrics_job_steps(k, iteration, seed):
        """
        create spark job (step) based on hyper parameters

        Args:
            k ([type]): number of K
            iteration ([type]): max iteration
            seed ([type]): random seed

        Returns:
            [type]: return job steps
        """
        return [
            {
                'Name': 'lda_topics_classifier',
                'ActionOnFailure': 'TERMINATE_CLUSTER',
                'HadoopJarStep': {
                    'Properties': [],
                    'Jar': 'command-runner.jar',
                    'Args': [
                        'spark-submit',
                        '--deploy-mode',
                        'cluster',
                        '--py-files',
                        ETL_ARTIFACTS_DIST,
                        '--master',
                        'yarn',
                        '--files',
                        ETL_ARTIFACTS_SETTINGS,
                        '--packages',
                        'com.amazonaws:aws-java-sdk:1.11.900,org.apache.hadoop:hadoop-aws:3.2.0',
                        ETL_ARTIFACTS_MAIN,
                        f'--k={k}',
                        f'--iteration={iteration}',
                        f'--seed={seed}'
                    ]
                }
            }
        ]

    k = "{{dag_run.conf['k'] or '5'}}"
    iteration = "{{dag_run.conf['iteration'] or '50'}}"
    seed = "{{dag_run.conf['seed'] or '0'}}"

    cluster_creator = EmrCreateJobFlowOperator(
        task_id='spin_up_emr_cluster',
        job_flow_overrides=JOB_FLOW_OVERRIDES,
        aws_conn_id='aws_default',
        emr_conn_id='emr_default',
    )

    etl_step_add = EmrAddStepsOperator(
        task_id=f'etl_step_add',
        job_flow_id=cluster_creator.output,
        aws_conn_id='aws_default',
        steps=_metrics_job_steps(k, iteration, seed),
    )

    emr_step_listener = EmrStepSensor(
        task_id=f'emr_step_listener',
        job_flow_id=cluster_creator.output,
        step_id=etl_step_add.output[0],
        aws_conn_id='aws_default',
    )

    cluster_terminator = EmrTerminateJobFlowOperator(
        task_id='remove_cluster',
        job_flow_id=cluster_creator.output,
        aws_conn_id='aws_default',
    )

    cluster_creator >> etl_step_add >> emr_step_listener >> cluster_terminator


dag = metrics_etl_emr()
