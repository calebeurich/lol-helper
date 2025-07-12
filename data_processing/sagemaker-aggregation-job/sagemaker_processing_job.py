import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

BUCKET = 'lolhelper'
REGION = 'us-east-2'
ROLE = 'arn:aws:iam::550149510806:role/sagemaker-processing-role'

session = sagemaker.Session(boto3.Session(region_name=REGION))

image_uri = sagemaker.image_uris.retrieve(
    framework='pytorch',
    region=REGION,
    version='2.0',
    py_version='py310',
    image_scope='training',
    instance_type='ml.m5.large' 
)

script_processor = ScriptProcessor(
    image_uri=image_uri,  
    command=['python3'],
    instance_type='ml.m5.large',
    instance_count=1,
    base_job_name='pandas-processing-job',
    role=ROLE,
    max_runtime_in_seconds=3600,
    env={'AWS_DEFAULT_REGION': REGION},
    tags=[{'Key': 'UseSpotInstances', 'Value': 'True'}],
    sagemaker_session=session,
)

script_processor.run(
    code='pandas_aggregation_script.py',
    inputs=[
        ProcessingInput(
            source=f's3://{BUCKET}/raw_data/match_data/patch_15_6/',
            destination='/opt/ml/processing/input'
        ),
        ProcessingInput(
            source=f's3://{BUCKET}/dependencies/modules.zip',
            destination='/opt/ml/processing/code/modules.zip'
        )
    ],
    outputs=[
        ProcessingOutput(
            source='/opt/ml/processing/output',
            destination=f's3://{BUCKET}/processed_data/champion_data/patch_15_6/'
        )
    ],
    arguments=[
        '--input-path', '/opt/ml/processing/input',
        '--output-path', '/opt/ml/processing/output'
    ]
)

print('✅ SageMaker Processing job submitted!')