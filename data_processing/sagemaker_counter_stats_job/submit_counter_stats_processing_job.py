import os, boto3, sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("REGION", "us-east-2")
sess = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
ROLE = os.getenv("ROLE")

BUCKET = os.getenv("BUCKET") or sess.default_bucket()
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_FOLDER", "processed")
INPUT_PATH = os.getenv("CHAMPIONS_RAW_DATA")
out_s3_uri = f"s3://{BUCKET}/{PROCESSED_DATA_PATH}"

# Rrun from the folder that has the entry script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)                       # avoid Windows "c:" URL scheme issue
CODE_FILE = "pandas_counter_stats_script.py"   # now relative

processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=ROLE,
    instance_type="ml.m5.4xlarge",
    instance_count=1,
    base_job_name="counter-stats",
    sagemaker_session=sess,
)

processor.run(
    code=CODE_FILE,  # relative path only
    arguments=["--input-path","/opt/ml/processing/input/raw",
               "--output-path","/opt/ml/processing/output"],
    inputs=[ProcessingInput(source=INPUT_PATH, destination="/opt/ml/processing/input/raw", input_name="raw")],
    outputs=[ProcessingOutput(source="/opt/ml/processing/output", destination=out_s3_uri, output_name="counter_stats")],
    logs=True,
    wait=True,
)