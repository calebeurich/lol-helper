from dotenv import load_dotenv
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

import boto3, os, sagemaker

# Load environment variables and set up
load_dotenv()
REGION = os.getenv("REGION")
ROLE = os.getenv("ROLE")
INPUT_PATH = os.getenv("CHAMPIONS_RAW_DATA")
OUTPUT_PATH = os.getenv("PROCESSED_DATA_FOLDER")
DATA_MAPPING_PATH = os.getenv("DATA_MAPPING_FILE")

session = sagemaker.Session(boto3.Session(region_name=REGION))

spark_processor = PySparkProcessor(
    base_job_name             = "spark-champion-aggregation", # SageMaker job names need to be in kebab case
    framework_version         = "3.3",  
    role                      = ROLE,
    instance_count            = 2,
    instance_type             = "ml.m5.xlarge",
    max_runtime_in_seconds    = 1800,
    sagemaker_session         = session,
    env                       = {"AWS_DEFAULT_REGION" : REGION},
    tags                      = [{"Key" : "UseSpotInstances", "Value" : "True"}],
)

spark_config = [{
    "Classification": "spark-defaults",
    "Properties": {
        "spark.executor.memory": "12g",
        "spark.executor.cores": "4",
        "spark.driver.memory": "10g",
        "spark.executor.memoryOverhead": "2g",  # Explicit overhead
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.shuffle.partitions": "100",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.default.parallelism": "100"  # Match shuffle partitions
    }
}]

spark_processor.run(
    submit_app      = "spark_aggregation_script.py",
    submit_py_files = ["spark_champion_aggregation.py"],
    arguments       = [
        "--input-path", "/opt/ml/processing/input",
        "--output-path", "/opt/ml/processing/output",
        "--items-json-path", "/opt/ml/processing/config/item_id_tags.json"
    ],
    configuration=spark_config,
    inputs          = [
        ProcessingInput(
            source      = INPUT_PATH,
            destination = "/opt/ml/processing/input"
        ),
        ProcessingInput(
            source      = DATA_MAPPING_PATH,
            destination = "/opt/ml/processing/config"
    ),
    ],
    outputs         = [
        ProcessingOutput(
            source      = f"/opt/ml/processing/output",
            destination = OUTPUT_PATH
        )
    ]
)

print("Spark Processing Job submitted!")