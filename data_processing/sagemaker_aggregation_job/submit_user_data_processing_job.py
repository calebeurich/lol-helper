from dotenv import load_dotenv
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

import boto3, os, sagemaker

# Load environment variables and set up
load_dotenv()
REGION = os.getenv("REGION")
ROLE = os.getenv("ROLE") # ARN role, unrelated to team position
INPUT_PATH = os.getenv("SINGLE_USER_RAW_DATA")

# Paths to files in the same folder as this module
submit_app = "data_processing/sagemaker_aggregation_job/single_user_spark_aggregation_script.py"
submit_py_files = ["data_processing/sagemaker_aggregation_job/spark_champion_aggregation.py"]

def submit_user_processing_job(user_name: str, user_tag_line:str, user_queue_type:str) -> str:
    output_folder_path = os.getenv("PROCESSED_DATA_FOLDER")
    data_mapping_path = os.getenv("DATA_MAPPING_FILE")
    
    session = sagemaker.Session(boto3.Session(region_name=REGION))

    spark_processor = PySparkProcessor(
        base_job_name             = "spark-champion-aggregation", # SageMaker job names need to be in kebab case
        framework_version         = "3.3",  
        role                      = ROLE,
        instance_count            = 1,
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
        submit_app      = submit_app,
        submit_py_files = submit_py_files,
        arguments       = [
            "--user_name", user_name,
            "--user_tag_line", user_tag_line,
            "--user_queue_type", user_queue_type,
            "--output-path", "/opt/ml/processing/output",
            "--items-json-path", "/opt/ml/processing/config/item_id_tags.json"
        ],
        configuration=spark_config,
        inputs          = [
            ProcessingInput(
                source      = data_mapping_path,
                destination = "/opt/ml/processing/config"
        ),
        ],
        outputs         = [
            ProcessingOutput(
                source      = f"/opt/ml/processing/output",
                destination = output_folder_path
            )
        ],
        wait=True,
        logs=True,
    )

print("Spark Processing Job submitted!")