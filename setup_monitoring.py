# setup_monitoring.py
import os
from azure.ai.ml import Input
from azure.ai.ml.entities import (
    MonitorSchedule,
    MonitorDefinition,
    MonitoringTarget,
    DataDriftSignal,
    ReferenceData,
    ProductionData,
    ServerlessSparkCompute,
    AlertNotification,
    CronTrigger,
)
from config import get_ml_client

ml_client = get_ml_client()

# Define monitoring target
monitoring_target = MonitoringTarget(
    ml_task="classification",
    endpoint_deployment_id="azureml:loan-approval-endpoint:green",
)

# Reference data (training data)
reference_data = ReferenceData(
    input_data=Input(
        type="uri_file",
        path="azureml:loan-approval-dataset:1",
    ),
    data_context="training",
    target_column_name="Approved",
)

# Data drift signal
data_drift = DataDriftSignal(
    reference_data=reference_data,
    production_data=ProductionData(
        input_data=Input(
            type="uri_folder",
            path="azureml:loan-approval-endpoint:green",
        ),
        data_context="model_inputs",
    ),
    alert_enabled=True,
)

alert_email = os.environ.get('ALERT_EMAIL', 'your-email@domain.com')

# Create monitor schedule
monitor_definition = MonitorDefinition(
    compute=ServerlessSparkCompute(
        instance_type="Standard_E4s_v3",
        runtime_version="3.3",
    ),
    monitoring_target=monitoring_target,
    monitoring_signals={'data_drift': data_drift},
    alert_notification=AlertNotification(emails=[alert_email]),
)

monitor_schedule = MonitorSchedule(
    name="loan-approval-monitor",
    trigger=CronTrigger(expression="0 0 * * *"),  # Daily
    create_monitor=monitor_definition,
)

ml_client.schedules.begin_create_or_update(monitor_schedule).result()
print("Monitoring schedule created - runs daily.")
print(f"Alert email: {alert_email}")
