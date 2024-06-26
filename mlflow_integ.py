import mlflow
import AAE_pytorch_v1
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("test_experiment")

with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(AAE_pytorch_v3.params)

    # Log the loss metric
    mlflow.log_metric("loss", AAE_pytorch_v3.g_loss)
    mlflow.log_metric("d_loss", AAE_pytorch_v3.d_loss)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Test")

    # Infer the model signature

    # Log the model
    model_info_gen = mlflow.sklearn.log_model(
        sk_model = AAE_pytorch_v3.encoder_generator,
        artifact_path="mlflow/gen",
        input_example=AAE_pytorch_v3.in_out_rs,
        registered_model_name="G_tracking",
    )
    model_info_disc = mlflow.sklearn.log_model(
        sk_model=AAE_pytorch_v3.discriminator,
        artifact_path="mlflow/discriminator",
        input_example=AAE_pytorch_v3.z_dim,
        registered_model_name="D_tracking",
    )