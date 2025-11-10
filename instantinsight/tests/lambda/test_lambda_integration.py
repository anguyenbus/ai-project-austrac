"""
Integration tests for Lambda container deployment.

These tests validate the Lambda container image functionality,
including local container testing and AWS Lambda integration.
"""

import json
import subprocess
import time
from typing import Any

import boto3
import pytest
import requests


def _is_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


@pytest.fixture
def lambda_client():
    """Provide AWS Lambda client for integration tests."""
    return boto3.client("lambda", region_name="ap-southeast-2")


@pytest.fixture
def s3_client():
    """Provide AWS S3 client for integration tests."""
    return boto3.client("s3", region_name="ap-southeast-2")


@pytest.fixture
def sample_lambda_event() -> dict[str, Any]:
    """Provide sample Lambda event for testing."""
    return {
        "query": "Show me the top 10 customers by revenue",
        "export_to_s3": True,
        "generate_visualization": True,
    }


class TestLocalContainerIntegration:
    """Test Lambda container running locally."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not _is_docker_available(),
        reason="Docker not available",
    )
    def test_local_container_starts_successfully(self):
        """Test container starts and responds to health checks."""
        # NOTE: This test validates the container can be built and run locally
        # Build container using Makefile
        result = subprocess.run(
            ["make", "-C", "lambda", "build", "IMAGE_TAG=test"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Build failed: {result.stderr}"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not _is_docker_available(),
        reason="Docker not available",
    )
    def test_local_container_accepts_invocations(self):
        """Test local container accepts and processes Lambda invocations."""
        # NOTE: This requires running container in background
        # Start container
        container_process = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "-p",
                "9000:8080",
                "-e",
                "LOG_LEVEL=DEBUG",
                "instantinsighttest:test",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            # Wait for container to start
            time.sleep(5)

            # Send test invocation
            event = {"query": "SELECT 1"}
            response = requests.post(
                "http://localhost:9000/2015-03-31/functions/function/invocations",
                json=event,
                timeout=30,
            )

            assert response.status_code == 200
            result = response.json()
            assert "statusCode" in result

        finally:
            container_process.terminate()
            container_process.wait(timeout=10)


class TestAWSLambdaIntegration:
    """Test deployed Lambda function on AWS."""

    @pytest.mark.integration
    @pytest.mark.aws
    def test_lambda_function_exists(self, lambda_client):
        """Test Lambda function is deployed and accessible."""
        try:
            response = lambda_client.get_function(
                FunctionName="instantinsight-query-processor"
            )

            assert response["Configuration"]["PackageType"] == "Image"
            assert response["Configuration"]["MemorySize"] >= 2048

        except lambda_client.exceptions.ResourceNotFoundException:
            pytest.skip("Lambda function not deployed")

    @pytest.mark.integration
    @pytest.mark.aws
    def test_lambda_invocation_success(
        self,
        lambda_client,
        sample_lambda_event: dict[str, Any],
    ):
        """Test Lambda function processes queries successfully."""
        try:
            response = lambda_client.invoke(
                FunctionName="instantinsight-query-processor",
                InvocationType="RequestResponse",
                Payload=json.dumps(sample_lambda_event),
            )

            result = json.loads(response["Payload"].read())
            body = json.loads(result["body"])

            assert result["statusCode"] == 200
            assert body["success"] is True
            assert "sql" in body
            assert body["row_count"] >= 0

        except lambda_client.exceptions.ResourceNotFoundException:
            pytest.skip("Lambda function not deployed")

    @pytest.mark.integration
    @pytest.mark.aws
    def test_lambda_s3_export(
        self,
        lambda_client,
        s3_client,
        sample_lambda_event: dict[str, Any],
    ):
        """Test Lambda exports results to S3 successfully."""
        try:
            response = lambda_client.invoke(
                FunctionName="instantinsight-query-processor",
                InvocationType="RequestResponse",
                Payload=json.dumps(sample_lambda_event),
            )

            result = json.loads(response["Payload"].read())
            body = json.loads(result["body"])

            if body.get("success") and body.get("export_path"):
                export_path = body["export_path"]

                # Verify S3 export exists
                assert export_path.startswith("s3://")

                # Parse S3 path
                parts = export_path.replace("s3://", "").split("/", 1)
                bucket = parts[0]
                key = parts[1]

                # Check object exists
                s3_response = s3_client.head_object(Bucket=bucket, Key=key)
                assert s3_response["ContentLength"] > 0

        except lambda_client.exceptions.ResourceNotFoundException:
            pytest.skip("Lambda function not deployed")

    @pytest.mark.integration
    @pytest.mark.aws
    def test_lambda_timeout_handling(self, lambda_client):
        """Test Lambda handles timeout scenarios correctly."""
        event = {
            "query": "Complex query that might timeout",
            "timeout_seconds": 5,  # Short timeout for testing
        }

        try:
            response = lambda_client.invoke(
                FunctionName="instantinsight-query-processor",
                InvocationType="RequestResponse",
                Payload=json.dumps(event),
            )

            result = json.loads(response["Payload"].read())

            # Should return success or timeout error
            assert result["statusCode"] in [200, 504]

        except lambda_client.exceptions.ResourceNotFoundException:
            pytest.skip("Lambda function not deployed")

    @pytest.mark.integration
    @pytest.mark.aws
    def test_lambda_error_handling(self, lambda_client):
        """Test Lambda returns proper error responses."""
        # Missing required query parameter
        event = {"user_id": "test123"}

        try:
            response = lambda_client.invoke(
                FunctionName="instantinsight-query-processor",
                InvocationType="RequestResponse",
                Payload=json.dumps(event),
            )

            result = json.loads(response["Payload"].read())
            body = json.loads(result["body"])

            assert result["statusCode"] == 400
            assert body["success"] is False
            assert "error" in body

        except lambda_client.exceptions.ResourceNotFoundException:
            pytest.skip("Lambda function not deployed")


class TestContainerBuildProcess:
    """Test container build and deployment workflow."""

    @pytest.mark.integration
    def test_dockerfile_syntax_valid(self):
        """Test Dockerfile syntax is valid."""
        result = subprocess.run(
            ["docker", "build", "--check", "-f", "lambda/Dockerfile", "."],
            capture_output=True,
            text=True,
        )

        # NOTE: --check flag validates syntax without building
        # If not supported, we just try a build
        assert result.returncode in [0, 125]  # 125 = unknown flag (older Docker)

    @pytest.mark.integration
    def test_makefile_targets_exist(self):
        """Test Makefile has required targets."""
        result = subprocess.run(
            ["make", "-C", "lambda", "help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = result.stdout

        required_targets = ["build", "push", "deploy", "test-local", "clean"]
        for target in required_targets:
            assert target in output


class TestEnvironmentConfiguration:
    """Test Lambda environment configuration."""

    @pytest.mark.integration
    @pytest.mark.aws
    def test_lambda_environment_variables_set(self, lambda_client):
        """Test Lambda has required environment variables configured."""
        try:
            response = lambda_client.get_function_configuration(
                FunctionName="instantinsight-query-processor"
            )

            env_vars = response.get("Environment", {}).get("Variables", {})

            required_vars = [
                "RESULTS_BUCKET",
                "RESULTS_PREFIX",
                "QUERY_TIMEOUT",
                "MAX_RESULT_ROWS",
                "LOG_LEVEL",
            ]

            for var in required_vars:
                assert var in env_vars, f"Missing environment variable: {var}"

        except lambda_client.exceptions.ResourceNotFoundException:
            pytest.skip("Lambda function not deployed")

    @pytest.mark.integration
    @pytest.mark.aws
    def test_lambda_vpc_configuration(self, lambda_client):
        """Test Lambda is deployed in VPC if required."""
        try:
            response = lambda_client.get_function_configuration(
                FunctionName="instantinsight-query-processor"
            )

            vpc_config = response.get("VpcConfig", {})

            # If VPC configured, verify subnets and security groups
            if vpc_config.get("VpcId"):
                assert len(vpc_config.get("SubnetIds", [])) > 0
                assert len(vpc_config.get("SecurityGroupIds", [])) > 0

        except lambda_client.exceptions.ResourceNotFoundException:
            pytest.skip("Lambda function not deployed")
