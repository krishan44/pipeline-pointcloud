# MIT License
#
# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY

"""EC2 construct for Gaussian Splatting pipeline execution"""

from aws_cdk import (
    Environment,
    aws_ec2 as ec2,
    aws_iam as iam,
    CfnOutput,
    Tags
)
from constructs import Construct

class Ec2Compute(Construct):
    """Class for EC2 Compute Construct"""
    def __init__(
            self,
            scope: Construct,
            id: str,
            env: Environment,
            ecr_repo_name: str,
            s3_bucket_name: str,
            container_role_name: str,
            prefix: str,
            **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Use default VPC
        self.vpc = ec2.Vpc.from_lookup(self, "DefaultVPC", is_default=True)

        # Create security group
        self.security_group = ec2.SecurityGroup(
            self,
            "GaussianSplattingSecurityGroup",
            vpc=self.vpc,
            description="Security group for Gaussian Splatting EC2 instances",
            allow_all_outbound=True
        )

        # Create IAM role for EC2 instance
        self.instance_role = iam.Role(
            self,
            "EC2InstanceRole",
            role_name=f"{prefix}-ec2-instance-role",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly"),
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchLogsFullAccess")
            ]
        )

        # Add S3 permissions
        self.instance_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket",
                    "s3:DeleteObject"
                ],
                resources=[
                    f"arn:aws:s3:::{s3_bucket_name}",
                    f"arn:aws:s3:::{s3_bucket_name}/*"
                ]
            )
        )

        # Add ECR permissions
        self.instance_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage"
                ],
                resources=["*"]
            )
        )

        # Add DynamoDB permissions (for status updates)
        self.instance_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                    "dynamodb:UpdateItem"
                ],
                resources=[f"arn:aws:dynamodb:{env.region}:{env.account}:table/*"]
            )
        )

        # Add Lambda invoke permissions (for completion notification)
        self.instance_role.add_to_policy(
            iam.PolicyStatement(
                actions=["lambda:InvokeFunction"],
                resources=[f"arn:aws:lambda:{env.region}:{env.account}:function:*"]
            )
        )

        # Add SNS publish permissions
        self.instance_role.add_to_policy(
            iam.PolicyStatement(
                actions=["sns:Publish"],
                resources=[f"arn:aws:sns:{env.region}:{env.account}:*"]
            )
        )

        # Create instance profile
        self.instance_profile = iam.CfnInstanceProfile(
            self,
            "InstanceProfile",
            roles=[self.instance_role.role_name],
            instance_profile_name=f"{prefix}-ec2-instance-profile"
        )

        # Create launch template
        self.launch_template = ec2.CfnLaunchTemplate(
            self,
            "LaunchTemplate",
            launch_template_name=f"{prefix}-gaussian-splatting-lt",
            launch_template_data=ec2.CfnLaunchTemplate.LaunchTemplateDataProperty(
                image_id=self.get_deep_learning_ami(env.region),
                iam_instance_profile=ec2.CfnLaunchTemplate.IamInstanceProfileProperty(
                    arn=self.instance_profile.attr_arn
                ),
                security_group_ids=[self.security_group.security_group_id],
                block_device_mappings=[
                    ec2.CfnLaunchTemplate.BlockDeviceMappingProperty(
                        device_name="/dev/sda1",
                        ebs=ec2.CfnLaunchTemplate.EbsProperty(
                            volume_size=100,
                            volume_type="gp3",
                            delete_on_termination=True
                        )
                    )
                ],
                monitoring=ec2.CfnLaunchTemplate.MonitoringProperty(enabled=True),
                tag_specifications=[
                    ec2.CfnLaunchTemplate.TagSpecificationProperty(
                        resource_type="instance",
                        tags=[
                            {"key": "Name", "value": f"{prefix}-gaussian-splatting"},
                            {"key": "Application", "value": "GaussianSplatting"},
                            {"key": "ManagedBy", "value": "StepFunctions"}
                        ]
                    )
                ],
                instance_market_options=ec2.CfnLaunchTemplate.InstanceMarketOptionsProperty(
                    market_type="spot",
                    spot_options=ec2.CfnLaunchTemplate.SpotOptionsProperty(
                        spot_instance_type="one-time",
                        instance_interruption_behavior="terminate"
                    )
                ),
                metadata_options=ec2.CfnLaunchTemplate.MetadataOptionsProperty(
                    http_tokens="required",
                    http_put_response_hop_limit=1
                )
            )
        )

        # Outputs
        CfnOutput(self, "LaunchTemplateId", value=self.launch_template.ref)
        CfnOutput(self, "InstanceProfileArn", value=self.instance_profile.attr_arn)
        CfnOutput(self, "InstanceRoleName", value=self.instance_role.role_name)

    def get_deep_learning_ami(self, region: str) -> str:
        """Get the latest AWS Deep Learning AMI with Docker and NVIDIA drivers pre-installed"""
        # AWS Deep Learning Base AMI (Ubuntu 20.04) - has Docker, NVIDIA drivers, and CUDA
        # These are regional AMI IDs for the AWS Deep Learning AMI
        ami_map = {
            "us-east-1": "ami-0c7217cdde317cfec",
            "us-east-2": "ami-0b8b44ec9a8f90422", 
            "us-west-1": "ami-0d948363d39854d52",
            "us-west-2": "ami-0efcece6bed30fd98",
            "eu-west-1": "ami-0f3c5d1e8c9e28b5f",
            "eu-west-2": "ami-0dbec48abfe298cab",
            "eu-central-1": "ami-0e067cc8a2b58de59",
            "ap-southeast-1": "ami-0df7a207adb9748c7",
            "ap-southeast-2": "ami-0310483fb2b488153",
            "ap-northeast-1": "ami-0bba69335379e17f8",
            "ap-northeast-2": "ami-0e9bfdb247cc8de84",
        }
        return ami_map.get(region, ami_map["us-east-1"])
