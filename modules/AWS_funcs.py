
"""
Filename:    AWS_funcs.py
Author:      Deanna Nash, dnash@ucsd.edu
Description: Functions to deal with AWS bucket data
"""

import boto3
import botocore

def get_s3_keys(bucket, prefix = ''):
    """
    Generate the keys in an S3 bucket.
    
    This code is from Hamed Alemohammad:
    https://github.com/HamedAlemo/visualize-goes16
    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    """
    s3 = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))
    kwargs = {'Bucket': bucket}

    if isinstance(prefix, str):
        kwargs['Prefix'] = prefix

    while True:
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            key = obj['Key']
            if key.startswith(prefix):
                yield key

        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break