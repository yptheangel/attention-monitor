import boto3
import json
from datetime import datetime
import time

client = boto3.client('kinesis', region_name='ap-southeast-1')
stream_name = "kinesis-attention-stream"




def main():
    response = client.describe_stream(StreamName=stream_name)

    my_shard_id = response['StreamDescription']['Shards'][0]['ShardId']

    shard_iterator = client.get_shard_iterator(StreamName=stream_name,
                                               ShardId=my_shard_id,
                                               ShardIteratorType='LATEST')

    my_shard_iterator = shard_iterator['ShardIterator']

    record_response = client.get_records(ShardIterator=my_shard_iterator,
                                         Limit=2)

    while 'NextShardIterator' in record_response:
        record_response = client.get_records(ShardIterator=record_response['NextShardIterator'],
                                                     Limit=2)

        print(record_response)

        # wait for 5 seconds
        time.sleep(0.1)

if __name__ == '__main__':
    main()