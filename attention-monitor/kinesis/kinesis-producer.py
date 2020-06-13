import boto3
import json
import random
import calendar
from datetime import datetime
import time

client = boto3.client('kinesis', region_name='ap-southeast-1')
stream_name = "attention-stream"


def main():
    while(True):
        property_value = random.randint(40, 120)
        property_timestamp = calendar.timegm(datetime.utcnow().timetuple())
        thing_id = 'aa-bb'

        put_to_stream(thing_id, property_value, property_timestamp)

        # wait for 5 second
        time.sleep(5)



def put_to_stream(thing_id, property_value, property_timestamp):
    payload = {
                'prop': str(property_value),
                'timestamp': str(property_timestamp),
                'thing_id': thing_id
              }

    print(payload)

    put_response = client.put_record(
                        StreamName=stream_name,
                        Data=json.dumps(payload),
                        PartitionKey=thing_id)
    print(put_response)

if __name__ == '__main__':
    main()