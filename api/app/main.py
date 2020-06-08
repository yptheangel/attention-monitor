from flask import Flask
import json
import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal

dynamodb = boto3.resource('dynamodb', region_name='ap-southeast-1')
table = dynamodb.Table('test-table5')

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World from Flask"

@app.route('/user/<int:user_id>')
def get_user_data(user_id):
    response = table.query(
        KeyConditionExpression=Key('id').eq(str(user_id))
    )
    return json.dumps(response, default=default)


def default(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=80)