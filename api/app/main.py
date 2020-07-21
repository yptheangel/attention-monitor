from flask import Flask
import json
import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal
import pandas as pd
from flask_cors import CORS, cross_origin

dynamodb = boto3.resource('dynamodb', region_name='ap-southeast-1')
table = dynamodb.Table('test-table5')

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def hello():
    return "Hello World from Flask"

@app.route('/user/<int:user_id>')
@cross_origin()
def get_user_data(user_id):
    response = table.query(
        KeyConditionExpression=Key('id').eq(str(user_id))
    )

    df = pd.DataFrame(response['Items'])

    df[['blink_count', 'id', 'lost_focus_count', 'yawn_count']] = df[
        ['blink_count', 'id', 'lost_focus_count', 'yawn_count']].astype(int)
    df[['ear', 'face_not_present_duration', 'lost_focus_duration', 'pitch', 'roll', 'yaw', 'mar', 'timestamp']] = df[
        ['ear', 'face_not_present_duration', 'lost_focus_duration', 'pitch', 'roll', 'yaw', 'mar', 'timestamp']].astype(float)

    df["datetime"] = pd.to_datetime(df['timestamp'], unit='s')
    df2 = df.set_index('timestamp').sort_index(ascending=True)
    df2 = df2.sort_values(by='datetime')

    df2['hour'] = df2['datetime'].apply(lambda x: x.hour)
    df2['minute'] = df2['datetime'].apply(lambda x: x.minute)
    df2['second'] = df2['datetime'].apply(lambda x: x.second)

    t1 = df2[['minute', 'hour', 'blink_count']].groupby(['hour', 'minute']).max().astype(int) - df2[['hour', 'minute', 'blink_count']].groupby(['hour', 'minute']).min().astype(int)

    t2 = df2[['minute', 'hour', 'yawn_count']].groupby(['hour', 'minute']).max().astype(int) - df2[['hour', 'minute', 'yawn_count']].groupby(['hour', 'minute']).min().astype(int)

    t3 = df2[['minute', 'hour', 'lost_focus_count']].groupby(['hour', 'minute']).max() - df2[['hour', 'minute', 'lost_focus_count']].groupby(['hour', 'minute']).min()

    t4 = (df2[['minute', 'hour', 'lost_focus_duration']].groupby(['hour', 'minute']).max() - df2[['hour', 'minute', 'lost_focus_duration']].groupby(['hour', 'minute']).min()).apply(round).astype(int)

    t5 = (df2[['minute', 'hour', 'face_not_present_duration']].groupby(['hour', 'minute']).max() - df2[['hour', 'minute', 'face_not_present_duration']].groupby(['hour', 'minute']).min()).apply(round).astype(int)

    out = t1.join(t2).join(t3).join(t4).join(t5)
    new_index = out.index.map(lambda x: str(pd.Timestamp("%s:%s" % (x[0], x[1]), unit='minute').time()))
    out.index = new_index

    blink_count_final = [{"date": i, "value": int(row["blink_count"])} for i, row in out.iterrows()]
    yawn_count_final = [{"date": i, "value": int(row["yawn_count"])} for i, row in out.iterrows()]
    lost_focus_count_final = [{"date": i, "value": int(row["lost_focus_count"])} for i, row in out.iterrows()]
    lost_focus_duration_final = [{"date": i, "value": int(row["lost_focus_duration"])} for i, row in out.iterrows()]
    face_not_present_duration_final = [{"date": i, "value": int(row["face_not_present_duration"])} for i, row in out.iterrows()]

    output = {
        'blink_count': blink_count_final,
        'yawn_count_final': yawn_count_final,
        'lost_focus_count_final': lost_focus_count_final,
        'lost_focus_duration_final': lost_focus_duration_final,
        'face_not_present_duration_final': face_not_present_duration_final
    }

    ypr = df2[['yaw', 'pitch', 'roll', 'datetime']]
    ypr.loc[:, 'datetime'] = ypr['datetime'].map(lambda x: str(x.time()))

    ypr_final = [{'date': row['datetime'], 'type': 'yaw', 'value': row['yaw']} for i, row in
                 ypr[['yaw', 'datetime']].iterrows()]

    ypr_final.extend({'date': row['datetime'], 'type': 'pitch', 'value': row['pitch']} for i, row in
                     ypr[['pitch', 'datetime']].iterrows())
    ypr_final.extend({'date': row['datetime'], 'type': 'roll', 'value': row['roll']} for i, row in
                     ypr[['roll', 'datetime']].iterrows())

    ypr_data = {"ypr": ypr_final}
    output.update(ypr_data)

    focus_ratio = (df2['lost_focus_duration'].diff(1) == 0).sum() / len(df2)
    lost_focus_ratio = 1 - focus_ratio
    face_present_ratio = (df2['face_not_present_duration'].diff(1) == 0).sum() / len(df2)
    face_absent_ratio = 1 - face_present_ratio

    output.update(
        {'focus_ratio': focus_ratio,
         'lost_focus_ratio': lost_focus_ratio,
         'face_present_ratio': face_present_ratio,
         'face_absent_ratio': face_absent_ratio}
    )
    return json.dumps(output, default=default)


def default(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=80)