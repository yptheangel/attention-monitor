import React, { useState, useEffect } from 'react';
import { Line } from '@ant-design/charts';


const YawPicthRollLines = (props) => {

  // const userID = props.userID;
  // const url = 'http://13.229.155.101/user/'.concat(userID.toString())
  // console.log("Lines: " + userID)
  // console.log(url)
  // console.log('http://13.229.155.101/user/2')
  
  console.log("props: ")
  console.log(props)

    const [data, setData] = useState([]);
    useEffect(() => {
      asyncFetch();
    }, []);
    const asyncFetch = () => {
      fetch(url)
        // .then((response) => response.json())
        // .then((json) => setData(json))
        .then(res => res.text())          // convert to plain text
        .then(text => console.log(text))  // then log it out
        .catch((error) => {
          console.log('fetch data failed', error);
        });
    };

    const config = {
        title: {
          visible: true,
          text: 'Yaw Pitch Roll',
        },
        description: {
          visible: true,
          text: 'Change of Yaw, Pitch and Roll over time',
        },
        padding: 'auto',
        forceFit: true,
        data: data["ypr"],
        xField: 'date',
        yField: 'value',
        // yAxis: { label: { formatter: (v) => `${v}`.replace(/\d{1,3}(?=(\d{3})+$)/g, (s) => `${s},`) } },
        legend: { position: 'right-top' },
        seriesField: 'type',
        color: ['#1979C9', '#D62A0D', '#FAA219'],
        responsive: true,
      };

     return <Line {...config} />
        
};

export default YawPicthRollLines;



