import React, { useState, useEffect } from 'react';
import { Liquid } from '@ant-design/charts';


const FacePresentRatio = () => {

    const [data, setData] = useState([]);
    useEffect(() => {
      asyncFetch();
    }, []);
    const asyncFetch = () => {
      fetch('http://13.229.155.101/user/2')
        .then((response) => response.json())
        .then((json) => setData(json))
        .catch((error) => {
          console.log('fetch data failed', error);
        });
    };

    console.log(data)

    const face_present_ratio = data.face_present_ratio * 10000;

    const config = {
        title: {
          visible: true,
        //   text: '水波图',
        },
        description: {
          visible: true,
        //   text: '水波图 - 百分比显示',
        },
        min: 0,
        max: 10000,
        value: face_present_ratio,
        statistic: { formatter: (value) => ((100 * value) / 10000).toFixed(1) + '%' },
      };

     return <Liquid {...config} />;
        
};

export default FacePresentRatio;