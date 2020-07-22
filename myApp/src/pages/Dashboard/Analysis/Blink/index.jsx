import React, { useState, useEffect } from 'react';
import { StepLine } from '@ant-design/charts';

const BlinkCountStepLine = () => {

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

    const config = {
        title: {
          visible: true,
          text: 'Blink Count ',
        },
        description: {
          visible: true,
          text: 'Blink Count per Minute',
        },
        forceFit: true,
        data: data['blink_count'],
        padding: 'auto',
        xField: 'date',
        yField: 'value'
      };

     return <StepLine {...config} />
        
};

export default BlinkCountStepLine;
