import React, { Component } from 'react';
import { Card, Row, Col } from 'antd';
import Focusratio from './focusratio';
import FacePresentRatio from './facepresentratio';

class Monitor extends Component {
    render(){
        return (
            <div className="monitor">
            <Row  gutter= {24}>
                <Col span= {12}>
                    <Card title="Focus Level" style={{textAlign: "center" }}>
                       <Focusratio/>
                    </Card >
                </Col>
                <Col span= {12}>
                    <Card title="Face Present Level" style={{textAlign: "center" }}>
                        <FacePresentRatio />
                    </Card>
                </Col>
            </Row>
            </div>
        )
    }
}

export default Monitor;