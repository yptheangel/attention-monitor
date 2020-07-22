import React, { Component } from 'react';
import YawPicthRollLines from './Lines';
import BlinkCountStepLine from './Blink';
import Facenotpresentduration from './Facenotpresentduration';
import Lossfocuscount from './Lossfocuscount';
import Lossfocusduration from './Lossfocusduration';
import Yawncount from './Yawncount';
import { Card, Row, Col } from 'antd';

class Analysis extends Component {
    render(){
        return (
            <div className="analysis">
            <Row  gutter= {24}>
                <Col span= {24}>
                    <Card title="Face Position Analytics">
                        <YawPicthRollLines></YawPicthRollLines>
                    </Card>
                </Col>
            </Row>
            <Row  gutter= {24}>
                <Col span= {24}>
                    <Card title="Blink">
                        <BlinkCountStepLine/>
                    </Card>
                </Col>
            </Row>
            <Row  gutter= {24}>
                <Col span= {24}>
                    <Card title="Duration of face not present">
                        < Facenotpresentduration/>
                    </Card>
                </Col>
            </Row>
            <Row  gutter= {24}>
                <Col span= {24}>
                    <Card title="Loss focus count">
                        <Lossfocuscount/>
                    </Card>
                </Col>
            </Row>
            <Row  gutter= {24}>
                <Col span= {24}>
                    <Card title="Loss focus duration">
                        <Lossfocusduration/>
                    </Card>
                </Col>
            </Row>
            <Row  gutter= {24}>
                <Col span= {24}>
                    <Card title="Yawn Count">
                        <Yawncount/>
                    </Card>
                </Col>
            </Row>
        </div>
        )
    }
}

export default Analysis;
