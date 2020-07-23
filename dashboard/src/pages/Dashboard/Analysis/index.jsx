import React, { Component } from 'react';
import YawPicthRollLines from './Lines';
import BlinkCountStepLine from './Blink';
import Facenotpresentduration from './Facenotpresentduration';
import Lossfocuscount from './Lossfocuscount';
import Lossfocusduration from './Lossfocusduration';
import Yawncount from './Yawncount';
import { Card, Row, Col, Input, Button } from 'antd';

class Analysis extends Component {

    constructor(props) {
        super(props);
        this.state = {value: ''};
        this.handleSubmit = this.handleSubmit.bind(this);
    }
    
    handleSubmit = event => {
        event.preventDefault()
        this.setState({
            value: this.element.value
        })
    }

    componentDidMount(){
        this.setState({
            value: "2"
        })
    }
    
    
    render(){
        const {value} = this.state.value;
        console.log("state:");
        console.log(this.state.value);
        const url = 'http://13.229.155.101/user/'.concat(this.state.value.toString())
        return (
            <div className="analysis">
                <form onSubmit={this.handleSubmit}>
                    <label>
                    <input type="text" ref={el => this.element = el} />
                    </label>
                    <input type="submit" value="Submit" />
                </form>
            <Row  gutter= {24}>
                <Col span= {24}>
                    <Card title="Face Position Analytics">
                        <p>User ID: {this.state.value}</p>
                        <YawPicthRollLines api={url}></YawPicthRollLines>
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
