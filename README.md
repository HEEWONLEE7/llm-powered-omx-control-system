# 🧠 Natural Language-Based OMX Control System

## Project Overview

This project implements a natural language-based control system that allows the OMX robot to perform movements through intuitive text commands.

Using a rule-based architecture, the system analyzes user input and translates it into predefined motion sequences. This enables the robot to respond to human language in a structured and predictable way, reducing the need for traditional control interfaces such as joysticks or complex command panels.

The primary focus of this project is to enhance human-robot interaction by enabling users to control the robot in a more natural and accessible manner, while maintaining reliability and safety in movement execution.


## Demo Video

[![OMX Natural Language Control Demo](https://img.youtube.com/vi/JGMOZXorOjo/0.jpg)](https://youtu.be/JGMOZXorOjo?si=fsFUdSmBNe75S8lg)


## Implementation & UI

- ✅ OMX Control Code  
  👉 [Implementation Repository](https://github.com/HEEWONLEE7/llm-powered-omx-control-system)

- ✅ User Interface for OMX Control  
  👉 [UI Repository](https://github.com/HEEWONLEE7/llm-robot-commander)


## ROBOTIS OMX Related Resources

- ROBOTIS OMX Official Documentation  
  👉 https://ai.robotis.com/omx/introduction_omx.html

- ROBOTIS OpenMANIPULATOR GitHub Repository  
  👉 https://github.com/ROBOTIS-GIT/open_manipulator


## How to Run

To operate the system, both the robot control node and the UI must be launched separately.

### 1️⃣ Launch OMX Robot Control (ROS 2)

First, start the OMX robot control system by running the following command in the robot environment:

```bash
ros2 launch open_manipulator_bringup omx_f_control.launch.py
```
This initializes the OMX robot and prepares it to receive motion commands based on natural language input.

### 2️⃣ Launch Control UI
In another terminal, run the UI that handles natural language input and command transmission:
``` bash
python3 ws/llm-robot-commander/llm-robot-commander/control_ui.launch.py
```
This UI allows users to enter natural language commands, which are then processed and converted into executable motion instructions for the OMX robot.
