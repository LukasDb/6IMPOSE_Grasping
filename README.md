# 6IMPOSE - Grasping
Graphical Control Interface for robotic grasping using 6D pose estimation.
For more information, check [6IMPOSE](https://github.com/HP-CAO/6IMPOSE)

___
## Requirements
- Python 3.9
- FanucHardware Interface
- Connected Intel Realsense (L515 or D415)

## Setup
- symlink 6IMPOSE repository to networks/pvn

## Usage:
- Run GUI
```
python run_app.py
```
- Calibrate Eye-In-Hand (from GUI):
    - Generate Charuco Board and place it fixed on a table
    - Collect Data (either with 'Setup->Acquisition' or manually moving the robot and taking pictures)
    - Calibrate Camera
- Perform Grasping bin 'Bin Picking' Task (check starting poses in data/poses.py)    
- Generate Grasp Poses for Objects:
```
python generate_grasp_poses.py [-h] OBJ_ID
```
- Visualize Grasp Poses:
```
python visualize_grasp_poses.py [-h] [--all] [--no-all] OBJ_ID
```