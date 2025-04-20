# 3DPathGenerator

An app that opens a point cloud and allows the user to shift-click 4 points that are then used to draw a path in a quad, where each row has a specified distance.
Based on the Point Cloud Library.

## Description

Once the pointcloud is loaded, the user is prompted to select 4 points. These points are checked for orientation and the order is adjusted accordingly. The user is then prompted to enter the distance between rows.

Once the app is finished, the path is drawn on the point cloud.

Example (with normal vectors drawn on every 5th point) :
![image](https://github.com/user-attachments/assets/73d7236a-3614-4efe-9ae0-f52dc915edcc)

