# Advanced Face Detection & Recognition System  

## Overview  
This project implements real-time face detection, recognition, multi-face tracking, and emotion analysis using Dlib, OpenCV, and Deep Learning (ResNet-34). It supports face position detection (left, center, right), face matching using Euclidean Distance, and smile detection using Haar Cascade.  

## Features  
- Face detection using CNN (Dlib) for high accuracy  
- Face recognition using Deep Metric Learning (ResNet-34)  
- Multi-face tracking using Dlib’s Correlation Tracker  
- Smile detection using Haar Cascade  
- Face position analysis to detect if a face is positioned left, center, or right  
- Face matching using Euclidean Distance between embeddings  
- Live webcam processing with real-time video streams  

## Technologies Used  
- Python  
- OpenCV, Dlib, Face Recognition, NumPy  
- Deep Learning model based on ResNet-34 for feature extraction  
- Algorithms including CNN, Haar Cascade, and Deep Metric Learning  

## Project Structure  
```
Face_Recognition_Project
 ├── faces/                 # Folder to store known face images
 ├── saved_faces/           # Folder to save detected faces
 ├── advanced.py            # Main project script (Face Detection & Recognition)
 ├── requirements.txt       # List of dependencies
 ├── README.md              # Project documentation
 ├── LICENSE                # MIT License
```

## Installation & Setup  

### Clone the Repository  
```
git clone https://github.com/yourusername/Face_Recognition_Project.git
cd Face_Recognition_Project
```

### Set Up Virtual Environment (Recommended)  
```
python -m venv face_env
source face_env/bin/activate  # For Mac/Linux
face_env\Scripts\activate     # For Windows
```

### Install Dependencies  
```
pip install -r requirements.txt
```

### Run the Program  
```
python advanced.py
```
- Press **Q** to exit  
- Press **S** to save a detected face  

## Example Output  
*(Add screenshots or example outputs here if needed.)*  

## Contributing  
Contributions are welcome. Fork the repository, submit pull requests, or open issues for improvements.  

## License  
This project is licensed under the MIT License. See the LICENSE file for details.  

```
MIT License

Copyright (c) 2025 Kamalesh

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
```