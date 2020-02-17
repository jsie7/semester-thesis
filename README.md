## Code of ETH Semester Project
This repository contains code I developed during my semester project at ETH. Since it contained some code which can't be published, some parts of the code will not work out of the box.

The repository contains a data pipeline runner, a videoprocessing module, and a turbidity classification module. These parts are detailed below.

### Code Overview

1. 'videoprocessing.py' defines a video processor class, which opens a video file, cuts the file into frames and links the frames to labels extracted from an annotation file (ELAN .eaf file). In addition, process operations which are performed on all the frames can be defined. The module also implements a caching functionality allowing to cache processed frames and load them from cache again.
1. 'run.py' implements a pipeline runner, which runs pipelines defined by YAML configuration files. In addition, it defines a data pipeline that gathers data produced in each pipeline step and passes it to the next step. The video processor class was used within such a data pipeline to produce the results of my project.
1. 'turbidityprediction.py' implements a turbidity detection algorithm based on a Multi-Layer Perceptron with user-defined features.
