Spike-SNN Event Vision Kit Documentation
=======================================

Production-ready toolkit for event-camera object detection with spiking neural networks.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   tutorials/index
   api/index
   examples/index

Overview
--------

The Spike-SNN Event Vision Kit provides a complete framework for deploying 
spiking neural networks on event-based vision systems. This toolkit enables 
real-time, ultra-low-power vision processing for robotics and edge AI applications.

Key Features
------------

* **Event Camera Support**: Native integration with DVS128, DAVIS346, Prophesee sensors
* **Hardware Backends**: CUDA, Intel Loihi 2, BrainChip Akida acceleration  
* **ROS2 Integration**: Plug-and-play robotics deployment
* **Real-time Processing**: Sub-millisecond latency object detection

Quick Start
-----------

.. code-block:: bash

   pip install spike-snn-event-vision-kit
   
.. code-block:: python

   from spike_snn_event import EventSNN, DVSCamera
   from spike_snn_event.models import SpikingYOLO
   
   # Initialize event camera and model
   camera = DVSCamera(sensor_type="DVS128")
   model = SpikingYOLO.from_pretrained("yolo_v4_spiking_dvs")
   
   # Real-time detection
   for events in camera.stream():
       detections = model.detect(events)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`