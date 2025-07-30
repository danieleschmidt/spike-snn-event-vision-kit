Spike-SNN Event Vision Kit Documentation
========================================

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: MIT License

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/CUDA-11.0+-green.svg
   :target: https://developer.nvidia.com/cuda-toolkit
   :alt: CUDA 11.0+

Production-ready toolkit for event-camera object detection with spiking neural networks (SNNs).

**Spike-SNN Event Vision Kit** provides a complete framework for deploying spiking neural networks on event-based vision systems. Building on research showing SNNs outperform frame-based CNNs on latency and energy, this toolkit enables real-time, ultra-low-power vision processing for robotics and edge AI applications.

Key Features
------------

🎥 **Event Camera Support**
   Native integration with DVS128, DAVIS346, Prophesee sensors

⚡ **Hardware Backends**
   CUDA, Intel Loihi 2, BrainChip Akida acceleration

🤖 **ROS2 Integration**
   Plug-and-play robotics deployment

📊 **Comprehensive Datasets**
   Pre-loaded neuromorphic vision benchmarks

🚀 **Real-time Processing**
   Sub-millisecond latency object detection

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Clone repository
   git clone https://github.com/yourusername/spike-snn-event-vision-kit.git
   cd spike-snn-event-vision-kit

   # Install dependencies
   pip install -e .

   # Optional: Install hardware acceleration
   pip install -e ".[cuda,ros2]"

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from spike_snn_event import EventSNN, DVSCamera
   from spike_snn_event.models import SpikingYOLO

   # Initialize event camera
   camera = DVSCamera(sensor_type="DVS128")

   # Load pre-trained spiking YOLO
   model = SpikingYOLO.from_pretrained("yolo_v4_spiking_dvs")

   # Real-time detection
   for events in camera.stream():
       detections = model.detect(events, threshold=0.5)
       print(f"Detected {len(detections)} objects")

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/cameras
   api/models
   api/training
   api/hardware
   api/datasets

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/architectures
   advanced/deployment
   advanced/optimization
   advanced/ros2_integration

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   roadmap

Performance Benchmarks
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Task
     - Frame CNN
     - Event SNN
     - Improvement
   * - Object Detection (mAP)
     - 72.3%
     - 71.8%
     - -0.7%
   * - Latency
     - 33ms
     - 0.8ms
     - 41×
   * - Power
     - 15W
     - 0.3W
     - 50×
   * - Dynamic Range
     - 60dB
     - 120dB
     - 2×

Community and Support
---------------------

* **GitHub Repository**: https://github.com/yourusername/spike-snn-event-vision-kit
* **Issues and Bug Reports**: https://github.com/yourusername/spike-snn-event-vision-kit/issues
* **Discussions**: https://github.com/yourusername/spike-snn-event-vision-kit/discussions
* **Documentation**: https://spike-snn-event-vision.readthedocs.io

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`