#!/usr/bin/env python3
"""
Advanced scaling and optimization examples for Spike SNN Event Vision Kit.

This script demonstrates the advanced capabilities including:
- Auto-scaling configuration
- Load balancing
- Concurrent processing
- Performance optimization
- Resource monitoring
"""

import asyncio
import logging
import time
import numpy as np
from typing import List, Dict, Any

# Import the spike-snn-event toolkit
import spike_snn_event as snn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_concurrent_processing():
    """Example 1: Concurrent processing with resource pooling."""
    logger.info("=== Example 1: Concurrent Processing ===")
    
    try:
        # Get the global concurrent processor
        processor = snn.get_concurrent_processor()
        
        # Define a sample processing function
        def process_events(events: np.ndarray) -> Dict[str, Any]:
            """Simulate event processing."""
            time.sleep(0.1)  # Simulate processing time
            return {
                'event_count': len(events),
                'mean_timestamp': np.mean(events[:, 2]) if len(events) > 0 else 0,
                'polarity_ratio': np.mean(events[:, 3] > 0) if len(events) > 0 else 0.5
            }
        
        # Generate sample event batches
        event_batches = []
        for i in range(10):
            events = np.random.rand(100, 4)
            events[:, 0] *= 128  # x
            events[:, 1] *= 128  # y  
            events[:, 2] = np.sort(np.random.rand(100) * 0.01)  # timestamps
            events[:, 3] = np.random.choice([-1, 1], 100)  # polarity
            event_batches.append(events)
        
        logger.info(f"Created {len(event_batches)} event batches for processing")
        
        # Submit tasks for concurrent processing
        task_ids = []
        start_time = time.time()
        
        for i, events in enumerate(event_batches):
            task_id = f"process_batch_{i}"
            processor.submit_task(
                task_id=task_id,
                func=process_events,
                events,
                execution_mode="thread",
                priority=snn.concurrency.TaskPriority.NORMAL
            )
            task_ids.append(task_id)
        
        logger.info(f"Submitted {len(task_ids)} tasks for processing")
        
        # Wait for all tasks to complete
        results = processor.wait_for_completion(task_ids, timeout=30.0)
        
        processing_time = time.time() - start_time
        logger.info(f"All tasks completed in {processing_time:.2f} seconds")
        
        # Display results
        for task_id, result in results.items():
            if result.success:
                logger.info(f"{task_id}: {result.result}")
            else:
                logger.error(f"{task_id} failed: {result.error}")
        
        # Get processor statistics
        stats = processor.get_stats()
        logger.info(f"Processor stats: {stats}")
        
    except ImportError:
        logger.warning("Advanced concurrency features not available")
    except Exception as e:
        logger.error(f"Concurrent processing example failed: {e}")
    
    logger.info("Concurrent processing example completed\n")


def example_2_model_pooling():
    """Example 2: Model pooling for high-throughput inference."""
    logger.info("=== Example 2: Model Pooling ===")
    
    try:
        # Define a simple model factory
        def create_model():
            """Factory function to create SNN models."""
            return snn.CustomSNN(
                input_size=(64, 64),
                hidden_channels=[32, 64],
                output_classes=2
            )
        
        # Create model pool
        model_pool = snn.ModelPool(
            model_factory=create_model,
            pool_size=4
        )
        
        logger.info(f"Created model pool with {model_pool.size()} models")
        logger.info(f"Available models: {model_pool.available()}")
        
        # Simulate inference requests
        def run_inference(model, events: np.ndarray) -> np.ndarray:
            """Simulate model inference."""
            time.sleep(0.1)  # Simulate inference time
            return np.random.rand(2)  # Fake prediction
        
        # Generate test events
        test_events = []
        for _ in range(10):
            events = np.random.rand(50, 4)
            events[:, 0] *= 64
            events[:, 1] *= 64
            events[:, 2] = np.sort(np.random.rand(50) * 0.01)
            events[:, 3] = np.random.choice([-1, 1], 50)
            test_events.append(events)
        
        # Run inference with model pooling
        results = []
        start_time = time.time()
        
        for i, events in enumerate(test_events):
            # Use context manager for automatic model acquisition/release
            with model_pool.get_model(timeout=5.0) as model:
                logger.info(f"Processing batch {i} with pooled model")
                result = run_inference(model, events)
                results.append(result)
        
        inference_time = time.time() - start_time
        logger.info(f"Completed {len(results)} inferences in {inference_time:.2f} seconds")
        logger.info(f"Average inference time: {inference_time/len(results):.3f} seconds")
        
        # Final pool statistics
        logger.info(f"Final pool status - Size: {model_pool.size()}, Available: {model_pool.available()}")
        
    except ImportError:
        logger.warning("Model pooling features not available")
    except Exception as e:
        logger.error(f"Model pooling example failed: {e}")
    
    logger.info("Model pooling example completed\n")


def example_3_auto_scaling():
    """Example 3: Auto-scaling configuration and monitoring."""
    logger.info("=== Example 3: Auto-scaling ===")
    
    try:
        # Create custom scaling policy
        scaling_policy = snn.scaling.ScalingPolicy(
            cpu_scale_up_threshold=70.0,
            cpu_scale_down_threshold=30.0,
            memory_scale_up_threshold=80.0,
            memory_scale_down_threshold=40.0,
            min_instances=2,
            max_instances=10,
            scale_up_cooldown=60.0,  # 1 minute
            scale_down_cooldown=120.0  # 2 minutes
        )
        
        logger.info("Created custom scaling policy")
        
        # Initialize auto-scaler
        auto_scaler = snn.AutoScaler(policy=scaling_policy)
        
        # Start monitoring (this would run in background in production)
        logger.info("Starting auto-scaler monitoring...")
        auto_scaler.start()
        
        # Simulate running for a short time
        time.sleep(5.0)
        
        # Get scaling status
        status = auto_scaler.get_scaling_status()
        logger.info(f"Auto-scaler status: {status}")
        
        # Stop auto-scaler
        auto_scaler.stop()
        logger.info("Auto-scaler stopped")
        
    except ImportError:
        logger.warning("Auto-scaling features not available")
    except Exception as e:
        logger.error(f"Auto-scaling example failed: {e}")
    
    logger.info("Auto-scaling example completed\n")


def example_4_load_balancing():
    """Example 4: Load balancing configuration."""
    logger.info("=== Example 4: Load Balancing ===")
    
    try:
        # Create load balancer configuration
        lb_config = snn.scaling.LoadBalancerConfig(
            algorithm="weighted_response_time",
            health_check_interval=10.0,
            unhealthy_threshold=3,
            healthy_threshold=2
        )
        
        # Initialize load balancer
        load_balancer = snn.LoadBalancer(config=lb_config)
        
        # Add mock instances
        instances = [
            ("instance_1", "http://192.168.1.10:8000", 1.0),
            ("instance_2", "http://192.168.1.11:8000", 1.5),
            ("instance_3", "http://192.168.1.12:8000", 2.0),
        ]
        
        for instance_id, endpoint, weight in instances:
            load_balancer.add_instance(instance_id, endpoint, weight)
            logger.info(f"Added instance {instance_id} with weight {weight}")
        
        # Simulate request routing
        logger.info("Simulating request routing...")
        
        for i in range(20):
            # Get next instance for request
            instance_id = load_balancer.get_next_instance()
            
            if instance_id:
                # Simulate request processing
                load_balancer.record_request_start(instance_id)
                
                # Simulate variable response times
                response_time = np.random.uniform(50, 200)  # 50-200ms
                time.sleep(response_time / 1000)  # Convert to seconds
                
                load_balancer.record_request_end(instance_id, response_time)
                
                logger.info(f"Request {i} routed to {instance_id} (response: {response_time:.1f}ms)")
        
        # Get load balancer status
        status = load_balancer.get_status()
        logger.info(f"Load balancer status: {status}")
        
    except ImportError:
        logger.warning("Load balancing features not available")
    except Exception as e:
        logger.error(f"Load balancing example failed: {e}")
    
    logger.info("Load balancing example completed\n")


def example_5_scaling_orchestrator():
    """Example 5: Complete scaling orchestrator."""
    logger.info("=== Example 5: Scaling Orchestrator ===")
    
    try:
        # Create orchestrator with custom configurations
        scaling_policy = snn.scaling.ScalingPolicy(
            min_instances=1,
            max_instances=5,
            cpu_scale_up_threshold=60.0,
            queue_scale_up_threshold=20
        )
        
        lb_config = snn.scaling.LoadBalancerConfig(
            algorithm="least_connections",
            health_check_interval=5.0
        )
        
        # Initialize orchestrator
        orchestrator = snn.scaling.ScalingOrchestrator(
            scaling_policy=scaling_policy,
            lb_config=lb_config
        )
        
        logger.info("Created scaling orchestrator")
        
        # Start orchestrator
        orchestrator.start()
        
        # Add initial instances
        orchestrator.add_instance("initial_instance", "http://localhost:8000", 1.0)
        
        # Simulate some activity
        time.sleep(3.0)
        
        # Route some requests
        for i in range(5):
            instance = orchestrator.get_next_instance()
            if instance:
                logger.info(f"Request {i} would be routed to {instance}")
        
        # Get comprehensive status
        status = orchestrator.get_status()
        logger.info(f"Orchestrator status:")
        logger.info(f"  Auto-scaler: {status['auto_scaler']['current_instances']} instances")
        logger.info(f"  Load balancer: {status['load_balancer']['total_instances']} total instances")
        
        # Stop orchestrator
        orchestrator.stop()
        logger.info("Orchestrator stopped")
        
    except ImportError:
        logger.warning("Scaling orchestrator features not available")
    except Exception as e:
        logger.error(f"Scaling orchestrator example failed: {e}")
    
    logger.info("Scaling orchestrator example completed\n")


async def example_6_async_processing():
    """Example 6: Asynchronous event processing."""
    logger.info("=== Example 6: Async Processing ===")
    
    try:
        # Get async processor
        async_processor = snn.AsyncProcessor(max_concurrent=10)
        
        # Define async processing function
        async def async_process_events(events: np.ndarray) -> Dict[str, Any]:
            """Simulate async event processing."""
            await asyncio.sleep(0.1)  # Simulate async I/O
            return {
                'processed_events': len(events),
                'processing_time': 0.1
            }
        
        # Generate event batches
        event_batches = []
        for i in range(15):
            events = np.random.rand(50, 4)
            events[:, 0] *= 128
            events[:, 1] *= 128
            events[:, 2] = np.sort(np.random.rand(50) * 0.01)
            events[:, 3] = np.random.choice([-1, 1], 50)
            event_batches.append(events)
        
        # Submit async tasks
        task_ids = []
        for i, events in enumerate(event_batches):
            task_id = await async_processor.submit_async_task(
                f"async_task_{i}",
                async_process_events,
                events
            )
            task_ids.append(task_id)
        
        logger.info(f"Submitted {len(task_ids)} async tasks")
        
        # Wait for all tasks to complete
        results = await async_processor.wait_for_all(task_ids)
        
        logger.info(f"Completed {len(results)} async tasks")
        for task_id, result in results.items():
            if not isinstance(result, Exception):
                logger.info(f"{task_id}: {result}")
            else:
                logger.error(f"{task_id} failed: {result}")
        
        # Get pending task count
        pending = async_processor.get_pending_count()
        logger.info(f"Pending tasks: {pending}")
        
    except ImportError:
        logger.warning("Async processing features not available")
    except Exception as e:
        logger.error(f"Async processing example failed: {e}")
    
    logger.info("Async processing example completed\n")


def example_7_event_stream_processing():
    """Example 7: Continuous event stream processing."""
    logger.info("=== Example 7: Event Stream Processing ===")
    
    try:
        # Create event stream processor
        stream_processor = snn.EventStreamProcessor(
            buffer_size=1000,
            batch_size=50,
            processing_interval=0.01
        )
        
        # Define processing function
        def process_event_batch(event_batches: List[np.ndarray]) -> None:
            """Process a batch of event arrays."""
            total_events = sum(len(batch) for batch in event_batches)
            logger.info(f"Processed batch with {total_events} events")
        
        # Start stream processing
        stream_processor.start_processing(process_event_batch)
        
        # Simulate event stream
        logger.info("Simulating event stream...")
        for i in range(10):
            # Generate events
            events = np.random.rand(np.random.randint(20, 100), 4)
            events[:, 0] *= 128
            events[:, 1] *= 128
            events[:, 2] = np.sort(np.random.rand(len(events)) * 0.01)
            events[:, 3] = np.random.choice([-1, 1], len(events))
            
            # Add to stream processor
            stream_processor.add_events(events)
            
            # Small delay between batches
            time.sleep(0.05)
        
        # Let processing complete
        time.sleep(0.5)
        
        # Get processing statistics
        stats = stream_processor.get_stats()
        logger.info(f"Stream processing stats: {stats}")
        
        # Stop processing
        stream_processor.stop_processing()
        
    except ImportError:
        logger.warning("Event stream processing features not available")
    except Exception as e:
        logger.error(f"Event stream processing example failed: {e}")
    
    logger.info("Event stream processing example completed\n")


def main():
    """Run all advanced scaling examples."""
    logger.info("Starting Advanced Scaling Examples")
    logger.info("=" * 50)
    
    # Check if advanced features are available
    if not hasattr(snn, 'ADVANCED_FEATURES_AVAILABLE') or not snn.ADVANCED_FEATURES_AVAILABLE:
        logger.warning("Advanced features not available. Some examples will be skipped.")
    
    # Sync examples
    sync_examples = [
        example_1_concurrent_processing,
        example_2_model_pooling,
        example_3_auto_scaling,
        example_4_load_balancing,
        example_5_scaling_orchestrator,
        example_7_event_stream_processing
    ]
    
    for i, example_func in enumerate(sync_examples, 1):
        try:
            example_func()
        except Exception as e:
            logger.error(f"Example {i} failed: {e}")
            import traceback
            traceback.print_exc()
        
        time.sleep(0.5)
    
    # Async example
    try:
        logger.info("Running async processing example...")
        asyncio.run(example_6_async_processing())
    except Exception as e:
        logger.error(f"Async example failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("All advanced scaling examples completed!")


if __name__ == "__main__":
    main()