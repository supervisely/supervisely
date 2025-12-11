"""
Test script to generate a mock Live Training report
Place this in supervisely/supervisely repo root
"""
import os
import supervisely as sly
from supervisely.template.live_training.live_training_generator import LiveTrainingGenerator
from datetime import datetime, timedelta
import random

task_id = 100001

def generate_mock_loss_history(num_iterations: int = 690):
    """Generate realistic loss curves"""
    loss_history = {
        "loss": [],
        "decode.loss_ce": [],
        "decode.loss_mask": [],
        "decode.loss_dice": [],
    }
    
    # Start with high loss and decay
    base_loss = 5.0
    for i in range(0, num_iterations, 10):  # Save every 10 iterations
        # Exponential decay with noise
        factor = 0.995 ** i
        noise = random.uniform(-0.05, 0.05)
        
        total_loss = base_loss * factor + noise
        loss_history["loss"].append({"step": i, "value": max(0.5, total_loss)})
        loss_history["decode.loss_ce"].append({"step": i, "value": max(0.1, total_loss * 0.4 + noise)})
        loss_history["decode.loss_mask"].append({"step": i, "value": max(0.2, total_loss * 0.3 + noise)})
        loss_history["decode.loss_dice"].append({"step": i, "value": max(0.2, total_loss * 0.3 + noise)})
    
    return loss_history


def generate_mock_session_info(api: sly.Api, project_id: int):
    """Generate complete session_info with all required fields"""
    
    # Get real project info
    project_info = api.project.get_info_by_id(project_id)
    
    # Session timing
    start_time = datetime(2025, 12, 10, 10, 43, 42)
    end_time = start_time + timedelta(minutes=13)
    duration = "0h 13m"
    
    # Generate loss history
    total_iterations = 690
    loss_history = generate_mock_loss_history(total_iterations)
    
    # Checkpoints
    checkpoints = [
        {
            "name": "plateau_iter_689.pth",
            "iteration": 689,
            "loss": 3.403881,
        },
        {
            "name": "plateau_iter_759.pth",
            "iteration": 759,
            "loss": None,
        }
    ]
    
    # Artifacts directory (mock path)
    team_id = 51
    artifacts_dir = f"/experiments/{project_id}/{task_id}_online_training_Mask2Former"
    
    session_info = {
        # Session metadata
        "session_id": 1765363422,
        "task_id": task_id,
        "session_name": "Online Training - Task 54169",
        "project_id": project_id,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": duration,
        "status": "completed",
        # Training progress
        "current_iteration": total_iterations,
        "total_iterations": total_iterations,
        "loss_history": loss_history,
        
        # Device info (mock GPU)
        "device": "NVIDIA GeForce RTX 4090",
        
        # Dataset info
        "dataset_size": 2,
        "initial_samples": 2,
        "samples_added": 0,
        
        # Checkpoints
        "checkpoints": checkpoints,
        
        # Artifacts paths
        "artifacts_dir": artifacts_dir,
        "logs_dir": f"{artifacts_dir}/logs",
        
        # Session URL (mock)
        "session_url": f"{api.server_address}/apps/sessions/{54169}",
        
        # Hyperparameters (optional)
        "hyperparameters": {
            "learning_rate": 0.0001,
            "batch_size": 2,
            "max_epochs": 100,
            "optimizer": "AdamW",
            "weight_decay": 0.05,
            "crop_size": "(512, 512)",
        }
    }
    
    return session_info


def generate_mock_model_config():
    """Generate model configuration"""
    return {
        "model_name": "Mask2Former",
        "backbone": "SwinTransformer",
        "task_type": "semantic segmentation",
        "num_classes": 10,
        "config_file": "configs/mask2former/custom/swin-t-online-app.py",
    }


def generate_mock_model_meta():
    """Generate ProjectMeta with tomato classes"""
    classes = [
        "Columella",
        "Core", 
        "Locule",
        "Navel",
        "Pericarp",
        "Placenta",
        "Septum",
        "Tomato",
        "Sepal",
        "Background"
    ]
    
    # Generate random colors for each class
    obj_classes = []
    for class_name in classes:
        color = [random.randint(0, 255) for _ in range(3)]
        obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Bitmap, color=color)
        obj_classes.append(obj_class)
    
    return sly.ProjectMeta(obj_classes=obj_classes)


def main():
    """Main test function"""
    
    # Initialize API
    api = sly.Api()
    
    # Use your test project ID
    project_id = 4596  # Replace with your test project ID
    team_id = 51
    
    
    print("🔧 Generating mock data...")
    
    # Generate all mock data
    session_info = generate_mock_session_info(api, project_id)
    model_config = generate_mock_model_config()
    model_meta = generate_mock_model_meta()
    
    print(f"✓ Session ID: {session_info['session_id']}")
    print(f"✓ Iterations: {session_info['total_iterations']}")
    print(f"✓ Device: {session_info['device']}")
    print(f"✓ Classes: {len(model_meta.obj_classes)}")
    print(f"✓ Checkpoints: {len(session_info['checkpoints'])}")
    
    # Create generator
    print("\n📊 Initializing LiveTrainingGenerator...")
    
    output_dir = "./test_live_training_report"
    os.makedirs(output_dir, exist_ok=True)
    
    generator = LiveTrainingGenerator(
        api=api,
        session_info=session_info,
        model_config=model_config,
        model_meta=model_meta,
        output_dir=output_dir,
        team_id=team_id,
        task_type="semantic segmentation",
    )
    
    # Generate report
    print("\n🎨 Generating report...")
    print("\n🔍 DEBUG Context:")
    context = generator.context()
    print(f"session.task_id = {context['session'].get('task_id')}")
    print(f"model.classes = {context['model'].get('classes')}")
    print(f"resources = {context.get('resources')}")
    generator.generate()
    

    # Upload to Team Files
    print("\n☁️  Uploading to Team Files...")
    try:
        # ADD TIMESTAMP to make unique path
        import time
        timestamp = int(time.time())
        remote_dir = f"/test_reports/live_training_{timestamp}"
        
        # Upload report
        file_info = generator.upload_to_artifacts(remote_dir)
        
        # Get report URL
        report_url = generator.get_report()
        
        print(f"✅ Uploaded successfully!")
        print(f"🎯 Report URL: {report_url}")
        print(f"📂 Team Files path: {remote_dir}")
        
        # Fix: file_info can be int or FileInfo
        if isinstance(file_info, int):
            print(f"📄 File ID: {file_info}")
        else:
            print(f"📄 File path: {file_info.path}")
        
    except Exception as e:
        import traceback
        print(f"⚠️  Failed to upload: {e}")
        traceback.print_exc()

 

if __name__ == "__main__":
    main()