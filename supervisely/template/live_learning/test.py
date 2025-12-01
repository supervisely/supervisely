import os
import supervisely as sly
from live_learning_generator import LiveLearningGenerator
from supervisely.project.project_meta import ProjectMeta
from supervisely.annotation.obj_class import ObjClass
from supervisely.geometry.bitmap import Bitmap


def create_test_session_info():
    """Create test session info with all required fields"""
    project_id = 4709
    session_id = 1234568910
    
    artifacts_dir = f"/live-learning/{project_id}_test_project/{session_id}"
    
    # Generate sample loss history
    loss_history = []
    for i in range(1, 151):
        loss = 0.8 * (0.99 ** i) + 0.2  # Exponential decay
        if i % 10 == 0:  # Save every 10 iterations
            loss_history.append({"iteration": i, "loss": loss})
    
    return {
        "session_id": session_id,
        "session_name": "Live Learning", 
        "project_id": project_id,
        "start_time": "2025-12-01 15:49:44",
        "duration": "2h 15m",  
        "artifacts_dir": artifacts_dir,
        "logs_dir": f"{artifacts_dir}/logs",
        "checkpoints": [
            {"name": "checkpoint_50.pth", "iteration": 50, "loss": 0.456},
            {"name": "checkpoint_100.pth", "iteration": 100, "loss": 0.312},
            {"name": "checkpoint_150.pth", "iteration": 150, "loss": 0.234},
        ],
        "loss_history": loss_history,  
        "hyperparameters": {
            "learning_rate": 0.0002,
            "batch_size": 2,
            "crop_size": "(768, 768)",
            "max_epochs": 100000,
            "weight_decay": 0.0005,
            "optimizer": "AdamW",
        },
    }


def create_test_model_config():
    """Create test model configuration"""
    return {
        "model_name": "Mask2Former",
        "backbone": "Swin-T",
        "config_file": "configs/mask2former/custom/swin-t-online-app.py",
        "pretrained_weights": "/weights/mask2former_swin-t_8xb2-160k_ade20k-512x512_20221203_234230-7d64e5dd.pth",
    }


def create_test_model_meta():
    """Create test ProjectMeta with sample classes"""
    classes = [
        ObjClass("car", Bitmap),
        ObjClass("person", Bitmap),
        ObjClass("tree", Bitmap),
    ]
    return ProjectMeta(obj_classes=classes)


def save_test_config(output_dir: str):
    """Save a test config file"""
    config_content = """_base_ = '../mask2former_swin-t_8xb2-160k_ade20k-512x512.py'

# Image and crop settings
crop_size = (768, 768)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0005, eps=1e-8, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

# Training schedule
max_epochs = 100000
train_dataloader = dict(batch_size=2)
"""
    
    config_path = os.path.join(output_dir, "config.py")
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"✅ Test config saved: {config_path}")
    return config_path


def main():
    # Initialize API
    api = sly.Api()
    print(f"✅ Connected to {api.server_address}")
    
    # Create test data
    session_info = create_test_session_info()
    model_config = create_test_model_config()
    model_meta = create_test_model_meta()
    
    # Output directory
    output_dir = "./test_live_learning_report"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save test config
    save_test_config(output_dir)
    
    print("\n" + "="*50)
    print("Creating LiveLearningGenerator...")
    print("="*50)
    
    # Create generator
    generator = LiveLearningGenerator(
        api=api,
        session_info=session_info,
        model_config=model_config,
        model_meta=model_meta,
        output_dir=output_dir,
        team_id=51,
    )
    
    print("\n⚙️  Generating report...")
    generator.generate()
    
    print(f"\n✅ Report generated successfully!")
    print(f"📂 Local files: {output_dir}")
    print(f"   - template.vue")
    print(f"   - state.json")
    print(f"   - config.py")
    
    # Ask user if they want to upload
    upload_choice = input("\n🚀 Upload to Supervisely? (y/n): ").lower()
    
    if upload_choice == 'y':
        print("\n⬆️  Uploading to Supervisely...")
        
        artifacts_dir = session_info["artifacts_dir"]
        file_info = generator.upload_to_artifacts(artifacts_dir)
        
        report_url = generator.get_report()
        
        print(f"\n✅ Upload complete!")
        print(f"🔗 Report URL: {report_url}")
        print(f"📂 Artifacts: {api.server_address}/files/?path={artifacts_dir}")
    else:
        print("\n⏭️  Skipped upload")
    
    print("\n" + "="*50)
    print("Test completed!")
    print("="*50)


if __name__ == "__main__":
    main()