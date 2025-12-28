import os
import shutil
from tqdm import tqdm
from torchvision import datasets, transforms
from PIL import Image
from ultralytics import YOLO
import torch
import numpy as np

def recreate_clean_mnist_dataset():
    """COMPLETELY recreate a clean MNIST dataset"""
    print("COMPLETELY recreating clean MNIST dataset...")
    
    # Remove ALL old datasets
    if os.path.exists("mnist_yolo_dataset"):
        shutil.rmtree("mnist_yolo_dataset")
        print(" Deleted old corrupted dataset")
    
    # Remove training runs
    if os.path.exists("runs"):
        shutil.rmtree("runs")
        print(" Deleted old training runs")
    
    # Load fresh MNIST data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    # Create directories
    for split in ["train", "val"]:
        for cls in range(10):
            os.makedirs(f"mnist_yolo_dataset/{split}/{cls}", exist_ok=True)
    
    # Save training images (CLEAN)
    print("💾 Saving CLEAN training images...")
    for idx in tqdm(range(len(train_dataset))):
        img, label = train_dataset[idx]
        img = img.squeeze(0).numpy() * 255
        img = img.astype("uint8")
        img_pil = Image.fromarray(img, mode="L")
        img_path = f"mnist_yolo_dataset/train/{label}/train_{idx}.png"
        img_pil.save(img_path)
    
    # Save validation images (CLEAN)  
    print("💾 Saving CLEAN validation images...")
    for idx in tqdm(range(len(test_dataset))):
        img, label = test_dataset[idx]
        img = img.squeeze(0).numpy() * 255
        img = img.astype("uint8")
        img_pil = Image.fromarray(img, mode="L")
        img_path = f"mnist_yolo_dataset/val/{label}/val_{idx}.png"
        img_pil.save(img_path)
    
    print("CLEAN MNIST dataset completely recreated!")

def train_yolo_medium_accuracy():
    print("\n Training YOLO for MEDIUM accuracy (90-92%)...")
    
    # Check GPU
    if torch.cuda.is_available():
        device = 0
        print(" GPU Available")
    else:
        device = "cpu"
        print("Training on CPU")

    # Load model
    model = YOLO("yolov8n-cls.pt")
    
    #  PARAMETERS FOR MEDIUM ACCURACY (90-92%)
    print("\n Applying parameters for medium accuracy (90-92%)...")
    
    model.train(
        data="mnist_yolo_dataset",
        epochs=2,               #  Fewer epochs (reduced from 5)
        imgsz=32,               #  Smaller images (reduced from 64)
        batch=64,               #  Larger batch size (increased from 32)
        device=device,
        lr0=0.01,               #  Higher learning rate (increased from 0.001)
        lrf=0.1,                #  Higher learning rate decay
        momentum=0.8,           #  Lower momentum (reduced from 0.9)
        weight_decay=0.001,     #  Higher regularization (increased from 0.0005)
        warmup_epochs=0,        #  No warmup (reduced from 1)
        patience=5,             #  Less patience (reduced from 20)
        augment=True,           #  Keep augmentation
        hsv_h=0.05,             #  More color changes (increased from 0.015)
        hsv_s=0.8,              #  More saturation changes
        hsv_v=0.5,              #  More value changes
        degrees=15.0,           #  More rotation (increased from 10.0)
        translate=0.2,          #  More translation (increased from 0.1)
        scale=0.3,              #  More scaling (increased from 0.2)
        shear=0.2,              #  Add shearing (was 0.0)
        flipud=0.2,             #  Add vertical flips (was 0.0)
        fliplr=0.8,             #  More horizontal flips (increased from 0.5)
        mosaic=0.5,             #  Less mosaic (reduced from 1.0)
        mixup=0.3,              #  More mixup (increased from 0.1)
        copy_paste=0.1,         #  Add copy-paste (was 0.0)
        erasing=0.3,            #  More random erasing (increased from 0.1)
        crop_fraction=0.8,      #  Partial crops (reduced from 1.0)
        optimizer="SGD",        #  Use SGD instead of Adam
        overlap_mask=True,
        verbose=True
    )

    print("\n🎉 Training complete! Model saved in runs/classify/train")

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------
    print("\n Evaluating Model Accuracy...")
    
    # Load the trained model
    model_path = "runs/classify/train/weights/best.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        model = YOLO("runs/classify/train/weights/last.pt")
    
    results = model.val(
        data="mnist_yolo_dataset",
        split="val",
        imgsz=32,              #  Match training size
        device=device
    )

    print("\n=====  FINAL ACCURACY =====")
    print(f"Top-1 Accuracy: {results.top1:.2f}%")
    print(f"Top-5 Accuracy: {results.top5:.2f}%")

    # ----------------------------------------------------------------
    # Manual per-class accuracy for detailed analysis
    # ----------------------------------------------------------------
    print("\n===== CLASS-WISE ACCURACY (MANUAL) =====")

    num_classes = 10
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    val_dir = "mnist_yolo_dataset/val"
    imgsz = 32  # Match training size

    # iterate classes
    for cls in range(num_classes):
        cls_dir = os.path.join(val_dir, str(cls))
        if not os.path.exists(cls_dir):
            print(f"Warning: {cls_dir} not found, skipping class {cls}")
            continue
        files = sorted(os.listdir(cls_dir))
        # iterate images of this class
        for fname in tqdm(files, desc=f"Class {cls}", leave=False):
            img_path = os.path.join(cls_dir, fname)
            try:
                # load grayscale and convert to RGB (3 channels)
                img = Image.open(img_path).convert("L")
                img = img.resize((imgsz, imgsz), Image.BILINEAR)
                arr = np.array(img, dtype=np.uint8)
                arr_rgb = np.stack([arr, arr, arr], axis=2)

                # Use model.predict
                preds = model.predict(source=arr_rgb, device=device, verbose=False)
                if len(preds) == 0:
                    class_total[cls] += 1
                    continue
                res = preds[0]
                pred_cls = int(res.probs.top1)
                if pred_cls == cls:
                    class_correct[cls] += 1
                class_total[cls] += 1
            except Exception as e:
                continue

    # print per-class accuracy
    total_correct = 0
    total_samples = 0
    for cls in range(num_classes):
        total = class_total[cls]
        correct = class_correct[cls]
        acc = (100.0 * correct / total) if total > 0 else 0.0
        total_correct += correct
        total_samples += total
        print(f"Class {cls}: {acc:.2f}%  ({correct}/{total})")
    
    overall_acc = (100.0 * total_correct / total_samples) if total_samples > 0 else 0
    print(f"\nOverall Accuracy: {overall_acc:.2f}%")

    # ----------------------------------------------------------------
    # Final results analysis
    # ----------------------------------------------------------------
    if 90 <= overall_acc <= 92:
        print("\n PERFECT! Achieved target accuracy range (90-92%)!")
        print("Parameters optimized for medium accuracy!")
    elif overall_acc > 92:
        print(f"\n  Accuracy ({overall_acc:.2f}%) higher than target")
        print("Try reducing epochs to 1 or increasing learning rate to 0.02")
    else:
        print(f"\n  Accuracy ({overall_acc:.2f}%) lower than target")
        print("Try increasing epochs to 3 or reducing learning rate to 0.005")

    print("\n🎉 Final evaluation complete!")

def main():
    """
    MAIN EXECUTION - Run this to train for medium accuracy (90-92%)
    """
    print("=" * 60)
    print("MNIST YOLO MEDIUM ACCURACY TRAINING (90-92%)")
    print("=" * 60)
    
    # Step 1: Recreate clean dataset
    recreate_clean_mnist_dataset()
    
    print("\n" + "=" * 60)
    print(" STARTING MEDIUM ACCURACY TRAINING")
    print("=" * 60)
    
    # Step 2: Train for medium accuracy
    train_yolo_medium_accuracy()
    
    print("\n" + "=" * 60)
    print(" PROCESS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":

    main()
