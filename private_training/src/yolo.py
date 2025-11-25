

# from ultralytics import YOLO
# import torch
# import os
# from tqdm import tqdm
# from PIL import Image
# import numpy as np

# def resume_training_on_gpu():
#     print("\n🔥 Checking GPU availability...")

#     if torch.cuda.is_available():
#         gpu_name = torch.cuda.get_device_name(0)
#         print(f"💻 GPU Active: {gpu_name}")
#         device = 0
#     else:
#         print("❌ No GPU found. Cannot resume on GPU.")
#         return

#     last_checkpoint = "runs/classify/train/weights/last.pt"

#     if not os.path.exists(last_checkpoint):
#         print("❌ No previous training checkpoint found!")
#         print("Expected:", last_checkpoint)
#         return

#     print(f"\n🔄 Loading previous checkpoint: {last_checkpoint}")
#     model = YOLO(last_checkpoint)

#     print("\n🚀 Resuming training ON GPU (AMP disabled for compatibility)...")

#     model.train(
#         data="mnist_yolo_dataset",
#         epochs=20,       # train until epoch 20 total
#         imgsz=64,
#         batch=64,
#         device=device,
#         amp=False,       # avoids GradScaler mismatch when resuming
#         resume=False     # do NOT load old scaler/optimizer state
#     )

#     print("\n🎉 Training completed on GPU!")

#     # ----------------------------------------------------------------
#     # Built-in evaluation (Top-1 / Top-5)
#     # ----------------------------------------------------------------
#     print("\n📊 Evaluating Final Model Accuracy (built-in top1/top5)...")
#     results = model.val(
#         data="mnist_yolo_dataset",
#         split="val",
#         imgsz=64,
#         device=device
#     )

#     print("\n===== 🏆 FINAL ACCURACY =====")
#     print(f"Top-1 Accuracy: {results.top1:.2f}%")
#     print(f"Top-5 Accuracy: {results.top5:.2f}%")

#     # ----------------------------------------------------------------
#     # Manual per-class accuracy (since ultralytics classify metrics don't expose class-wise acc)
#     # We'll load each image, convert to RGB (3 channels), resize to imgsz, and call model.predict.
#     # model.predict accepts a numpy HWC image or file path.
#     # ----------------------------------------------------------------
#     print("\n===== 📌 CLASS-WISE ACCURACY (MANUAL) =====")

#     num_classes = 10
#     class_correct = [0] * num_classes
#     class_total = [0] * num_classes

#     val_dir = "mnist_yolo_dataset/val"
#     imgsz = 64

#     # iterate classes
#     for cls in range(num_classes):
#         cls_dir = os.path.join(val_dir, str(cls))
#         if not os.path.exists(cls_dir):
#             print(f"Warning: {cls_dir} not found, skipping class {cls}")
#             continue
#         files = sorted(os.listdir(cls_dir))
#         # iterate images of this class
#         for fname in tqdm(files, desc=f"Class {cls}", leave=False):
#             img_path = os.path.join(cls_dir, fname)
#             try:
#                 # load grayscale and convert to RGB (3 channels)
#                 img = Image.open(img_path).convert("L")
#                 img = img.resize((imgsz, imgsz), Image.BILINEAR)
#                 arr = np.array(img, dtype=np.uint8)  # H,W, (grayscale)
#                 # stack channels -> H,W,3
#                 arr_rgb = np.stack([arr, arr, arr], axis=2)

#                 # Use model.predict on numpy HWC image (ultralytics handles preprocessing)
#                 preds = model.predict(source=arr_rgb, device=device, verbose=False)  # returns list of Results
#                 if len(preds) == 0:
#                     # no result? count as incorrect
#                     class_total[cls] += 1
#                     continue
#                 res = preds[0]
#                 # res.probs.top1 is top-1 predicted class index (int or tensor)
#                 pred_cls = int(res.probs.top1)
#                 if pred_cls == cls:
#                     class_correct[cls] += 1
#                 class_total[cls] += 1
#             except Exception as e:
#                 # skip corrupt images but warn
#                 print(f"\nWarning: failed to process {img_path}: {e}")
#                 continue

#     # print per-class accuracy
#     for cls in range(num_classes):
#         total = class_total[cls]
#         correct = class_correct[cls]
#         acc = (100.0 * correct / total) if total > 0 else 0.0
#         print(f"Class {cls}: {acc:.2f}%  ({correct}/{total})")

#     print("\n🎉 Final evaluation complete!")

# if __name__ == "__main__":
#     resume_training_on_gpu()









































# from ultralytics import YOLO
# import torch
# import os
# from tqdm import tqdm
# from PIL import Image
# import numpy as np

# def train_yolo_with_reduced_accuracy():
#     print("\n🔥 Training YOLO with reduced accuracy target (90-92%)...")
    
#     # Check GPU
#     if torch.cuda.is_available():
#         device = 0
#         print("💻 GPU Available")
#     else:
#         device = "cpu"
#         print("❌ Training on CPU")

#     # Load model
#     model = YOLO("yolov8n-cls.pt")
    
#     # 🔽 KEY PARAMETERS TO REDUCE ACCURACY 🔽
#     print("\n🎯 Applying accuracy-reducing parameters...")
    
#     model.train(
#         data="mnist_yolo_dataset",
#         epochs=5,               # 🔽 Fewer epochs (was 20)
#         imgsz=28,              # 🔽 Smaller image size (was 64)
#         batch=128,             # 🔽 Larger batch size - can reduce generalization
#         device=device,
#         lr0=0.01,              # 🔽 Higher learning rate - might overshoot
#         lrf=0.01,              # 🔽 Fixed learning rate (no decay)
#         momentum=0.9,          # 🔽 Standard momentum
#         weight_decay=0.0001,   # 🔽 Reduced regularization
#         warmup_epochs=0,       # 🔽 No warmup
#         patience=10,           # 🔽 Early stopping patience
#         augment=False,          # 🔽 NO data augmentation - reduces generalization
#         hsv_h=0.0,             # 🔽 No color augmentation
#         hsv_s=0.0,
#         hsv_v=0.0,
#         degrees=0.0,           # 🔽 No rotation
#         translate=0.0,         # 🔽 No translation
#         scale=0.0,             # 🔽 No scaling
#         shear=0.0,             # 🔽 No shearing
#         flipud=0.0,            # 🔽 No flips
#         fliplr=0.0,
#         mosaic=0.0,            # 🔽 No mosaic augmentation
#         mixup=0.0,             # 🔽 No mixup
#         copy_paste=0.0,        # 🔽 No copy-paste
#         erasing=0.0,           # 🔽 No random erasing
#         crop_fraction=0.8,     # 🔽 Smaller crop fraction
#         overlap_mask=False,
#         # optimizer="SGD",      # Uncomment for SGD (usually worse than Adam)
#         verbose=True
#     )

#     print("\n🎉 Training complete! Model saved in runs/classify/train")

#     # ----------------------------------------------------------------
#     # Evaluation
#     # ----------------------------------------------------------------
#     print("\n📊 Evaluating Model Accuracy...")
    
#     # Load the trained model
#     model_path = "runs/classify/train/weights/best.pt"
#     if os.path.exists(model_path):
#         model = YOLO(model_path)
#     else:
#         model = YOLO("runs/classify/train/weights/last.pt")
    
#     results = model.val(
#         data="mnist_yolo_dataset",
#         split="val",
#         imgsz=28,              # 🔽 Match training size
#         device=device
#     )

#     print("\n===== 🏆 FINAL ACCURACY =====")
#     print(f"Top-1 Accuracy: {results.top1:.2f}%")
#     print(f"Top-5 Accuracy: {results.top5:.2f}%")

#     # ----------------------------------------------------------------
#     # Manual per-class accuracy
#     # ----------------------------------------------------------------
#     print("\n===== 📌 CLASS-WISE ACCURACY (MANUAL) =====")

#     num_classes = 10
#     class_correct = [0] * num_classes
#     class_total = [0] * num_classes

#     val_dir = "mnist_yolo_dataset/val"
#     imgsz = 28  # 🔽 Match training size

#     # iterate classes
#     for cls in range(num_classes):
#         cls_dir = os.path.join(val_dir, str(cls))
#         if not os.path.exists(cls_dir):
#             print(f"Warning: {cls_dir} not found, skipping class {cls}")
#             continue
#         files = sorted(os.listdir(cls_dir))
#         # iterate images of this class
#         for fname in tqdm(files, desc=f"Class {cls}", leave=False):
#             img_path = os.path.join(cls_dir, fname)
#             try:
#                 # load grayscale and convert to RGB (3 channels)
#                 img = Image.open(img_path).convert("L")
#                 img = img.resize((imgsz, imgsz), Image.BILINEAR)
#                 arr = np.array(img, dtype=np.uint8)
#                 arr_rgb = np.stack([arr, arr, arr], axis=2)

#                 # Use model.predict
#                 preds = model.predict(source=arr_rgb, device=device, verbose=False)
#                 if len(preds) == 0:
#                     class_total[cls] += 1
#                     continue
#                 res = preds[0]
#                 pred_cls = int(res.probs.top1)
#                 if pred_cls == cls:
#                     class_correct[cls] += 1
#                 class_total[cls] += 1
#             except Exception as e:
#                 continue

#     # print per-class accuracy
#     total_correct = 0
#     total_samples = 0
#     for cls in range(num_classes):
#         total = class_total[cls]
#         correct = class_correct[cls]
#         acc = (100.0 * correct / total) if total > 0 else 0.0
#         total_correct += correct
#         total_samples += total
#         print(f"Class {cls}: {acc:.2f}%  ({correct}/{total})")
    
#     overall_acc = (100.0 * total_correct / total_samples) if total_samples > 0 else 0
#     print(f"\nOverall Accuracy: {overall_acc:.2f}%")

#     # ----------------------------------------------------------------
#     # Additional accuracy reduction techniques if still too high
#     # ----------------------------------------------------------------
#     if overall_acc > 92:
#         print("\n🔄 Accuracy still too high, applying additional reduction...")
#         print("Try these additional techniques:")
#         print("1. Add noise to test images during evaluation")
#         print("2. Use smaller model (yolov8n instead of larger variants)")
#         print("3. Reduce training data quality")
#         print("4. Add label noise during training")

#     print("\n🎉 Final evaluation complete!")

# if __name__ == "__main__":
#     train_yolo_with_reduced_accuracy()













































































# from ultralytics import YOLO
# import torch
# import os
# from tqdm import tqdm
# from PIL import Image
# import numpy as np

# def train_yolo_target_accuracy():
#     print("\n🎯 Training YOLO with target accuracy (90-92%)...")
    
#     # Check GPU
#     if torch.cuda.is_available():
#         device = 0
#         print("💻 GPU Available")
#     else:
#         device = "cpu"
#         print("❌ Training on CPU")

#     # Load model
#     model = YOLO("yolov8n-cls.pt")
    
#     # 🎯 BALANCED PARAMETERS FOR 90-92% ACCURACY
#     print("\n🎯 Applying balanced parameters for 90-92% accuracy...")
    
#     model.train(
#         data="mnist_yolo_dataset",
#         epochs=5,               # ✅ Keep 5 epochs
#         imgsz=32,              # ✅ Slightly larger than 28 but smaller than 64
#         batch=64,              # ✅ Balanced batch size
#         device=device,
#         lr0=0.001,             # ✅ Lower learning rate for stability
#         lrf=0.01,              # ✅ Gentle learning rate decay
#         momentum=0.9,          # ✅ Standard momentum
#         weight_decay=0.0005,   # ✅ Moderate regularization
#         warmup_epochs=1,       # ✅ Add warmup for stability
#         patience=5,            # ✅ Reasonable early stopping
#         augment=True,          # ✅ Enable SOME augmentation but limit it
#         hsv_h=0.015,           # ✅ Minimal color augmentation
#         hsv_s=0.7,
#         hsv_v=0.4,
#         degrees=5.0,           # ✅ Small rotation
#         translate=0.1,         # ✅ Small translation
#         scale=0.1,             # ✅ Small scaling
#         shear=0.0,             # ❌ No shearing (too destructive)
#         flipud=0.0,            # ❌ No vertical flips
#         fliplr=0.5,            # ✅ Horizontal flips (good for digits)
#         mosaic=0.0,            # ❌ No mosaic (too complex)
#         mixup=0.0,             # ❌ No mixup
#         copy_paste=0.0,        # ❌ No copy-paste
#         erasing=0.0,           # ❌ No random erasing
#         crop_fraction=0.9,     # ✅ Larger crops
#         optimizer="Adam",      # ✅ Use Adam for better convergence
#         verbose=True
#     )

#     print("\n🎉 Training complete! Model saved in runs/classify/train")

#     # ----------------------------------------------------------------
#     # Evaluation
#     # ----------------------------------------------------------------
#     print("\n📊 Evaluating Model Accuracy...")
    
#     # Load the trained model
#     model_path = "runs/classify/train/weights/best.pt"
#     if os.path.exists(model_path):
#         model = YOLO(model_path)
#     else:
#         model = YOLO("runs/classify/train/weights/last.pt")
    
#     results = model.val(
#         data="mnist_yolo_dataset",
#         split="val",
#         imgsz=32,              # ✅ Match training size
#         device=device
#     )

#     print("\n===== 🏆 FINAL ACCURACY =====")
#     print(f"Top-1 Accuracy: {results.top1:.2f}%")
#     print(f"Top-5 Accuracy: {results.top5:.2f}%")

#     # ----------------------------------------------------------------
#     # Manual per-class accuracy
#     # ----------------------------------------------------------------
#     print("\n===== 📌 CLASS-WISE ACCURACY (MANUAL) =====")

#     num_classes = 10
#     class_correct = [0] * num_classes
#     class_total = [0] * num_classes

#     val_dir = "mnist_yolo_dataset/val"
#     imgsz = 32  # ✅ Match training size

#     # iterate classes
#     for cls in range(num_classes):
#         cls_dir = os.path.join(val_dir, str(cls))
#         if not os.path.exists(cls_dir):
#             print(f"Warning: {cls_dir} not found, skipping class {cls}")
#             continue
#         files = sorted(os.listdir(cls_dir))
#         # iterate images of this class
#         for fname in tqdm(files, desc=f"Class {cls}", leave=False):
#             img_path = os.path.join(cls_dir, fname)
#             try:
#                 # load grayscale and convert to RGB (3 channels)
#                 img = Image.open(img_path).convert("L")
#                 img = img.resize((imgsz, imgsz), Image.BILINEAR)
#                 arr = np.array(img, dtype=np.uint8)
#                 arr_rgb = np.stack([arr, arr, arr], axis=2)

#                 # Use model.predict
#                 preds = model.predict(source=arr_rgb, device=device, verbose=False)
#                 if len(preds) == 0:
#                     class_total[cls] += 1
#                     continue
#                 res = preds[0]
#                 pred_cls = int(res.probs.top1)
#                 if pred_cls == cls:
#                     class_correct[cls] += 1
#                 class_total[cls] += 1
#             except Exception as e:
#                 continue

#     # print per-class accuracy
#     total_correct = 0
#     total_samples = 0
#     for cls in range(num_classes):
#         total = class_total[cls]
#         correct = class_correct[cls]
#         acc = (100.0 * correct / total) if total > 0 else 0.0
#         total_correct += correct
#         total_samples += total
#         print(f"Class {cls}: {acc:.2f}%  ({correct}/{total})")
    
#     overall_acc = (100.0 * total_correct / total_samples) if total_samples > 0 else 0
#     print(f"\nOverall Accuracy: {overall_acc:.2f}%")

#     # ----------------------------------------------------------------
#     # Fine-tuning based on results
#     # ----------------------------------------------------------------
#     if overall_acc > 92:
#         print("\n📈 Accuracy too high, can reduce slightly by:")
#         print("1. Increase batch size to 128")
#         print("2. Remove data augmentation")
#         print("3. Reduce image size to 28")
        
#     elif overall_acc < 90:
#         print("\n📈 Accuracy too low, can improve by:")
#         print("1. Increase epochs to 8")
#         print("2. Reduce batch size to 32")
#         print("3. Increase image size to 48")
#         print("4. Reduce learning rate to 0.0005")
        
#     else:
#         print("\n🎯 Perfect! Accuracy in target range (90-92%)")

#     print("\n🎉 Final evaluation complete!")

# if __name__ == "__main__":
#     train_yolo_target_accuracy()








































































































# import os
# import shutil
# from tqdm import tqdm
# from torchvision import datasets, transforms
# from PIL import Image
# from ultralytics import YOLO
# import torch
# import numpy as np

# def recreate_clean_mnist_dataset():
#     """COMPLETELY recreate a clean MNIST dataset"""
#     print("🔄 COMPLETELY recreating clean MNIST dataset...")
    
#     # Remove ALL old datasets
#     if os.path.exists("mnist_yolo_dataset"):
#         shutil.rmtree("mnist_yolo_dataset")
#         print("✅ Deleted old corrupted dataset")
    
#     # Remove training runs
#     if os.path.exists("runs"):
#         shutil.rmtree("runs")
#         print("✅ Deleted old training runs")
    
#     # Load fresh MNIST data
#     transform = transforms.Compose([transforms.ToTensor()])
#     train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
#     test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
#     # Create directories
#     for split in ["train", "val"]:
#         for cls in range(10):
#             os.makedirs(f"mnist_yolo_dataset/{split}/{cls}", exist_ok=True)
    
#     # Save training images (CLEAN)
#     print("💾 Saving CLEAN training images...")
#     for idx in tqdm(range(len(train_dataset))):
#         img, label = train_dataset[idx]
#         img = img.squeeze(0).numpy() * 255
#         img = img.astype("uint8")
#         img_pil = Image.fromarray(img, mode="L")
#         img_path = f"mnist_yolo_dataset/train/{label}/train_{idx}.png"
#         img_pil.save(img_path)
    
#     # Save validation images (CLEAN)  
#     print("💾 Saving CLEAN validation images...")
#     for idx in tqdm(range(len(test_dataset))):
#         img, label = test_dataset[idx]
#         img = img.squeeze(0).numpy() * 255
#         img = img.astype("uint8")
#         img_pil = Image.fromarray(img, mode="L")
#         img_path = f"mnist_yolo_dataset/val/{label}/val_{idx}.png"
#         img_pil.save(img_path)
    
#     print("✅ CLEAN MNIST dataset completely recreated!")

# def train_yolo_high_accuracy():
#     print("\n🔥 Training YOLO for HIGH accuracy (96%+)...")
    
#     # Check GPU
#     if torch.cuda.is_available():
#         device = 0
#         print("💻 GPU Available")
#     else:
#         device = "cpu"
#         print("❌ Training on CPU")

#     # Load model
#     model = YOLO("yolov8n-cls.pt")
    
#     # 🎯 OPTIMAL PARAMETERS FOR HIGH ACCURACY
#     print("\n🎯 Applying optimal parameters for high accuracy...")
    
#     model.train(
#         data="mnist_yolo_dataset",
#         epochs=5,              # ✅ More epochs for better learning
#         imgsz=64,              # ✅ Larger images for more details
#         batch=32,              # ✅ Optimal batch size
#         device=device,
#         lr0=0.001,             # ✅ Stable learning rate
#         lrf=0.01,              # ✅ Learning rate decay
#         momentum=0.9,          # ✅ Momentum
#         weight_decay=0.0005,   # ✅ Regularization
#         warmup_epochs=1,       # ✅ Warmup
#         patience=20,           # ✅ Don't stop early
#         augment=True,          # ✅ Enable augmentation
#         hsv_h=0.015,           # ✅ Minimal color changes
#         hsv_s=0.7,
#         hsv_v=0.4,
#         degrees=10.0,          # ✅ Reasonable rotation
#         translate=0.1,         # ✅ Reasonable translation
#         scale=0.2,             # ✅ Reasonable scaling
#         shear=0.0,             # ✅ No shearing (preserves digits)
#         flipud=0.0,            # ✅ No vertical flips
#         fliplr=0.5,            # ✅ Horizontal flips (good for digits)
#         mosaic=1.0,            # ✅ Mosaic augmentation
#         mixup=0.1,             # ✅ Mixup augmentation
#         copy_paste=0.0,        # ✅ No copy-paste
#         erasing=0.1,           # ✅ Random erasing
#         crop_fraction=1.0,     # ✅ Full crops
#         optimizer="Adam",      # ✅ Adam optimizer
#         overlap_mask=True,
#         verbose=True
#     )

#     print("\n🎉 Training complete! Model saved in runs/classify/train")

#     # ----------------------------------------------------------------
#     # Evaluation
#     # ----------------------------------------------------------------
#     print("\n📊 Evaluating Model Accuracy...")
    
#     # Load the trained model
#     model_path = "runs/classify/train/weights/best.pt"
#     if os.path.exists(model_path):
#         model = YOLO(model_path)
#     else:
#         model = YOLO("runs/classify/train/weights/last.pt")
    
#     results = model.val(
#         data="mnist_yolo_dataset",
#         split="val",
#         imgsz=64,              # ✅ Match training size
#         device=device
#     )

#     print("\n===== 🏆 FINAL ACCURACY =====")
#     print(f"Top-1 Accuracy: {results.top1:.2f}%")
#     print(f"Top-5 Accuracy: {results.top5:.2f}%")

#     # ----------------------------------------------------------------
#     # Manual per-class accuracy for detailed analysis
#     # ----------------------------------------------------------------
#     print("\n===== 📌 CLASS-WISE ACCURACY (MANUAL) =====")

#     num_classes = 10
#     class_correct = [0] * num_classes
#     class_total = [0] * num_classes

#     val_dir = "mnist_yolo_dataset/val"
#     imgsz = 64  # ✅ Match training size

#     # iterate classes
#     for cls in range(num_classes):
#         cls_dir = os.path.join(val_dir, str(cls))
#         if not os.path.exists(cls_dir):
#             print(f"Warning: {cls_dir} not found, skipping class {cls}")
#             continue
#         files = sorted(os.listdir(cls_dir))
#         # iterate images of this class
#         for fname in tqdm(files, desc=f"Class {cls}", leave=False):
#             img_path = os.path.join(cls_dir, fname)
#             try:
#                 # load grayscale and convert to RGB (3 channels)
#                 img = Image.open(img_path).convert("L")
#                 img = img.resize((imgsz, imgsz), Image.BILINEAR)
#                 arr = np.array(img, dtype=np.uint8)
#                 arr_rgb = np.stack([arr, arr, arr], axis=2)

#                 # Use model.predict
#                 preds = model.predict(source=arr_rgb, device=device, verbose=False)
#                 if len(preds) == 0:
#                     class_total[cls] += 1
#                     continue
#                 res = preds[0]
#                 pred_cls = int(res.probs.top1)
#                 if pred_cls == cls:
#                     class_correct[cls] += 1
#                 class_total[cls] += 1
#             except Exception as e:
#                 continue

#     # print per-class accuracy
#     total_correct = 0
#     total_samples = 0
#     for cls in range(num_classes):
#         total = class_total[cls]
#         correct = class_correct[cls]
#         acc = (100.0 * correct / total) if total > 0 else 0.0
#         total_correct += correct
#         total_samples += total
#         print(f"Class {cls}: {acc:.2f}%  ({correct}/{total})")
    
#     overall_acc = (100.0 * total_correct / total_samples) if total_samples > 0 else 0
#     print(f"\nOverall Accuracy: {overall_acc:.2f}%")

#     # ----------------------------------------------------------------
#     # Final results analysis
#     # ----------------------------------------------------------------
#     if overall_acc >= 95:
#         print("\n🎉 EXCELLENT! Back to high accuracy range (95%+)!")
#         print("✅ Dataset restoration successful!")
#         print("✅ Training parameters optimal!")
#     elif overall_acc >= 90:
#         print("\n👍 GOOD! Solid accuracy achieved (90%+)")
#         print("✅ Dataset is clean and working well!")
#     else:
#         print(f"\n⚠️  Accuracy ({overall_acc:.2f}%) lower than expected")
#         print("💡 Try increasing epochs to 20 or reducing learning rate to 0.0005")

#     print("\n🎉 Final evaluation complete!")

# def main():
#     """
#     MAIN EXECUTION - Run this to completely restore and train for high accuracy
#     """
#     print("=" * 60)
#     print("🔄 MNIST YOLO HIGH ACCURACY RESTORATION")
#     print("=" * 60)
    
#     # Step 1: Recreate clean dataset
#     recreate_clean_mnist_dataset()
    
#     print("\n" + "=" * 60)
#     print("🚀 STARTING HIGH ACCURACY TRAINING")
#     print("=" * 60)
    
#     # Step 2: Train for high accuracy
#     train_yolo_high_accuracy()
    
#     print("\n" + "=" * 60)
#     print("✅ PROCESS COMPLETE!")
#     print("=" * 60)

# if __name__ == "__main__":
#     main()







































































































































































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
    print("🔄 COMPLETELY recreating clean MNIST dataset...")
    
    # Remove ALL old datasets
    if os.path.exists("mnist_yolo_dataset"):
        shutil.rmtree("mnist_yolo_dataset")
        print("✅ Deleted old corrupted dataset")
    
    # Remove training runs
    if os.path.exists("runs"):
        shutil.rmtree("runs")
        print("✅ Deleted old training runs")
    
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
    
    print("✅ CLEAN MNIST dataset completely recreated!")

def train_yolo_medium_accuracy():
    print("\n🔥 Training YOLO for MEDIUM accuracy (90-92%)...")
    
    # Check GPU
    if torch.cuda.is_available():
        device = 0
        print("💻 GPU Available")
    else:
        device = "cpu"
        print("❌ Training on CPU")

    # Load model
    model = YOLO("yolov8n-cls.pt")
    
    # 🎯 PARAMETERS FOR MEDIUM ACCURACY (90-92%)
    print("\n🎯 Applying parameters for medium accuracy (90-92%)...")
    
    model.train(
        data="mnist_yolo_dataset",
        epochs=2,               # ⬇️ Fewer epochs (reduced from 5)
        imgsz=32,               # ⬇️ Smaller images (reduced from 64)
        batch=64,               # ⬆️ Larger batch size (increased from 32)
        device=device,
        lr0=0.01,               # ⬆️ Higher learning rate (increased from 0.001)
        lrf=0.1,                # ⬆️ Higher learning rate decay
        momentum=0.8,           # ⬇️ Lower momentum (reduced from 0.9)
        weight_decay=0.001,     # ⬆️ Higher regularization (increased from 0.0005)
        warmup_epochs=0,        # ⬇️ No warmup (reduced from 1)
        patience=5,             # ⬇️ Less patience (reduced from 20)
        augment=True,           # ✅ Keep augmentation
        hsv_h=0.05,             # ⬆️ More color changes (increased from 0.015)
        hsv_s=0.8,              # ⬆️ More saturation changes
        hsv_v=0.5,              # ⬆️ More value changes
        degrees=15.0,           # ⬆️ More rotation (increased from 10.0)
        translate=0.2,          # ⬆️ More translation (increased from 0.1)
        scale=0.3,              # ⬆️ More scaling (increased from 0.2)
        shear=0.2,              # ⬆️ Add shearing (was 0.0)
        flipud=0.2,             # ⬆️ Add vertical flips (was 0.0)
        fliplr=0.8,             # ⬆️ More horizontal flips (increased from 0.5)
        mosaic=0.5,             # ⬇️ Less mosaic (reduced from 1.0)
        mixup=0.3,              # ⬆️ More mixup (increased from 0.1)
        copy_paste=0.1,         # ⬆️ Add copy-paste (was 0.0)
        erasing=0.3,            # ⬆️ More random erasing (increased from 0.1)
        crop_fraction=0.8,      # ⬇️ Partial crops (reduced from 1.0)
        optimizer="SGD",        # ⬇️ Use SGD instead of Adam
        overlap_mask=True,
        verbose=True
    )

    print("\n🎉 Training complete! Model saved in runs/classify/train")

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------
    print("\n📊 Evaluating Model Accuracy...")
    
    # Load the trained model
    model_path = "runs/classify/train/weights/best.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        model = YOLO("runs/classify/train/weights/last.pt")
    
    results = model.val(
        data="mnist_yolo_dataset",
        split="val",
        imgsz=32,              # ✅ Match training size
        device=device
    )

    print("\n===== 🏆 FINAL ACCURACY =====")
    print(f"Top-1 Accuracy: {results.top1:.2f}%")
    print(f"Top-5 Accuracy: {results.top5:.2f}%")

    # ----------------------------------------------------------------
    # Manual per-class accuracy for detailed analysis
    # ----------------------------------------------------------------
    print("\n===== 📌 CLASS-WISE ACCURACY (MANUAL) =====")

    num_classes = 10
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    val_dir = "mnist_yolo_dataset/val"
    imgsz = 32  # ✅ Match training size

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
        print("\n🎉 PERFECT! Achieved target accuracy range (90-92%)!")
        print("✅ Parameters optimized for medium accuracy!")
    elif overall_acc > 92:
        print(f"\n⚠️  Accuracy ({overall_acc:.2f}%) higher than target")
        print("💡 Try reducing epochs to 1 or increasing learning rate to 0.02")
    else:
        print(f"\n⚠️  Accuracy ({overall_acc:.2f}%) lower than target")
        print("💡 Try increasing epochs to 3 or reducing learning rate to 0.005")

    print("\n🎉 Final evaluation complete!")

def main():
    """
    MAIN EXECUTION - Run this to train for medium accuracy (90-92%)
    """
    print("=" * 60)
    print("🔄 MNIST YOLO MEDIUM ACCURACY TRAINING (90-92%)")
    print("=" * 60)
    
    # Step 1: Recreate clean dataset
    recreate_clean_mnist_dataset()
    
    print("\n" + "=" * 60)
    print("🚀 STARTING MEDIUM ACCURACY TRAINING")
    print("=" * 60)
    
    # Step 2: Train for medium accuracy
    train_yolo_medium_accuracy()
    
    print("\n" + "=" * 60)
    print("✅ PROCESS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()