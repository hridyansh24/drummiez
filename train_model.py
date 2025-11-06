
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
from PIL import Image

class DrumSheetDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        
        # The bbox is a string, so we need to convert it to a list of floats
        bbox_str = self.annotations.iloc[idx, 1]
        bbox = [float(x) for x in bbox_str.strip('[]').split(', ')]
        
        # Convert bbox to the format expected by PyTorch: [x_min, y_min, x_max, y_max]
        boxes = torch.as_tensor([bbox], dtype=torch.float32)
        
        # For now, we'll use a placeholder for labels
        labels = torch.ones((1,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transform:
            image = self.transform(image)

        return image, target

def get_model(num_classes):
    # Load a pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def main():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    dataset = DrumSheetDataset(csv_file='data/prepared_data.csv', root_dir='data/ds2_dense/images', transform=transform)
    
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Use a smaller subset of the validation data for quick evaluation
    val_subset_size = min(100, len(val_dataset))  # Evaluate on max 100 samples or less if dataset is smaller
    val_subset, _ = torch.utils.data.random_split(val_dataset, [val_subset_size, len(val_dataset) - val_subset_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_subset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # For now, we'll just use 2 classes: background and drum symbol
    num_classes = 2
    model = get_model(num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 1 # For now, we'll just train for 1 epoch

    for epoch in range(num_epochs):
        model.train()
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {losses.item()}")

        # Validation loop
        model.eval()
        val_loss = 0
        total_iou = 0
        num_predictions = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Get predictions
                outputs = model(images)

                for i, output in enumerate(outputs):
                    gt_boxes = targets[i]['boxes'].cpu().numpy()
                    pred_boxes = output['boxes'].cpu().numpy()

                    if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                        # For simplicity, we'll just compare each prediction to each ground truth
                        # and take the maximum IoU for each prediction.
                        # A more robust approach would involve Hungarian matching or similar.
                        for pred_box in pred_boxes:
                            best_iou = 0
                            for gt_box in gt_boxes:
                                iou = calculate_iou(pred_box, gt_box)
                                if iou > best_iou:
                                    best_iou = iou
                            total_iou += best_iou
                            num_predictions += 1

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_iou = total_iou / num_predictions if num_predictions > 0 else 0
        print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f}, Average IoU: {avg_iou:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'drum_omr_model.pth')
    print("Training complete. Model saved to drum_omr_model.pth")

def calculate_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

if __name__ == '__main__':
    main()
