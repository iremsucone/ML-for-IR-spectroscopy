import torch
import torchvision.transforms as transforms
from PIL import Image
from train_ir import IR_CNN, load_data  # Import trained model and dataset loader

# ✅ STEP 1: Set Dataset Path
root_dir = "/content/drive/MyDrive/IR_Spectro/data/"

# ✅ STEP 2: Load Functional Groups (Dynamically)
_, train_dataset = load_data(root_dir)  # Load dataset to get class names
class_names = train_dataset.classes  # Automatically fetch functional groups

# ✅ STEP 3: Load Trained Model
num_classes = len(class_names)
model = IR_CNN(num_classes)
model.load_state_dict(torch.load("/content/drive/MyDrive/ir_model.pth", weights_only=True))
model.eval()

# ✅ STEP 4: Define Image Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.3,), (0.7,))
])

# ✅ STEP 5: Prediction Function
def predict_image(image_path, model, class_names):
    """ Predicts the functional group of a given IR spectrum image. """
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return class_names[predicted.item()]  # Return the predicted class name

# ✅ STEP 6: Make Prediction
if __name__ == "__main__":
    image_path = "/content/drive/MyDrive/IR_Spectro/test_image.jpg"  # Replace with your test image path
    prediction = predict_image(image_path, model, class_names)
    print(f"🔍 **Predicted Functional Group:** {prediction}")
