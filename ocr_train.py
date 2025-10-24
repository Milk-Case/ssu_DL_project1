import os
import json
import numpy as np
from PIL import Image, ImageOps
from ocr_dataset import Label
from deep_convnet import DeepConvNet
from pydantic import ValidationError
import matplotlib.pyplot as plt

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m, self.v = {}, {}
        self.t = 0

    def update(self, params, grads):
        if not self.m:
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.t += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)

        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + self.eps)
            
def load_dataset(data_dir, target_size = (32,32)):
    images, labels, metas = [], [], []

    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            img_name = file.replace(".json", ".png")
            img_path = os.path.join(data_dir, img_name)
            json_path = os.path.join(data_dir, file)

            try:
                
                with open(json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                label = Label(**json_data)
            except ValidationError as e:
                continue

            img = np.array(Image.open(img_path).convert("L")) / 255.0
            
            age = int(label.Images.writer_age)
            sex = int(label.Images.writer_sex)
            
            # 각 bbox별 crop
            for box in label.bbox:
                x_min, x_max = min(box.x), max(box.x)
                y_min, y_max = min(box.y), max(box.y)
                
                crop = img[y_min:y_max, x_min:x_max]
                
                crop_img = Image.fromarray((crop * 255).astype(np.uint8))
                crop_padded = ImageOps.pad(crop_img, target_size, color = 0, centering = (0.5, 0.5))
                
                crop_resized = np.array(crop_padded)
                crop_resized = np.expand_dims(crop_resized, axis = 0)
                
                images.append(crop_resized)
                labels.append(box.data)
                metas.append([age, sex])
                
    images = np.array(images, dtype = np.float32)
    labels = np.array(labels)
    metas = np.array(metas, dtype = np.float32)

    return images, labels, metas

def encode_labels(labels_train, labels_test=None):
    words = sorted(set(labels_train))
    word_to_idx = {w: i for i, w in enumerate(words)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    train_idx = np.array([word_to_idx[l] for l in labels_train])
    test_idx = None
    if labels_test is not None:
        test_idx = np.array([
            word_to_idx[l] if l in word_to_idx else -1
            for l in labels_test
        ])

    return train_idx, test_idx, word_to_idx, idx_to_word


BASE_DIR = os.getcwd()
train_dir = os.path.join(BASE_DIR, "data", "train")
valid_dir = os.path.join(BASE_DIR, "data", "valid")

X_train, Y_train_text, _ = load_dataset(train_dir)
print(f"Train samples: {len(X_train)}")
X_valid, Y_valid_text, _ = load_dataset(valid_dir)
print(f"Validation samples: {len(X_valid)}")


Y_train, Y_test, w2i, i2w = encode_labels(Y_train_text, Y_valid_text)

net = DeepConvNet(input_dim=(1, 32, 32), output_size=len(w2i))

epochs = 10
batch_size = 8
optimizer = Adam(lr = 0.001)

train_losses, val_accs = [], []

for epoch in range(epochs):
    idx = np.random.permutation(len(X_train))
    X_train, Y_train = X_train[idx], Y_train[idx]

    for i in range(0, len(X_train), batch_size):
        x_batch = X_train[i:i+batch_size]
        t_batch = np.eye(len(w2i))[Y_train[i:i+batch_size]]

        grads = net.gradient(x_batch, t_batch)
        optimizer.update(net.params, grads)

    loss = net.loss(X_train[:batch_size], np.eye(len(w2i))[Y_train[:batch_size]])
    acc = net.accuracy(X_valid, np.eye(len(w2i))[Y_test], batch_size=len(X_valid))
    
    train_losses.append(loss)
    val_accs.append(acc)
    
    print(f"[Epoch {epoch+1}/{epochs}] Loss={loss:.4f}, Test Acc={acc:.4f}")

print("Training complete")
net.save_params("ocr_params.pkl")

plt.plot(range(1, epochs+1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, epochs+1), val_accs, marker='s', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Loss & Accuracy per Epoch (Adam Optimizer)')
plt.legend()
plt.grid(True)
plt.show()
