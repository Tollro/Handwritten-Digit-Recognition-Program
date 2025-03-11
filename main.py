import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from PIL import Image

# 1. 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. 数据预处理
# 归一化到0-1范围并添加通道维度
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# 3. 构建CNN模型
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 4. 训练模型
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=15,
    validation_split=0.1
)

# 5. 评估模型
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f}")

# 6. 保存模型
model.save('mnist_cnn.h5')

# 7. 使用自定义图像进行预测
def predict_digit(img_path):
    # 加载模型
    model = keras.models.load_model('mnist_cnn.h5')
    
    # 处理图像
    img = Image.open(img_path).convert('L')  # 转为灰度
    img = img.resize((28, 28))  # 调整尺寸
    
    # 转换像素值（白底黑字 -> 黑底白字）
    img_array = 1 - np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # 预测
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return digit, confidence

# 使用示例
digit, confidence = predict_digit('my_digit.png')
print(f"Predicted digit: {digit} with {confidence*100:.2f}% confidence")