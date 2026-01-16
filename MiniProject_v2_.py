!pip install -q gdown

import gdown

file_id = "1nLeVdkiuVM7zljI1GLXKm9I4MEox3MHl"
url = f"https://drive.google.com/uc?id={file_id}"

gdown.download(url, "data.zip", quiet=False)

!unzip -q data.zip

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42

ds = tf.keras.utils.image_dataset_from_directory(
    "data",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=True,
    seed=SEED
)

print("class names:", ds.class_names)

images, labels = next(iter(ds))

plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(ds.class_names[labels[i]])
    plt.axis("off")
plt.show()

import os
import shutil
import random

# ===== 설정 =====
SOURCE_DIR = "data"          # 원본 데이터 (클래스별 폴더)
TARGET_DIR = "dataset"       # 분할 결과 저장 폴더

CLASSES = ["trash", "plastic", "paper", "metal", "glass", "cardboard"]

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42
random.seed(SEED)

# ===== 결과 폴더 생성 =====
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

# ===== 클래스별 랜덤 분할 =====
for cls in CLASSES:
    src_cls_dir = os.path.join(SOURCE_DIR, cls)
    images = [f for f in os.listdir(src_cls_dir) if not f.startswith(".")]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for img in train_imgs:
        shutil.copy(
            os.path.join(src_cls_dir, img),
            os.path.join(TARGET_DIR, "train", cls, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(src_cls_dir, img),
            os.path.join(TARGET_DIR, "val", cls, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(src_cls_dir, img),
            os.path.join(TARGET_DIR, "test", cls, img)
        )

    print(
        f"{cls}: total={total}, "
        f"train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}"
    )

print("✅ Dataset split completed")

!ls dataset

!ls dataset/train
!ls dataset/val
!ls dataset/test

# 분할된 dataset/train·val·test 폴더를 TensorFlow Dataset으로 로딩해 클래스/배치가 정상인지 확인한다

IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/val",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=False
)

print("class names:", train_ds.class_names)

# train 배치에서 이미지-라벨 매칭이 맞는지 시각적으로 확인한다

images, labels = next(iter(train_ds))

plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_ds.class_names[labels[i]])
    plt.axis("off")
plt.show()


# 학습 시에만 적용되는 이미지 증강 레이어를 정의해 과적합을 줄이고 일반화를 높인다

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ],
    name="data_augmentation"
)


# 원본 이미지와 증강된 이미지를 비교해 증강이 정상 동작하는지 시각적으로 확인한다

images, labels = next(iter(train_ds))
augmented_images = data_augmentation(images)

plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(augmented_images[i].numpy().astype("uint8"))
    plt.title(train_ds.class_names[labels[i]])
    plt.axis("off")
plt.show()


# ImageNet 사전학습된 MobileNetV2를 불러와 feature extractor로 사용한다

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # 먼저 고정


# 데이터 증강과 전처리를 포함한 6-클래스 분류 모델을 구성한다

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(6, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)


# 다중 클래스 분류를 위한 옵티마이저와 손실함수를 설정한다

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# [DEBUG] 파이프라인 연결 확인용 1 epoch 테스트 (초기 검증용)
# history = model.fit(train_ds, validation_data=val_ds, epochs=1)

# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=1
# )

# 과적합을 감시하고 가장 성능이 좋은 모델만 저장하기 위한 콜백을 설정한다

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="mobilenetv2_best.keras",
        monitor="val_loss",
        save_best_only=True
    )
]


# MobileNetV2 기반 6-클래스 분류 모델을 15 epoch 동안 정식 학습한다

EPOCHS = 15

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# 학습/검증 정확도와 손실 곡선을 시각화해 학습 상태를 점검한다

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.legend()
plt.title("Loss")

plt.show()


# 학습이 전혀 개입되지 않은 test set에서 최종 정확도를 평가한다

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

# test set 전체에 대해 모델의 예측 결과와 실제 라벨을 수집한다

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# 클래스 불균형 영향을 배제하기 위해 macro 평균 기준으로 정밀도, 재현율, F1을 계산한다

class_names = test_ds.class_names

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

# 클래스 간 오분류 패턴을 한눈에 보기 위해 confusion matrix를 시각화한다

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.show()


# ImageNet 사전학습된 EfficientNet-B0를 feature extractor로 사용한다

base_model_eff = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model_eff.trainable = False


# MobileNet과 동일한 구조(증강→전처리→GAP→Dense)로 EfficientNet 모델을 구성한다,증강/전처리 동일

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model_eff(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(6, activation="softmax")(x)

eff_model = tf.keras.Model(inputs, outputs)

# MobileNet과 동일한 옵티마이저와 손실 함수로 컴파일한다,동일 설정

eff_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

eff_model.summary()


# MobileNet과 동일한 기준으로 EarlyStopping과 ModelCheckpoint를 설정한다,동일 설정

callbacks_eff = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="efficientnetb0_best.keras",
        monitor="val_loss",
        save_best_only=True
    )
]


# EfficientNet-B0 기반 6-클래스 분류 모델을 정식 학습한다

EPOCHS = 15

eff_history = eff_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_eff
)


# EfficientNet 모델의 test set 성능을 정량적으로 평가한다

test_loss_eff, test_acc_eff = eff_model.evaluate(test_ds)
print(f"EfficientNet Test Accuracy: {test_acc_eff:.4f}")

# 1) test 예측 수집
y_true_eff = []
y_pred_eff = []

for images, labels in test_ds:
    probs = eff_model.predict(images, verbose=0)
    y_pred_eff.extend(np.argmax(probs, axis=1))
    y_true_eff.extend(labels.numpy())

y_true_eff = np.array(y_true_eff)
y_pred_eff = np.array(y_pred_eff)

class_names = test_ds.class_names

# 2) macro 포함 성능 리포트
report_eff = classification_report(
    y_true_eff,
    y_pred_eff,
    target_names=class_names,
    digits=4,
    output_dict=True
)
print(classification_report(y_true_eff, y_pred_eff, target_names=class_names, digits=4))

macro_precision_eff = report_eff["macro avg"]["precision"]
macro_recall_eff = report_eff["macro avg"]["recall"]
macro_f1_eff = report_eff["macro avg"]["f1-score"]

print("\n[Macro Avg]")
print(f"Precision: {macro_precision_eff:.4f}")
print(f"Recall   : {macro_recall_eff:.4f}")
print(f"F1-score : {macro_f1_eff:.4f}")

# 3) confusion matrix 시각화 (matplotlib만 사용)
cm_eff = confusion_matrix(y_true_eff, y_pred_eff)
class_names = test_ds.class_names

plt.figure(figsize=(8,6))
plt.imshow(cm_eff, cmap="Blues")
plt.title("Confusion Matrix - EfficientNetB0 (Test)")
plt.colorbar()

plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
plt.yticks(range(len(class_names)), class_names)

# 숫자 색상 대비 자동 조절
thresh = cm_eff.max() / 2
for i in range(cm_eff.shape[0]):
    for j in range(cm_eff.shape[1]):
        plt.text(
            j, i, cm_eff[i, j],
            ha="center", va="center",
            color="white" if cm_eff[i, j] > thresh else "black"
        )

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# MobileNetV2와 EfficientNetB0의 test 핵심 지표(accuracy/macro precision/recall/f1)를 한 표로 정리해 비교 해석의 근거를 만든다

summary_df = pd.DataFrame([
    {
        "Model": "MobileNetV2",
        "Test Accuracy": 0.8787,
        "Macro Precision": 0.8796,
        "Macro Recall": 0.8775,
        "Macro F1": 0.8776,
        "Weighted F1": 0.8786,
    },
    {
        "Model": "EfficientNet-B0",
        "Test Accuracy": 0.9261,
        "Macro Precision": 0.9257,
        "Macro Recall": 0.9265,
        "Macro F1": 0.9258,
        "Weighted F1": 0.9262,
    }
])

# 보기 좋게 소수점 4자리로 표시
summary_df.style.format({
    "Test Accuracy": "{:.4f}",
    "Macro Precision": "{:.4f}",
    "Macro Recall": "{:.4f}",
    "Macro F1": "{:.4f}",
    "Weighted F1": "{:.4f}",
})


# 두 모델의 클래스별 F1-score를 비교하고, EfficientNet의 개선 폭을 정량적으로 계산한다

class_f1_df = pd.DataFrame({
    "Class": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],

    "MobileNetV2_F1": [
        0.8504,  # cardboard
        0.9027,  # glass
        0.8831,  # metal
        0.8571,  # paper
        0.8523,  # plastic
        0.9201,  # trash
    ],

    "EfficientNetB0_F1": [
        0.9265,  # cardboard
        0.9405,  # glass
        0.9365,  # metal
        0.8996,  # paper
        0.9045,  # plastic
        0.9476,  # trash
    ],
})

# F1-score 개선 폭 계산
class_f1_df["F1_Improvement"] = (
    class_f1_df["EfficientNetB0_F1"] - class_f1_df["MobileNetV2_F1"]
)

# 보기 좋게 정렬 (개선 폭 큰 순)
class_f1_df = class_f1_df.sort_values(
    by="F1_Improvement", ascending=False
)

# 출력 포맷 정리
class_f1_df.style.format({
    "MobileNetV2_F1": "{:.4f}",
    "EfficientNetB0_F1": "{:.4f}",
    "F1_Improvement": "{:+.4f}",
})


# 두 모델의 클래스별 F1-score를 막대그래프로 비교해 EfficientNet의 개선 폭을 직관적으로 시각화한다

classes = class_f1_df["Class"]
mobilenet_f1 = class_f1_df["MobileNetV2_F1"]
efficientnet_f1 = class_f1_df["EfficientNetB0_F1"]

x = np.arange(len(classes))
width = 0.35

plt.figure(figsize=(10,5))

plt.bar(x - width/2, mobilenet_f1, width, label="MobileNetV2", color="#4C72B0")
plt.bar(x + width/2, efficientnet_f1, width, label="EfficientNet-B0", color="#55A868")

plt.xticks(x, classes, rotation=30)
plt.ylim(0.8, 1.0)
plt.ylabel("F1-score")
plt.title("Class-wise F1-score Comparison (Test Set)")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


# OpenVINO Model Optimizer 포함 개발 패키지 설치
!pip install -q openvino-dev

import openvino
print(openvino.__version__)

import openvino.tools.mo
print("Model Optimizer import OK")


# OpenVINO 변환을 위해 증강을 제거한 추론 전용 MobileNetV2 모델을 만들고 학습 가중치를 그대로 이식한다
# base_model은 이미 위에서 만든 MobileNetV2(include_top=False, imagenet, trainable=False) 그대로 사용
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(6, activation="softmax")(x)

mobilenet_infer_model = tf.keras.Model(inputs, outputs, name="mobilenetv2_infer")

# ✅ 핵심: 학습된 가중치를 그대로 복사 (재학습 아님)
mobilenet_infer_model.set_weights(model.get_weights())

mobilenet_infer_model.summary()


# OpenVINO 변환을 위해 증강을 제거한 추론 전용 EfficientNetB0 모델을 만들고 학습 가중치를 그대로 이식한다

# base_model_eff는 이미 위에서 만든 EfficientNetB0(include_top=False, imagenet, trainable=False) 그대로 사용
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base_model_eff(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(6, activation="softmax")(x)

efficientnet_infer_model = tf.keras.Model(inputs, outputs, name="efficientnetb0_infer")

# ✅ 핵심: 학습된 가중치를 그대로 복사 (재학습 아님)
efficientnet_infer_model.set_weights(eff_model.get_weights())

efficientnet_infer_model.summary()


# 추론 전용 MobileNetV2 모델을 평가용으로 compile 한다 (학습은 절대 다시 안 함)
mobilenet_infer_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 추론 전용 EfficientNet 모델도 동일하게 compile
efficientnet_infer_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# 추론 전용 모델이 학습 모델과 동일하게 동작하는지 test accuracy로 빠르게 검증한다

mn_loss, mn_acc = mobilenet_infer_model.evaluate(test_ds, verbose=0)
ef_loss, ef_acc = efficientnet_infer_model.evaluate(test_ds, verbose=0)

print(f"[Infer] MobileNetV2 Test Acc: {mn_acc:.4f}")
print(f"[Infer] EfficientNetB0 Test Acc: {ef_acc:.4f}")


# OpenVINO 변환을 위해 추론 전용 모델을 SavedModel로 export 한다

mobilenet_infer_model.export("mobilenetv2_infer_savedmodel")
efficientnet_infer_model.export("efficientnetb0_infer_savedmodel")


# MobileNetV2 → OpenVINO IR
!python -m openvino.tools.mo \
  --saved_model_dir mobilenetv2_infer_savedmodel \
  --output_dir ov_mobilenet \
  --input_shape [1,224,224,3] \
  --log_level INFO


# EfficientNet-B0 → OpenVINO IR
!python -m openvino.tools.mo \
  --saved_model_dir efficientnetb0_infer_savedmodel \
  --output_dir ov_efficientnet \
  --input_shape [1,224,224,3] \
  --log_level INFO


!ls ov_mobilenet
!ls ov_efficientnet


# OpenVINO IR 파일명(saved_model.xml/bin)에 맞춰 모델을 로드하고 CPU로 컴파일한다

from openvino.runtime import Core

ie = Core()

mn_ov_model = ie.read_model("ov_mobilenet/saved_model.xml")
ef_ov_model = ie.read_model("ov_efficientnet/saved_model.xml")

mn_compiled = ie.compile_model(mn_ov_model, "CPU")
ef_compiled = ie.compile_model(ef_ov_model, "CPU")


# OpenVINO 추론 속도 측정을 위해 batch_size=1 dataset을 다시 만든다

test_ds_ov = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(224, 224),
    batch_size=1,
    shuffle=False
)


mn_ov_mean, mn_ov_std = measure_ov_latency(mn_compiled, test_ds_ov)
ef_ov_mean, ef_ov_std = measure_ov_latency(ef_compiled, test_ds_ov)

print(f"[OV] MobileNetV2: {mn_ov_mean:.2f} ± {mn_ov_std:.2f} ms")
print(f"[OV] EfficientNetB0: {ef_ov_mean:.2f} ± {ef_ov_std:.2f} ms")


final_df = pd.DataFrame([
    {
        "Model": "MobileNetV2",
        "Test Accuracy": 0.8787,
        "OpenVINO Latency (ms)": 7.43,
        "Latency Std (ms)": 0.53,
        "Inference Speed": "Very Fast",
    },
    {
        "Model": "EfficientNet-B0",
        "Test Accuracy": 0.9261,
        "OpenVINO Latency (ms)": 16.35,
        "Latency Std (ms)": 1.33,
        "Inference Speed": "Moderate",
    }
])

final_df.style.format({
    "Test Accuracy": "{:.4f}",
    "OpenVINO Latency (ms)": "{:.2f}",
    "Latency Std (ms)": "{:.2f}",
})

