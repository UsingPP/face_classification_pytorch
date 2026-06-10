import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder  # 또는 기존 Dataset 클래스 사용
from collections import Counter
import res_facenet
import numpy as np
import matplotlib

matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import itertools
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score


# --- 예측 분포 시각화 출력 함수 추가 ---
def print_distribution(pred_counts: Counter, title: str) -> None:
    print(f"\n{'='*20} {title} 예측 분포 {'='*20}")
    total_preds = sum(pred_counts.values())
    if total_preds == 0:
        print("예측 데이터가 없습니다.")
        print(f"{'='*55}\n")
        return

    for idx, cls in enumerate(CLASS_NAMES):
        count = pred_counts[idx]
        ratio = count / total_preds
        bar = "■" * int(ratio * 30)
        print(f"  {cls:<6} : {count:3d}건 ({ratio*100:5.1f}%)  {bar}")
    print(f"{'='*55}\n")

def save_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix"):
    """
    실제 라벨과 예측 라벨을 비교하여 혼동 행렬을 그리고,
    하단에 Accuracy, Precision, Recall, F1-Score를 삽입하여 이미지로 저장합니다.
    """
    # 1. 지표 계산 (Macro Average 기준: 각 클래스별 지표의 평균)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    # 2. 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred)
    
    # 3. 콘솔에 상세 리포트 출력
    print(f"\n{'='*15} [{title}] 상세 평가 지표 {'='*15}")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    
    # [추가] 콘솔에 혼동 행렬을 텍스트 형태로 표 정렬하여 출력
    print(f"\n{'='*15} [{title}] 혼동 행렬 (텍스트) {'='*15}")
    max_len = max(len(str(c)) for c in classes)
    label_text = "True \\ Pred"
    # 상단 열(Predicted) 라벨 출력
    header = f"{label_text:<{max_len + 5}}" + "".join(f"{str(c):>12}" for c in classes)
    print(header)
    print("-" * len(header))
    # 각 행(True) 라벨과 값 출력
    for i, row in enumerate(cm):
        row_str = f"{str(classes[i]):<{max_len + 5}}" + "".join(f"{val:12d}" for val in row)
        print(row_str)
    print("=" * 60)

    # 4. 혼동 행렬 시각화 (이미지 생성)
    fig, ax = plt.subplots(figsize=(10, 9)) # 텍스트가 들어갈 하단 여백을 위해 세로 크기 증가
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    fig.colorbar(im, ax=ax)
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=12)

    # 행렬 안쪽 셀에 숫자 표기
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=11)

    ax.set_ylabel('True Label', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=14)
    
    # 5. 전체 지표(Metrics)를 이미지 하단에 텍스트 박스로 삽입
    metrics_text = (f"Accuracy: {acc:.4f}  |  "
                    f"Precision (Macro): {precision:.4f}  |  "
                    f"Recall (Macro): {recall:.4f}  |  "
                    f"F1-Score: {f1:.4f}")
    
    # figtext를 사용하여 차트 바깥 영역에 텍스트 배치
    plt.figtext(0.5, 0.02, metrics_text, ha="center", fontsize=12, 
                bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":8, "boxstyle":"round,pad=0.5"})

    # 하단 텍스트가 잘리지 않도록 여백 조절
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    filename = f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f">>> 시각화 결과가 서버에 '{os.path.abspath(filename)}' 파일로 저장되었습니다.\n")

CLASS_NAMES = ["Police", "Reporter", "Lawyer", "King", "Doctor", "Gangster", "hacker"]

# ── 전처리 파이프라인 (학습 때와 동일하게) ──────────────────────────────
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── 데이터로더 구성 ────────────────────────────────────────────────────
# 방법 A: torchvision ImageFolder (폴더 구조가 클래스명/이미지 형태일 때)
db_dataset   = ImageFolder(root="../faceInterection/arcface-pytorch/data/Datasets/faceImage",   transform=transform)
test_dataset = ImageFolder(root="../faceInterection/arcface-pytorch/data/Datasets/faceValidationImage", transform=transform)

# 방법 B: 기존 커스텀 Dataset 클래스가 있다면 아래처럼 교체
# db_dataset   = Dataset(opt.train_root, opt.train_list, phase='test', input_shape=opt.input_shape)
# test_dataset = Dataset(opt.test_root,  opt.test_list,  phase='test', input_shape=opt.input_shape)

db_loader   = DataLoader(db_dataset,   batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model921을 로드했다고 가정
model = res_facenet.models.model_921().to(device)
model.eval()


# ── 테스트1: DB Best Fit ───────────────────────────────────────────────
# 원리: 테스트 이미지와 DB의 모든 이미지를 1:1 코사인 유사도 비교,
#       가장 유사한 DB 샘플의 클래스를 예측값으로 사용
def test_best_fit(model, db_loader, test_loader, device):
    # 1단계: DB 이미지 전체를 특징 벡터로 변환해 저장
    db_features, db_labels = [], []
    with torch.no_grad():
        for images, labels in db_loader:
            feats = model(images.to(device))
            feats = F.normalize(feats, p=2, dim=1)  # 코사인 유사도를 위해 L2 정규화
            db_features.append(feats)
            db_labels.append(labels)

    db_features = torch.cat(db_features, dim=0)  # (DB 전체 수, 특징 차원)
    db_labels   = torch.cat(db_labels,   dim=0)  # (DB 전체 수,)

    # 2단계: 테스트 이미지 각각에 대해 가장 유사한 DB 샘플 탐색
    all_preds, all_targets = [], []
    pred_counts = Counter()

    with torch.no_grad():
        for images, labels in test_loader:
            feats = model(images.to(device))
            feats = F.normalize(feats, p=2, dim=1)

            # (배치 크기, DB 전체 수) 형태의 유사도 행렬 계산
            sim_matrix   = torch.mm(feats, db_features.T)
            best_idx     = sim_matrix.argmax(dim=1)        # 각 테스트 샘플의 최근접 DB 인덱스
            preds        = db_labels[best_idx].cpu().tolist()
            targets      = labels.tolist()

            all_preds.extend(preds)
            all_targets.extend(targets)
            for p in preds:
                pred_counts[p] += 1

    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    acc     = correct / len(all_targets)
    return correct, len(all_targets), acc, pred_counts, all_targets, all_preds


# ── 테스트2: DB Average Best Fit ──────────────────────────────────────
# 원리: 테스트 이미지와 DB의 코사인 유사도를 클래스별로 평균 낸 뒤,
#       평균 유사도가 가장 높은 클래스를 예측값으로 사용
#       → 클래스당 샘플 수가 불균등해도 안정적
def test_average_best_fit(model, db_loader, test_loader, device):
    # 1단계: DB 특징 추출 (test_best_fit과 동일)
    db_features, db_labels = [], []
    with torch.no_grad():
        for images, labels in db_loader:
            feats = model(images.to(device))
            feats = F.normalize(feats, p=2, dim=1)
            db_features.append(feats)
            db_labels.append(labels)

    db_features = torch.cat(db_features, dim=0)
    db_labels   = torch.cat(db_labels,   dim=0)

    # 2단계: 클래스별 평균 유사도로 예측
    all_preds, all_targets = [], []
    pred_counts = Counter()

    with torch.no_grad():
        for images, labels in test_loader:
            feats = model(images.to(device))
            feats = F.normalize(feats, p=2, dim=1)

            sim_matrix = torch.mm(feats, db_features.T)  # (배치, DB 전체 수)

            for i in range(len(feats)):
                # 클래스별로 유사도를 모아 평균을 냄
                # ave_sims[k] = 테스트 샘플 i와 클래스 k인 DB 샘플들의 평균 유사도
                ave_sims = torch.zeros(len(CLASS_NAMES), device=device)
                for k in range(len(CLASS_NAMES)):
                    mask = (db_labels.to(device) == k)
                    if mask.sum() > 0:
                        ave_sims[k] = sim_matrix[i][mask].mean()

                pred   = ave_sims.argmax().item()
                target = labels[i].item()

                all_preds.append(pred)
                all_targets.append(target)
                pred_counts[pred] += 1

    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    acc     = correct / len(all_targets)
    return correct, len(all_targets), acc, pred_counts, all_targets, all_preds


# ── 실행 ──────────────────────────────────────────────────────────────
print("===== 테스트1: DB Best Fit =====")
c1, t1, acc1, cnt1, tgt1, pred1 = test_best_fit(model, db_loader, test_loader, device)
print(f"정확도: {c1}/{t1} = {acc1:.4f}")
save_confusion_matrix(tgt1, pred1, CLASS_NAMES, "Test 1_DB Best Fit Confusion Matrix")

print("\n===== 테스트2: DB Average Best Fit =====")
c2, t2, acc2, cnt2, tgt2, pred2 = test_average_best_fit(model, db_loader, test_loader, device)
print(f"정확도: {c2}/{t2} = {acc2:.4f}")
save_confusion_matrix(tgt2, pred2, CLASS_NAMES, "Test 2_DB Average Best Fit Confusion Matrix")