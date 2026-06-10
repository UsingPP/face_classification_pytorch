# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import os
import cv2
from torch.utils import data
from models import *
from data import Dataset
import torch
import numpy as np
import time
from config import Config
from torch.nn import DataParallel
from collections import Counter  # --- 예측 분포 추가 ---
import matplotlib
from PIL import Image
from torchvision import transforms as T
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import itertools
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score

CLASS_NAMES = ["Police", "Reporter", "Lawyer", "King", "Doctor", "Gangster", "hacker"]

def save_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix"):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*15} [{title}] 상세 평가 지표 {'='*15}")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    
    print(f"\n{'='*15} [{title}] 혼동 행렬 (텍스트) {'='*15}")
    max_len = max(len(str(c)) for c in classes)
    label_text = "True \\ Pred"

    header = f"{label_text:<{max_len + 5}}" + "".join(f"{str(c):>12}" for c in classes)
    print(header)
    print("-" * len(header))
    
    for i, row in enumerate(cm):
        row_str = f"{str(classes[i]):<{max_len + 5}}" + "".join(f"{val:12d}" for val in row)
        print(row_str)
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(10, 9)) 
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    fig.colorbar(im, ax=ax)
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=12)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=11)

    ax.set_ylabel('True Label', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=14)
    
    metrics_text = (f"Accuracy: {acc:.4f}  |  "
                    f"Precision (Macro): {precision:.4f}  |  "
                    f"Recall (Macro): {recall:.4f}  |  "
                    f"F1-Score: {f1:.4f}")
    
    plt.figtext(0.5, 0.02, metrics_text, ha="center", fontsize=12, 
                bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":8, "boxstyle":"round,pad=0.5"})

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    filename = f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f">>> 시각화 결과가 서버에 '{os.path.abspath(filename)}' 파일로 저장되었습니다.\n")

def visualize_features_with_tsne_server(device: torch.device, 
                                        model: torch.nn.Module, 
                                        dataloader: torch.utils.data.DataLoader, 
                                        title: str = "Face Features t-SNE") -> None:

    print(f"\n>>> '{title}' 특징 추출 및 t-SNE 연산 시작...")
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for data_input, label in dataloader:
            data_input = data_input.to(device)
            feature = model(data_input)
            
            if hasattr(torch.nn.functional, 'normalize'):
                feature = torch.nn.functional.normalize(feature, p=2, dim=1)
            
            all_features.append(feature.cpu().numpy())
            all_labels.append(label.numpy())

    features_np = np.concatenate(all_features, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)

    perplexity = min(30, max(5, len(features_np) // 2))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    features_2d = tsne.fit_transform(features_np)

    plt.figure(figsize=(10, 8))
    # 'tab10' 컬러맵을 사용하여 클래스별로 색상을 다르게 지정
    cmap = plt.cm.get_cmap('tab10', len(CLASS_NAMES))

    for i, class_name in enumerate(CLASS_NAMES):
        indices = np.where(labels_np == i)[0]
        if len(indices) == 0:
            continue
            
        plt.scatter(
            features_2d[indices, 0], 
            features_2d[indices, 1], 
            color=cmap(i), 
            label=class_name, # 여기서 영문 클래스 이름이 들어갑니다.
            alpha=0.8, 
            edgecolors='k', 
            s=60
        )

    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=11, frameon=True, shadow=True)
    
    plt.tight_layout()
    
    filename = f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f">>> 시각화 결과가 서버에 '{os.path.abspath(filename)}' 파일로 저장되었습니다.")
    

def print_result(img_path : str, result: dict) -> None:
    print(f"\n{'='*50}")
    print(f"이미지: {os.path.basename(img_path)}")
    print(f"{'─'*50}")
    sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    for rank, (cls, prob) in enumerate(sorted_result, 1):
        bar = "█" * int(prob * 30)
        print(f"  {rank}위 {cls:<10} {prob*100:6.2f}%  {bar}")
    print(f"{'='*50}")

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

def load_train_model(opt, device) :
    # Backbone
    if opt.backbone == 'resnet18':
        backbone = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        backbone = resnet34()
    elif opt.backbone == 'resnet50':
        backbone = resnet50()
    else:
        raise ValueError(f"Unknown backbone: {opt.backbone}")
 
    state_dict = torch.load(opt.load_model_path, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    backbone.load_state_dict(state_dict, strict=True)
    backbone.to(device)
    backbone = DataParallel(backbone)
    backbone.eval()


    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0,
                                     easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)
 
    fc_state = torch.load(opt.test_model_path, map_location='cpu')   # metric_fc pth
    fc_state = {k.replace('module.', ''): v for k, v in fc_state.items()}
    metric_fc.load_state_dict(fc_state, strict=True)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)
    metric_fc.eval()
 
    return backbone, metric_fc

def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return torch.from_numpy(image) 


def get_featurs(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))
            continue

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            output = model(data)
            output = output.data.cpu().numpy()

            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))
            # print(feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))



def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th
def test_best_fit_in_db(device, model: torch.nn.Module, test_loader, db_loader) :
    model.eval()

    db_features = []
    db_labels = []
    pred_counts = Counter()
    
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, label in db_loader :
            data = data.to(device)
            feature = model(data)
            feature = F.normalize(feature, p=2, dim=1)            
            db_features.append(feature)
            db_labels.append(label)

    correct = 0
    total = 0

    db_features = torch.cat(db_features, dim=0)
    db_labels = torch.cat(db_labels, dim=0)

    with torch.no_grad():
        for data, label in test_loader :
            data = data.to(device)
            feature = model(data)
            feature = F.normalize(feature, p=2, dim=1)
            
            cossim = torch.mm(feature, db_features.T)
            best_match_idx = torch.argmax(cossim, dim = 1)
            pred = db_labels[best_match_idx].to(device)
            
            for j in range(len(label)) :
                pred_cls_idx = pred[j].item()
                true_cls_idx = label[j].item()
                
                pred_counts[pred_cls_idx] += 1 
                
                all_targets.append(true_cls_idx)
                all_preds.append(pred_cls_idx)
                
                print(f"Label Class is {CLASS_NAMES[true_cls_idx]} and Pred Class is {CLASS_NAMES[pred_cls_idx]}.")
                print("So this prediction is a {0}". format("SUCCESS" if pred_cls_idx == true_cls_idx else "FAILURE"))
                print()
                correct += (pred[j] == label[j]).sum().item()
            total += len(label)

    acc = correct / total
    return correct, total, acc, pred_counts, all_targets, all_preds

def test_ave_best_fit_in_db(device : torch.device,
                            model : torch.nn.Module,
                            db_loader : data.DataLoader,
                            test_loader : data.DataLoader) :
    
    db_features = []
    db_labels = []
    correct = 0
    total = 0
    pred_counts = Counter() 
    
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, label in db_loader :
            data = data.to(device)
            feature = model(data)
            feature = F.normalize(input = feature, p = 2, dim = 1)
            db_features.append(feature)
            db_labels.append(label)

    db_features = torch.cat(db_features, dim=0)
    db_labels = torch.cat(db_labels, dim=0)

    with torch.no_grad() :
        for data, label in test_loader :
            data = data.to(device)
            feature = model(data)
            feature = F.normalize(input=feature, p = 2, dim = 1)

            cossim = torch.mm(feature, db_features.T)

            for i in range(len(cossim)) :
                ave_list = [[] for _ in range(len(CLASS_NAMES))]
                for j in range(len(cossim[0])) :
                    ave_list[db_labels[j].item()].append(cossim[i][j])

                ave_tensor = torch.zeros(len(CLASS_NAMES)).to(device)
                for k in range(len(CLASS_NAMES)):
                    if len(ave_list[k]) > 0:
                        ave_tensor[k] = torch.stack(ave_list[k]).mean(dim=0)

                pred = torch.argmax(ave_tensor).item()
                true_cls_idx = label[i].item()
                
                pred_counts[pred] += 1 
                
                all_targets.append(true_cls_idx)
                all_preds.append(pred)

                print(f"Label Class is {CLASS_NAMES[true_cls_idx]} and Pred Class is {CLASS_NAMES[pred]}.")
                print("So this prediction is a {0}".format("SUCCESS" if pred == true_cls_idx else "FAILURE"))
                print()

                correct += (pred == true_cls_idx)
                total += 1
                
    acc = correct/total
    return correct, total, acc, pred_counts, all_targets, all_preds

def test_trained_fc_layer(device : torch.device,
                          model : torch.nn.Module, 
                          metric_fc : torch.nn.Module, 
                          test_dataloader : data.DataLoader) :
    correct = 0
    total = 0
    pred_counts = Counter() 
    
    all_targets = []
    all_preds = []

    for ii, data in enumerate(test_dataloader) : 
        data_input, label = data
        data_input = data_input.to(device)
        label = label.to(device).long()

        with torch.no_grad() :
            feature = model(data_input)
            output = metric_fc(feature, label = None)

            prob = F.softmax(output, dim=1)          
            pred = torch.argmax(prob, dim=1)

            correct += (pred == label).sum().item()
            total += label.size(0)

            prob_np = prob.cpu().numpy()
            pred_np = pred.cpu().numpy()
            label_np = label.cpu().numpy()

            for j in range(len(label_np)) :
                pred_cls_idx = pred_np[j]
                true_cls_idx = label_np[j]
                
                pred_counts[pred_cls_idx] += 1 
                
                all_targets.append(true_cls_idx)
                all_preds.append(pred_cls_idx)

                sorted_idx = np.argsort(prob_np[j])[::-1]
                print(f"해당 클래스 : {CLASS_NAMES[true_cls_idx]}")
                for k in sorted_idx :
                    bar = '█' * int(prob_np[j][k]* 50)
                    print( f' {CLASS_NAMES[k]:20s} | {bar:50s} | {prob_np[j][k]: .4f}')
                print()         

    acc = correct / total
    return correct, total, acc, pred_counts, all_targets, all_preds
    
def lfw_test(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc

def prepare_single_image(image_path, input_shape=(1, 128, 128)):
    img = Image.open(image_path).convert('L')

    normalize = T.Normalize(mean=[0.5], std=[0.5])
    transforms = T.Compose([
        T.CenterCrop(input_shape[1:]), 
        T.ToTensor(),                 
        normalize                      
    ])

    img_tensor = transforms(img)  # 결과 크기: [1, 128, 128]

    img_tensor = img_tensor.unsqueeze(0)  
    
    return img_tensor

if __name__ == '__main__':
    opt = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    print("모델 로딩 중...")
    backbone, metric_fc = load_train_model(opt, device)
    print("모델 로딩 완료")

    # print("데이터셋 준비 중...")
    # db_dataset = Dataset(opt.train_root, opt.train_list, phase='test', input_shape=opt.input_shape)
    # db_dataloader = data.DataLoader(db_dataset,
    #                                 shuffle=False,
    #                               batch_size=opt.train_batch_size,
    #                               num_workers=opt.num_workers)
    
    # test_dataset = Dataset(opt.test_root, opt.test_list, phase='test', input_shape=opt.input_shape)
    # test_dataloader = data.DataLoader(test_dataset,
    #                                   shuffle=False,
    #                                   batch_size=opt.test_batch_size,
    #                                   num_workers=opt.num_workers)
    # print("데이터셋 준비 완료...\n")

    # # --- 테스트 및 혼동 행렬 시각화 ---
    # print("================ 1번 테스트 (DB Best Fit) ================")
    # t1_correct, t1_total, t1_acc, t1_counts, t1_targets, t1_preds = test_best_fit_in_db(device=device, model=backbone, test_loader=test_dataloader, db_loader=db_dataloader)
    # save_confusion_matrix(t1_targets, t1_preds, CLASS_NAMES, "Test 1_DB Best Fit Confusion Matrix")
    
    # print("\n================ 2번 테스트 (DB Average Best Fit) ================")
    # t2_correct, t2_total, t2_acc, t2_counts, t2_targets, t2_preds = test_ave_best_fit_in_db(device=device, model=backbone, db_loader=db_dataloader, test_loader=test_dataloader)
    # save_confusion_matrix(t2_targets, t2_preds, CLASS_NAMES, "Test 2_DB Average Best Fit Confusion Matrix")
    
    # print("\n================ 3번 테스트 (Trained FC Layer) ================")
    # t3_correct, t3_total, t3_acc, t3_counts, t3_targets, t3_preds = test_trained_fc_layer(device=device, model=backbone, metric_fc=metric_fc, test_dataloader=test_dataloader)
    # save_confusion_matrix(t3_targets, t3_preds, CLASS_NAMES, "Test 3_Trained FC Layer Confusion Matrix")
    
    # # --- 각 테스트 별 예측 분포 출력 ---
    # print_distribution(t1_counts, "1번 테스트 (DB Best Fit)")
    # print_distribution(t2_counts, "2번 테스트 (DB Average Best Fit)")
    # print_distribution(t3_counts, "3번 테스트 (Trained FC Layer)")

    # print("=== 각각의 테스트 정확도 ===")
    # print(f'Test1 최종 정확도: {t1_correct}/{t1_total} = {t1_acc:.4f}')
    # print(f'Test2 최종 정확도: {t2_correct}/{t2_total} = {t2_acc:.4f}')
    # print(f'Test3 최종 정확도: {t3_correct}/{t3_total} = {t3_acc:.4f}')

    # visualize_features_with_tsne_server(device=device, 
    #                              model=backbone, 
    #                              dataloader=test_dataloader, 
    #                              title="Test Dataset t-SNE Distribution")
    
    # testList = [t1_acc, t2_acc, t3_acc]
    # bestAcc = max(testList)
    # bestAccIdx = testList.index(bestAcc)
    # print(f"\nBest Test : Test {bestAccIdx+1} \t ACC : {testList[bestAccIdx]:.4f}")

    #====================사진 1장만 라벨 추출====================
    my_face = "/data/wsx1386/repos/faceInterection/arcface-pytorch/data/Datasets/faceValidationImage/해커/해커-1.webp"
    input = prepare_single_image(my_face).to(device)

    with torch.no_grad() :
        feature = backbone(input)
        output = metric_fc(feature, label = None)

        prob = F.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1)

        prob_np = prob.cpu().numpy()[0]
        pred_cls_idx = pred.cpu().item()

    print(f"예측 클래스 : {CLASS_NAMES[pred_cls_idx]}\n")
    print(f"{'클래스 이름':20s} | {'확률 그래프 (0% ~ 100%)':50s} | 확률값")
    print("-" * 85)

    sorted_idx = np.argsort(prob_np)[::-1]

    for k in sorted_idx:
        bar = '█' * int(prob_np[k] * 50)
        print(f'{CLASS_NAMES[k]:20s} | {bar:50s} | {prob_np[k]: .4f}')