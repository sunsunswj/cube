import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import pickle
import json

def draw_line_chart(y_true, y_pred, path):
    plt.title('Line Chart')
    plt.plot(y_true.values, alpha=0.8)
    plt.plot(y_pred, alpha=0.8)
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def inference(save_model_dir, inference_dataset, feature_columns):
    save_model_path = os.path.join(save_model_dir, 'lr_model.pkl')

    with open(save_model_path, 'rb') as file:
        model = pickle.load(file)
        df = pd.read_csv(inference_dataset)
        feature_columns = [x.strip() for x in feature_columns.split(',') if x.strip()]
        X = df[feature_columns]  # 特征变量
        y_pred = model.predict(X)
        print('预测值')
        print(y_pred)
        y_df = pd.DataFrame(y_pred, columns=['y'])

        result = pd.concat([X, y_df], axis=1)

        result.to_csv(os.path.join(save_model_dir, 'inference_result.csv'), index=False, header=True)

def val(save_model_dir, val_dataset, label_columns, feature_columns):
    save_model_path = os.path.join(save_model_dir, 'lr_model.pkl')
    save_val_path = os.path.join(save_model_dir, 'val_result.json')

    with open(save_model_path, 'rb') as file:
        model = pickle.load(file)
        df = pd.read_csv(val_dataset)
        label_columns = [x.strip() for x in label_columns.split(',') if x.strip()]
        feature_columns = [x.strip() for x in feature_columns.split(',') if x.strip()]

        X_val = df[feature_columns]  # 特征变量
        y_val = df[label_columns]  # 目标变量

        y_pred = model.predict(X_val)

        # 评估模型
        print('预测值')
        print(y_pred)

        draw_line_chart(y_val, y_pred, os.path.join(save_model_dir, 'line_chart.png'))

        R2 = r2_score(y_pred, y_val)
        print('R2:', R2)

        train_test = json.load(open(save_val_path, "r")) if os.path.exists(save_val_path) else {}
        train_test.update({"R2": R2})
        json.dump(train_test, open(save_val_path, "w"))

        metrics = [
            {
                "metric_type": "image",
                "describe": "验证集与事实数据的折线表示",
                "image": os.path.join(save_model_dir, 'line_chart.png')
            }
        ]
        json.dump(metrics, open(os.path.join(save_model_dir, 'metric.json'), mode='w'))

def train(save_model_dir, train_dataset, label_columns, feature_columns, model_params):
    # 读取数据
    df = pd.read_csv(train_dataset)
    label_columns = [x.strip() for x in label_columns.split(',') if x.strip()]
    feature_columns = [x.strip() for x in feature_columns.split(',') if x.strip()]

    # 处理数据
    X = df[feature_columns]  # 特征变量
    y = df[label_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 划分训练集和测试集

    # 训练lr模型
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # 计算R2
    R2_test = r2_score(reg.predict(X_test), y_test)
    R2_train = r2_score(reg.predict(X_train), y_train)

    os.makedirs(save_model_dir, exist_ok=True)

    save_model_path = os.path.join(save_model_dir, 'lr_model.pkl')
    with open(save_model_path, 'wb') as file:
        pickle.dump(reg, file)

    file = open(os.path.join(save_model_dir, 'val_result.json'), "w")
    file.write(json.dumps({"train_R2": R2_train, "test_R2": R2_test}))
    file.close()