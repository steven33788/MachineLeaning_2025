import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, mean_squared_error

# --- 檔案與資料設定 ---
FILE_NAME = 'O-A0038-003.xml'
# XML 命名空間，從檔案開頭的 xmlns="urn:cwa:gov:tw:cwacommon:0.1" 取得
NAMESPACE = {'cwa': 'urn:cwa:gov:tw:cwacommon:0.1'}
# 資料資訊
GRID_SIZE_LON = 67  # 經向格點數 (一列)
GRID_SIZE_LAT = 120 # 緯向格點數 (一列/總列數)
GRID_RES = 0.03     # 經緯度解析度
LON_START = 120.00  # 左下角經度
LAT_START = 21.88   # 左下角緯度
MISSING_VALUE = -999.0 # 資料無效值 (以浮點數形式表示)

# --- (1) 資料轉換：解析與建立資料集 ---

def parse_xml_and_extract_data(file_path):
    """
    解析 CWA XML 檔案，提取格點資料 (GridData) 和地理資訊 (GeoInfo)。
    """
    try:
        # 1. 解析 XML 檔案
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 1. 尋找 GridData 內容
        griddata_element = root.find('.//cwa:GridData', NAMESPACE)
        
        if griddata_element is None or griddata_element.text is None:
            content_element = root.find('.//cwa:Content', NAMESPACE)
            
            if content_element is None or content_element.text is None:
                raise ValueError("在 XML 中找不到 cwa:GridData 或 cwa:Content 內容。請檢查 XML 結構。")
            else:
                grid_data_raw = content_element.text
        else:
            grid_data_raw = griddata_element.text
            
        # 2. 【核心修正】使用正規表達式提取所有數值
        # 模式: 匹配浮點數和科學記號 (例如: -999.0E+00, 25.100E+00)
        # re.findall 會返回所有匹配的數字字串列表，完全忽略分隔符、逗號和換行符。
        pattern = r'[-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?'
        data_points_str = re.findall(pattern, grid_data_raw)

        if not data_points_str:
             raise ValueError("使用正規表達式無法從 GridData 中提取任何數值。請檢查資料格式。")
            
        # 3. 將字串轉換為 NumPy 陣列
        data_points_float = np.array([float(d) for d in data_points_str])

        # 4. 檢查資料總數
        expected_size = GRID_SIZE_LON * GRID_SIZE_LAT
        if data_points_float.size != expected_size:
            print(f"警告: 預期的格點數為 {expected_size}, 實際讀取到 {data_points_float.size} 個。")

        # 5. 重塑為 120x67 的網格 (緯度 x 經度)
        # 緯向遞增 (列), 經向遞增 (欄)
        grid_array = data_points_float.reshape((GRID_SIZE_LAT, GRID_SIZE_LON))

        # 額外印出成功訊息
        print(f"✅ 成功提取 {grid_array.size} 個格點資料。")
        
        return grid_array

    except FileNotFoundError:
        print(f"錯誤: 檔案 '{file_path}' 不存在。請檢查路徑。")
        return None
    except ET.ParseError as e:
        print(f"錯誤: 無法解析 XML 檔案 '{file_path}': {e}")
        return None
    except ValueError as e:
        print(f"錯誤: 資料轉換失敗: {e}")
        return None


def create_datasets(grid_array):
    """
    根據格點資料和地理資訊建立分類與回歸資料集。
    """
    if grid_array is None:
        return None, None

    # 初始化經緯度網格
    # 緯度 (行) 範圍: 從 LAT_START 開始，向上遞增 120 次
    latitudes = LAT_START + np.arange(GRID_SIZE_LAT) * GRID_RES
    # 經度 (列) 範圍: 從 LON_START 開始，向右遞增 67 次
    longitudes = LON_START + np.arange(GRID_SIZE_LON) * GRID_RES

    # 建立經緯度網格矩陣
    # lon_mesh 的形狀是 (120, 67)，每一列都是 longitudes
    # lat_mesh 的形狀是 (120, 67)，每一行都是 latitudes
    lon_mesh, lat_mesh = np.meshgrid(longitudes, latitudes[::-1]) # latitudes[::-1] 使緯度從高往低排序 (即從頂部向下)
    
    # 將所有資料攤平 (Flatten)
    lons = lon_mesh.flatten()
    lats = lat_mesh.flatten()
    values = grid_array.flatten()

    # --- (a) 分類 (Classification) 資料集 ---
    
    # 規則：無效值 (-999.) -> label = 0；有效值 -> label = 1
    labels = np.where(values == MISSING_VALUE, 0, 1)

    df_classification = pd.DataFrame({
        'Longitude': lons,
        'Latitude': lats,
        'Label': labels
    })

    # --- (b) 回歸 (Regression) 資料集 ---

    # 規則：僅保留有效的溫度觀測值 (Value != -999.0)
    valid_mask = (values != MISSING_VALUE)
    
    df_regression = pd.DataFrame({
        'Longitude': lons[valid_mask],
        'Latitude': lats[valid_mask],
        'Value': values[valid_mask] # Value 為對應的攝氏溫度
    })
    
    print("\n--- 資料集建立結果 ---")
    print(f"分類資料集總筆數: {len(df_classification)}")
    print(f"回歸資料集 (有效值) 總筆數: {len(df_regression)} (無效值已剔除)")
    
    return df_classification, df_regression


# --- (2) 模型訓練：分類與回歸模型 ---

def train_and_evaluate_models(df_classification, df_regression):
    """
    使用建立的資料集分別訓練和評估分類與回歸模型。
    """
    # 檢查資料完整性
    if df_classification is None or df_regression is None:
        print("\n資料集建立失敗，跳過模型訓練。")
        return

    # 1. 分類模型訓練與評估 (預測是否為有效值)
    print("\n" + "="*50)
    print(" (2) 訓練與評估：分類模型 (Logistic Regression)")
    print("="*50)

    X_cls = df_classification[['Longitude', 'Latitude']]
    y_cls = df_classification['Label']

    # 分割訓練集與測試集
    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
        X_cls, y_cls, test_size=0.3, random_state=42, stratify=y_cls
    )

    # 訓練 Logistic Regression 模型
    model_cls = LogisticRegression(solver='liblinear', random_state=42)
    model_cls.fit(X_cls_train, y_cls_train)

    # 評估模型
    y_cls_pred = model_cls.predict(X_cls_test)
    print("\n分類模型 (預測有效/無效值) 評估報告:")
    print(classification_report(y_cls_test, y_cls_pred, target_names=['Invalid (0)', 'Valid (1)']))


    # 2. 回歸模型訓練與評估 (預測有效溫度值)
    print("\n" + "="*50)
    print(" (2) 訓練與評估：回歸模型 (Linear Regression)")
    print("="*50)

    X_reg = df_regression[['Longitude', 'Latitude']]
    y_reg = df_regression['Value']

    # 分割訓練集與測試集
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )

    # 訓練 Linear Regression 模型
    model_reg = LinearRegression()
    model_reg.fit(X_reg_train, y_reg_train)

    # 評估模型
    y_reg_pred = model_reg.predict(X_reg_test)
    mse = mean_squared_error(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n回歸模型 (預測溫度值) 評估結果:")
    print(f"模型截距 (Intercept): {model_reg.intercept_:.4f}")
    print(f"經度係數 (Longitude Coef): {model_reg.coef_[0]:.4f}")
    print(f"緯度係數 (Latitude Coef): {model_reg.coef_[1]:.4f}")
    print(f"均方誤差 (MSE): {mse:.4f}")
    print(f"均方根誤差 (RMSE): {rmse:.4f}")
    print(f"測試集實際溫度平均: {y_reg_test.mean():.2f}°C")
    print(f"測試集預測溫度平均: {y_reg_pred.mean():.2f}°C")


# --- 主要執行區塊 ---
if __name__ == "__main__":
    # 步驟 1: 解析 XML 並提取格點資料
    grid_data_array = parse_xml_and_extract_data(FILE_NAME)

    # 步驟 2: 建立分類和回歸資料集
    df_cls, df_reg = create_datasets(grid_data_array)
    
    # 步驟 3: 訓練和評估模型
    train_and_evaluate_models(df_cls, df_reg)