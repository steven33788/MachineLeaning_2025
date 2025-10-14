import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import xml.etree.ElementTree as ET
import warnings
import re 

# --- 設定中文字體 ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang HK', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 解決負號 '-' 顯示為方塊的問題
# ------------------------------

# Global Constants
TARGET_VALUE_C0 = -999.0
RANDOM_STATE = 42

# 資料設定 (與 XML 資料結構對應)
GRID_SIZE_LON = 67  # 經向格點數 (一列)
GRID_SIZE_LAT = 120 # 緯向格點數 (總列數)
GRID_RES = 0.03     # 經緯度解析度
NAMESPACE = {'cwa': 'urn:cwa:gov:tw:cwacommon:0.1'}


# --- 1. Data Loading and Transformation (Q1) ---

def load_and_transform_data(xml_file_path):
    """
    載入並解析 CWA 氣溫格點資料 XML，進行資料轉換。
    返回完整的特徵矩陣、分類標籤，以及過濾後的迴歸特徵和數值。
    
    當發生錯誤時，此函式現在會拋出異常 (FileNotFoundError 或 RuntimeError)，
    不再返回模擬數據。
    """
    try:
        # Load XML
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # 1. 提取地理資訊
        geo_info = root.find('cwa:dataset/cwa:GeoInfo', NAMESPACE)
        if geo_info is None:
            raise ValueError("無法找到 GeoInfo 標籤。XML 結構可能不符合預期。")
        
        lon_start = float(geo_info.find('cwa:BottomLeftLongitude', NAMESPACE).text)
        lat_start = float(geo_info.find('cwa:BottomLeftLatitude', NAMESPACE).text)
        
        # 2. 提取格點數據
        data_content_tag = root.find('cwa:dataset/cwa:Resource/cwa:Content', NAMESPACE)
        
        if data_content_tag is None or data_content_tag.text is None:
             raise ValueError("無法找到或提取格點資料 Content 內容。")
        
        grid_data_raw = data_content_tag.text
        
        # 核心修正：使用正規表達式提取所有數值 (支援浮點數和科學記號)
        pattern = r'[-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?'
        data_points_str = re.findall(pattern, grid_data_raw)

        if not data_points_str:
             raise ValueError("使用正規表達式無法從 GridData 中提取任何數值。")
            
        # 轉換為浮點數陣列
        temp_flat = np.array([float(d) for d in data_points_str])
        
        expected_size = GRID_SIZE_LON * GRID_SIZE_LAT
        if len(temp_flat) != expected_size:
            warnings.warn(f"警告：預期 {expected_size} 個數據點，但實際載入了 {len(temp_flat)} 個。", UserWarning)
        
        # 重塑為網格 (ROWS=緯度, COLS=經度)
        temp_grid = temp_flat.reshape(GRID_SIZE_LAT, GRID_SIZE_LON)

        # 3. 生成座標網格
        lon_values = np.linspace(lon_start, lon_start + (GRID_SIZE_LON - 1) * GRID_RES, GRID_SIZE_LON)
        lat_values = np.linspace(lat_start, lat_start + (GRID_SIZE_LAT - 1) * GRID_RES, GRID_SIZE_LAT)
        
        # 建立完整的 (Lon, Lat) 網格
        Lon_grid, Lat_grid = np.meshgrid(lon_values, lat_values)
        
        # 壓平數據和座標
        X_features = np.vstack([Lon_grid.ravel(), Lat_grid.ravel()]).T # (N, 2) features (Lon, Lat)
        Temp_flat = temp_grid.ravel()
        
        # --- 資料轉換 (Q1) ---
        
        # (a) 分類資料集: label = 1 (有效值), 0 (無效值)
        Y_cls = (Temp_flat != TARGET_VALUE_C0).astype(int)
        
        # (b) 迴歸資料集: 僅保留有效的溫度觀測值
        valid_indices = Y_cls == 1
        X_reg = X_features[valid_indices]
        Y_reg = Temp_flat[valid_indices]
        
        print(f"--- 氣象資料載入與轉換完成 ---")
        print(f"總格點數: {len(Temp_flat)}")
        print(f"有效溫度觀測值 (Label=1) 數量: {len(X_reg)}")
        print(f"無效值 (-999.0, Label=0) 數量: {np.sum(Y_cls == 0)}")
        
        return X_features, Y_cls, X_reg, Y_reg

    except FileNotFoundError:
        raise FileNotFoundError(f"錯誤: 檔案 {xml_file_path} 未找到。請確保檔案已上傳且名稱正確。")
    except Exception as e:
        raise RuntimeError(f"XML 解析或資料轉換過程中發生錯誤: {e}")

# --- 2. GDA Custom Implementation (Part Ia) ---

class GDAClassifier:
    """
    自定義 GDA 實現 (QDA 形式, 每個類別使用獨立的協方差矩陣 Sigma_k)。
    """
    def __init__(self):
        self.phi = None
        self.mu_0, self.mu_1 = None, None
        self.sigma_0, self.sigma_1 = None, None

    def fit(self, X, Y_cls):
        """通過最大似然估計 (MLE) 估計 GDA 參數。"""
        m = len(X)
        X_0 = X[Y_cls == 0]
        X_1 = X[Y_cls == 1]
        
        self.phi = len(X_1) / m
        self.mu_0 = np.mean(X_0, axis=0)
        self.mu_1 = np.mean(X_1, axis=0)
        
        # 使用無偏估計 np.cov(..., rowvar=False) 估計協方差矩陣
        self.sigma_0 = np.cov(X_0, rowvar=False) 
        self.sigma_1 = np.cov(X_1, rowvar=False)

    def _log_posterior_numerator(self, X, mu, sigma, phi):
        """計算 log(P(x|y=k) * P(y=k))。"""
        try:
            log_P_x_given_y = mvn.logpdf(X, mean=mu, cov=sigma, allow_singular=True)
        except np.linalg.LinAlgError:
            return np.full(len(X), -np.inf)

        return log_P_x_given_y + np.log(phi)

    def predict(self, X):
        """預測類別標籤 C(x)。"""
        log_prob_1 = self._log_posterior_numerator(X, self.mu_1, self.sigma_1, self.phi)
        log_prob_0 = self._log_posterior_numerator(X, self.mu_0, self.sigma_0, 1 - self.phi)
        
        predictions = (log_prob_1 > log_prob_0).astype(int)
        return np.atleast_1d(predictions)

    def get_decision_boundary_fn(self):
        """返回用於繪製決策邊界 (log-後驗概率分子差異為 0) 的函數。"""
        phi_0 = 1 - self.phi
        phi_1 = self.phi
        mu_0, mu_1 = self.mu_0, self.mu_1
        sigma_0, sigma_1 = self.sigma_0, self.sigma_1
        
        def boundary_fn(X_plot):
            log_prob_1 = mvn.logpdf(X_plot, mean=mu_1, cov=sigma_1) + np.log(phi_1)
            log_prob_0 = mvn.logpdf(X_plot, mean=mu_0, cov=sigma_0) + np.log(phi_0)
            return log_prob_1 - log_prob_0
        
        return boundary_fn

# --- 3. Main Execution and Model Training ---

if __name__ == '__main__':
    # 載入並轉換資料
    # 使用檔案的名稱作為路徑
    XML_FILE_PATH = "O-A0038-003.xml" 
    
    try:
        # X_cls: Full features for classification (Lon, Lat)
        # Y_cls: Classification labels (0 or 1)
        # X_reg: Filtered features for regression (Lon, Lat, where Temp != -999)
        # Y_reg: Filtered temperature values
        X_cls, Y_cls, X_reg, Y_reg = load_and_transform_data(XML_FILE_PATH)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n致命錯誤: 無法載入或處理氣象資料。程式終止。\n{e}")
        exit() # 終止程式執行

    # 劃分分類資料集的訓練/測試集 (用於 GDA)
    X_train_cls, X_test_cls, Y_train_cls, Y_test_cls = \
        train_test_split(X_cls, Y_cls, test_size=0.2, random_state=RANDOM_STATE, stratify=Y_cls)
    
    # --- (2) 簡單模型訓練 (作為背景資訊和 R(x) 的來源) ---
    
    # 簡單分類模型 (C_simple(x)): 使用 Logistic Regression
    C_simple_model = LogisticRegression(random_state=RANDOM_STATE)
    C_simple_model.fit(X_train_cls, Y_train_cls)
    simple_cls_accuracy = C_simple_model.score(X_test_cls, Y_test_cls)
    
    # 迴歸模型 (R(x)): 使用 Linear Regression (僅在有效數據上訓練)
    R_model = LinearRegression()
    R_model.fit(X_reg, Y_reg) # 在所有有效數據上訓練，以獲得最佳 R(x)

    print(f"\n--- 簡單模型訓練結果 (Q2) ---")
    print(f"R(x) 迴歸模型係數: {R_model.coef_}, 截距: {R_model.intercept_:.2f}")
    print(f"簡單分類模型 (LogisticReg) 測試集準確度: {simple_cls_accuracy * 100:.2f}%")
    print("-" * 45)
    
    # --- GDA 模型訓練與分析 (Part I) ---
    
    # a) 訓練 GDA 模型 (作為最終 C(x))
    gda_model = GDAClassifier()
    gda_model.fit(X_train_cls, Y_train_cls)
    
    # c) 性能報告：使用測試集準確度
    Y_cls_pred_test = gda_model.predict(X_test_cls)
    gda_accuracy = np.mean(Y_cls_pred_test == Y_test_cls)

    print(f"\n--- GDA 分類性能報告 (Part Ic) ---")
    print(f"測試集準確度 (Accuracy): {gda_accuracy * 100:.2f}% (使用 20% 測試集)")
    
    # --- Part II: Piecewise Regression Model h(x) ---
    
    def R(x):
        """迴歸模型 R(x): 線性迴歸預測 (使用訓練好的 R_model)"""
        return R_model.predict(np.atleast_2d(x))

    def h(x):
        """
        組合分段平滑函數 h(x)。 (Part IIa)
        h(x) = { R(x), if GDA(x) = 1
               { -999, if GDA(x) = 0
        """
        x_2d = np.atleast_2d(x)
        
        # 1. 執行 GDA 分類 C(x)
        classification = gda_model.predict(x_2d)
        
        # 2. 執行 R(x) 迴歸預測
        regression_output = R(x_2d)
        
        # 3. 根據分類結果構建 h(x) 輸出
        h_output = np.full(len(x_2d), TARGET_VALUE_C0)
        
        # classification 由於在 gda_model.predict 中已使用 np.atleast_1d 處理，
        # 因此 classification[0] 總是可用的，但這裡使用 classification == 1 
        # (預期是 array-to-array 比較) 也能正常運作，因為 x_2d 只有一行。
        indices_C1 = classification == 1
        h_output[indices_C1] = regression_output[indices_C1]
        
        return h_output
    
    # b) 應用與分段定義驗證
    print(f"\n--- 組合模型 h(x) 分段驗證 (Part IIb) ---")
    
    # 選擇幾個測試點進行驗證 (選取 GDA 預測為 0 和 1 的點)
    indices_pred_C0 = np.where(Y_cls_pred_test == 0)[0][:3]
    indices_pred_C1 = np.where(Y_cls_pred_test == 1)[0][:3]
    
    # 確保有足夠的樣本
    if len(indices_pred_C0) == 0 or len(indices_pred_C1) == 0:
        print("警告: 測試集中某類別樣本不足，驗證跳過。")
    else:
        X_verify = np.vstack([X_test_cls[indices_pred_C0], X_test_cls[indices_pred_C1]])
        
        print(f"{'Feature (Lon, Lat)':<18} | {'GDA C(x)':<9} | {'R(x) 預測值':<12} | {'h(x) 最終輸出':<12} | {'分段驗證'}")
        print("-" * 75)
        for x_point in X_verify:
            # 這裡 gda_model.predict 現在保證返回一個陣列，所以 [0] 是安全的
            cls_pred = gda_model.predict(x_point.reshape(1, -1))[0]
            reg_pred = R(x_point)[0]
            h_out = h(x_point)[0]
            
            if cls_pred == 1:
                status = "R(x) 成功應用" if np.isclose(h_out, reg_pred, atol=1e-4) else "錯誤"
            else:
                status = "-999 成功應用" if np.isclose(h_out, TARGET_VALUE_C0, atol=1e-4) else "錯誤"
            
            print(f"({x_point[0]:.2f}, {x_point[1]:.2f}) | {cls_pred:<9} | {reg_pred:<12.4f} | {h_out:<12.4f} | {status}")

    # --- d) 繪圖 ---
    
    # 設定繪圖網格
    x_min, x_max = X_cls[:, 0].min(), X_cls[:, 0].max()
    y_min, y_max = X_cls[:, 1].min(), X_cls[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                         np.linspace(y_min, y_max, 250))
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    
    
    plt.figure(figsize=(18, 7))
    
    # --- 5.1 Plot GDA Decision Boundary (Part Id) ---
    plt.subplot(1, 2, 1)
    
    # 計算 GDA 決策邊界等高線
    Z_gda = gda_model.get_decision_boundary_fn()(X_plot).reshape(xx.shape)
    plt.contour(xx, yy, Z_gda, levels=[0], colors='darkorange', linewidths=3, linestyles='--')
    
    # 繪製所有格點數據點
    scatter = plt.scatter(X_cls[:, 0], X_cls[:, 1], c=Y_cls, 
                          cmap='RdYlBu', edgecolor='k', alpha=0.5, s=5)
    
    plt.title('GDA 決策邊界 (Part Id) - 預測有效溫度區域', fontsize=16)
    plt.xlabel('經度 (Longitude)', fontsize=12)
    plt.ylabel('緯度 (Latitude)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    
    
    # --- 5.2 Plot Combined Function h(x) Behavior (Part IId) ---
    plt.subplot(1, 2, 2)
    
    # 計算 h(x) 在繪圖網格上的值
    Z_h = h(X_plot).reshape(xx.shape)
    
    # 將 -999 的區域遮罩 (Mask) 起來，以便用特殊的顏色顯示
    Z_h_masked = np.ma.masked_where(Z_h == TARGET_VALUE_C0, Z_h)
    cmap_h = plt.cm.plasma.copy()
    cmap_h.set_bad('lightgray', 1.0) # 將 -999 (即遮罩值) 設定為灰色
    
    # 繪製函數值表面 (僅顯示 R(x) 區域)
    contour_h = plt.contourf(xx, yy, Z_h_masked, levels=30, cmap=cmap_h, alpha=0.9)
    
    # 繪製 GDA 決策邊界 (分隔線)
    plt.contour(xx, yy, Z_gda, levels=[0], colors='black', linewidths=1.5, linestyles='-')
    
    # 添加顏色條 (僅適用於 R(x) 的輸出範圍)
    cbar = plt.colorbar(contour_h, label='h(x) 輸出值 (溫度 °C)')
    
    # 標記 -999 區域
    plt.scatter([], [], c='lightgray', marker='s', label='h(x) = -999 區域 (GDA 預測為無效值 C=0)')
    plt.legend(loc="upper right", framealpha=0.8, fontsize=10)
    
    plt.title('分段函數 h(x) 的行為 (Part IId)', fontsize=16) 
    plt.xlabel('經度 (Longitude)', fontsize=12)
    plt.ylabel('緯度 (Latitude)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.3)
    
    
    plt.tight_layout()
    plt.show()
