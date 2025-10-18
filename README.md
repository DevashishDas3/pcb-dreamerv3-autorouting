# 使用 DreamerV3 訓練 Freerouting 自動佈線 (JPype 方法)

本專案展示瞭如何透過 `JPype` 函式庫，將一個現成的 Java 應用程式 (`freerouting.jar`) 包裝成一個自訂的 `gymnasium` 強化學習環境，並使用 DreamerV3 框架進行訓練。

## 專案設定

### 安裝前置套件

```bash
pip install -r requirements.txt

### 開始訓練（描述）

當您已經將所有專案檔案（包含基於分析產生的 `freerouting_env.py`）放到正確位置後，即可啟動 DreamerV3 的訓練流程：

1. 開啟終端機  
    - 在 Windows 使用「命令提示字元」或「PowerShell」，在 macOS/Linux 使用 Terminal。

2. 切換到專案根目錄  
    - 使用 cd 指令切換到專案最外層資料夾（範例資料夾名稱：dreamerv3-freerouting-env）：
    ```bash
    cd path/to/your/projects/dreamer-Autorouting
    ```
    - 確保您在此目錄下執行訓練指令，才能正確找到 dreamerv3 程式庫與自訂的 env 模組。

3. 執行訓練指令  
    - 在終端機中輸入：
    ```bash
    python dreamerv3/train.py --configs defaults freerouting
    ```
    指令拆解：
    - `python dreamerv3/train.py`：執行 DreamerV3 的主要訓練腳本。  
    - `--configs`：告訴 train.py 接受後續一或多個設定區塊。  
    - `defaults freerouting`：先載入 `defaults`（一般預設參數），再載入並覆寫 `freerouting`（專案客製設定，例如使用 Freerouting-v0 環境）。

4. 觀察訓練過程  
    - 訓練啟動時會在終端機顯示日誌（可能包含 JPype / JVM 啟動訊息）。  
    - 期望訊息例子：創建環境（Creating Gym environment: Freerouting-v0）、訓練步數、reward、loss 等。  
    - 訓練結果會儲存在專案根目錄下的 `logdir`（包含模型權重、評估影片和 TensorBoard 日誌）。

注意事項：
- 確保 `freerouting.jar` 與 `freerouting_env.py` 的 API 配置與 JPype 啟動參數正確，否則可能在 JVM 啟動或環境建立時出現錯誤。  
- 若遇到錯誤，可先檢查 Python 相依套件、JPype 版本以及 JVM 路徑設定。