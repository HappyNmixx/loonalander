LunarLander DQN 抗風能力測試與訓練  (VS Code 適用)

檔案:
- LunarLander_DQN_Wind.ipynb  Jupyter Notebook 原始碼，最推薦使用
- LunarLander_DQN_Wind.py     Python 程式檔
- LunarLander_DQN_Wind.txt    文字版完整程式碼，無法開檔可複製

使用步驟:
1. 在 VS Code: File > Open File...，選擇 LunarLander_DQN_Wind.ipynb。
2. 若尚未安裝套件，在第一個 Code Cell 執行:
   %pip install "gymnasium[box2d]" torch numpy matplotlib pillow ipykernel
3. 如果原本已訓練完成，在舊 Notebook 執行:
   torch.save(model.state_dict(), "lunarlander_dqn.pth")
4. 把 .pth 放到目前 Notebook 的工作資料夾，或在第 0 節改 BASELINE_PATH 的完整路徑。
5. 由上到下執行 Notebook 的每個 Code Cell。
6. 結果會輸出到 lunarlander_wind_results 資料夾。

注意:
- 如果沒找到舊模型檔，程式會重新訓練 800 回合無風模型。
- 風力微調預設額外訓練 300 回合；先試跑可改為 10。
- GIF 會顯示在 Notebook 中，不需要開 pygame 視窗。
- 若 VS Code 顯示沒有 Kernel，請安裝 Python/Jupyter 擴充套件並選擇已安裝依賴的 Python 環境。

此程式尚未在使用者電腦的真實 Box2D 環境驗證訓練結果。
