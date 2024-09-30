# Topic4 Report

## 作業要求
1. 開通 server 的使用權限
2. 在 server 上的自己的帳號底下先從這個 URL 將 repo clone 下來(此步驟會需要大家在自己的 server 帳號底下 create ssh key 並且把 pubkey 上傳到 github，才能進行後續的 git clone 到 server 的動作)
3. 在 Github 上創建一個空的 repo 並同樣在 server 上 clone 到自己的帳號底下
4. 完成 transfer_learning_resnet34.ipynb 的 todo 部分
5. 將 transfer_learning_resnet34.ipynb 複製到自己的空 repo 裡面，並 push 到 github 端(注意是 push 到自己創建的 repo上，不是 push 到上方那個 repo URL)

## TO-DO
```python
import torchvision.models as models
from torchvision.models import resnet34

# Load the pre-trained ResNet34 model
model = resnet34(pretrained = True)
model.eval()
```
只有這裡引入 ResNet34

## Problems
寫作業時遇到以下問題：
1. [conda](https://anaconda.org/conda-forge/pytorch-model-summary) 下載的 torchsummary 無法被讀取（import 執行後會顯示 no module named torchsummary 之類的），只能使出下策，在 conda 環境用 pip 安裝
2. torchsummary 在使用上，如果以原始題目程式 `summary(model, (3, 224, 224))` 會出錯，只能以 `summary(model, (1, 3, 224, 224))` 執行，而且執行結果畫面和原始檔案好像不太一樣。不知道是不是我的 torchsummary 安裝出錯，但好像也不影響訓練