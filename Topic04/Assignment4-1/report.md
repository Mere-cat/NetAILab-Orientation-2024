# Topic4 Report
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