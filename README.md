### Amazon reviews sentiment analysis using LSTM from scratch

dataset: https://www.kaggle.com/bittlingmayer/amazonreviews?select=train.ft.txt.bz2
<p align="center">
<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png" width="400"/>
</p>'

```bash
python train.py -b 500 -e 10 -l 1e-2 -f dataset
```

<p align="center">
<img src="./img/Training_result.png" width="500"/>
</p>

<p align="center">
<img src="./img/possitive.png" width="400"/>
</p>

<p align="center">
<img src="./img/negative.png" width="400"/>
</p>

<p align="center">
<img src="./img/fake_possitive.png" width="400"/>
</p>
